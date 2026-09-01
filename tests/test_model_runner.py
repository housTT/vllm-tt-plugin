# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for ``TTModelRunner`` (non-lane path)."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_input import TTModelInput, slice_tt_sampling_params
from vllm_tt_plugin.model_runner import TTModelRunner

# region Constants
VOCAB_SIZE = 64
BLOCK_SIZE = 16
MAX_MODEL_LEN = 32
MAX_NUM_SEQS = MAX_NUM_REQS = 4
DP_SIZE = 1
SAMPLED_TOKEN_ID = 42
# endregion Constants


# region Test helpers


def _batch_with_one_request(
    prompt_len: int,
    output_len: int,
    num_computed_tokens: int,
) -> tuple[InputBatch, CachedRequestState]:
    """Creates a batch with a single request, for testing purposes."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    request = CachedRequestState(
        req_id="r",
        prompt_token_ids=list(range(prompt_len)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=list(range(prompt_len, prompt_len + output_len)),
    )
    batch.add_request(request)
    return batch, request


def _fake_runner(batch: InputBatch, request: CachedRequestState) -> SimpleNamespace:
    """Creates a fake runner with the given batch and request, for testing purposes."""
    return SimpleNamespace(
        input_batch=batch,
        requests={"r": request},
        _output_tokens_per_step=1,
        tt_per_lane_max_num_seqs=MAX_NUM_SEQS,
        tt_data_parallel_size=DP_SIZE,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        model_config=SimpleNamespace(is_multimodal_model=False),
        check_perform_device_sampling=lambda **_: False,
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _sampling_params_for_padded_decode=lambda params, req_indices, n: params,
        _decode_layout_changed_since_last_decode=False,
        _build_host_generators=TTModelRunner._build_host_generators,
    )


def _batch_with_sampling_params(
    sampling_params: SamplingParams,
    *,
    vocab_size: int = VOCAB_SIZE,
) -> tuple[InputBatch, CachedRequestState]:
    generator = None
    if sampling_params.seed is not None:
        generator = torch.Generator().manual_seed(sampling_params.seed)
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=vocab_size,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    request = CachedRequestState(
        req_id="r",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=sampling_params,
        generator=generator,
        block_ids=([0],),
        num_computed_tokens=1,
        output_token_ids=[],
    )
    batch.add_request(request)
    return batch, request


def _device_sampling_runner(
    batch: InputBatch,
    model_capabilities: dict | None,
) -> SimpleNamespace:
    model = SimpleNamespace()
    if model_capabilities is not None:
        model.model_capabilities = model_capabilities
    return SimpleNamespace(
        sample_on_device_mode="all",
        num_devices=4,
        tt_data_parallel_size=1,
        input_batch=batch,
        model=model,
        model_config=SimpleNamespace(logits_processors=None),
        supports_topk_logprobs=True,
    )


def _prepare(runner, *rows):
    """Run _prepare_model_inputs for cached rows of
    (req_id, num_scheduled, num_computed, num_output)."""
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = {r: s for r, s, _, _ in rows}
    out.total_num_scheduled_tokens = sum(s for _, s, _, _ in rows)
    out.scheduled_cached_reqs = CachedRequestData(
        req_ids=[r for r, *_ in rows],
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in rows],
        all_token_ids={},
        new_block_ids=[None for _ in rows],
        num_computed_tokens=[c for _, _, c, _ in rows],
        num_output_tokens=[o for *_, o in rows],
    )
    return TTModelRunner._prepare_model_inputs(runner, out, None)


# endregion Test helpers

# region Device sampling policy


@pytest.mark.parametrize("top_k", [-1, VOCAB_SIZE])
def test_model_top_k_capability_routes_unrestricted_sampling_to_host(top_k: int):
    batch, request = _batch_with_sampling_params(
        SamplingParams(temperature=1.0, top_p=1.0, top_k=top_k)
    )
    runner = _device_sampling_runner(
        batch,
        {"max_device_sampling_top_k": 32},
    )

    assert request.sampling_params.top_k == top_k
    assert batch.sampling.top_k[0].item() == VOCAB_SIZE
    assert not TTModelRunner.check_perform_device_sampling(
        runner,
        is_decode=True,
        has_structured_outputs=False,
    )


@pytest.mark.parametrize("top_k", [1, 32])
def test_model_top_k_capability_keeps_supported_sampling_on_device(top_k: int):
    batch, _ = _batch_with_sampling_params(
        SamplingParams(temperature=0.0 if top_k == 1 else 1.0, top_k=top_k)
    )
    runner = _device_sampling_runner(
        batch,
        {"max_device_sampling_top_k": 32},
    )

    assert TTModelRunner.check_perform_device_sampling(
        runner,
        is_decode=True,
        has_structured_outputs=False,
    )


def test_model_without_top_k_capability_keeps_existing_device_policy():
    batch, _ = _batch_with_sampling_params(
        SamplingParams(temperature=1.0, top_p=1.0, top_k=-1)
    )
    runner = _device_sampling_runner(batch, None)

    assert TTModelRunner.check_perform_device_sampling(
        runner,
        is_decode=True,
        has_structured_outputs=False,
    )


def test_supported_top_k_preserves_device_penalties_and_seed():
    batch, _ = _batch_with_sampling_params(
        SamplingParams(
            temperature=1.0,
            top_k=32,
            seed=42,
            presence_penalty=0.25,
            frequency_penalty=0.5,
            repetition_penalty=1.125,
        )
    )
    runner = _device_sampling_runner(
        batch,
        {"max_device_sampling_top_k": 32},
    )

    assert TTModelRunner.check_perform_device_sampling(
        runner,
        is_decode=True,
        has_structured_outputs=False,
    )
    assert batch.sampling.seed[0].item() == 42
    assert batch.sampling.presence_penalty[0].item() == 0.25
    assert batch.sampling.frequency_penalty[0].item() == 0.5
    assert batch.sampling.repetition_penalty[0].item() == 1.125


def test_host_fallback_receives_unrestricted_sampling_semantics():
    batch, request = _batch_with_sampling_params(
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            seed=42,
        )
    )
    captured = {}

    def capture_host_sampling(*, logits, sampling_metadata):
        captured["logits"] = logits
        captured["sampling_metadata"] = sampling_metadata
        return SimpleNamespace(
            sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
            logprobs_tensors=None,
        )

    sampling_params = slice_tt_sampling_params(batch.sampling, [0])
    model_input = TTModelInput(
        input_tokens=torch.tensor([[1]], dtype=torch.int32),
        input_positions=torch.tensor([0], dtype=torch.int32),
        prompt_lens=None,
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        block_tables_per_group=[torch.zeros((1, 1), dtype=torch.int32)],
        block_tables_per_layer=None,
        unpadded_batch_size=1,
        tt_sampling_params=sampling_params,
        multi_modal_kwargs={},
        perform_device_sampling=False,
        grammar_bitmask=[None],
        logitsprocs_list=[None],
        bad_words_token_ids_list=[{}],
        allowed_token_ids_mask_list=[None],
        generators_list=[dict(batch.sampling.generators)],
        max_num_logprobs=[None],
    )
    runner = SimpleNamespace(
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_lane_mode=False,
        tt_per_lane_max_num_seqs=MAX_NUM_REQS,
        vocab_size=VOCAB_SIZE,
        host_sampler=capture_host_sampling,
    )
    logits = torch.linspace(-1.0, 1.0, VOCAB_SIZE).reshape(1, 1, -1)

    sampled, logprobs = TTModelRunner._get_output_tokens(
        runner,
        tt_out=logits,
        tt_log_probs=None,
        sampling_params=sampling_params,
        model_input=model_input,
        batch_size_per_dp=[1],
        perform_device_sampling=False,
        is_decode=True,
    )

    metadata = captured["sampling_metadata"]
    assert request.sampling_params.top_k == -1
    assert metadata.top_k.tolist() == [VOCAB_SIZE]
    assert metadata.top_p.tolist() == [1.0]
    assert metadata.temperature.tolist() == [1.0]
    assert metadata.generators[0].initial_seed() == 42
    assert torch.equal(captured["logits"], logits[:, -1, :])
    assert sampled[0].tolist() == [[7]]
    assert logprobs == [None]


# endregion Device sampling policy

# region Prefill classification


@pytest.mark.parametrize(
    "num_computed_tokens, intermediate_prefill_mask",
    [(0, True), (2, True), (5, True), (6, False)],
)
def test_cached_chunked_prefill_classification(
    num_computed_tokens: int,
    intermediate_prefill_mask: bool,
):
    """Classify a cached chunked-prefill continuation and its boundary."""
    batch, request = _batch_with_one_request(
        prompt_len=8,
        output_len=0,
        num_computed_tokens=num_computed_tokens,
    )
    runner = _fake_runner(batch, request)
    runner._decode_layout_change_removal_only = True
    num_scheduled_tokens = 2

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": num_scheduled_tokens}
    scheduler_output.total_num_scheduled_tokens = num_scheduled_tokens
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[num_computed_tokens],
        num_output_tokens=[0],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    assert model_input.prompt_lens.tolist() == [
        num_computed_tokens + num_scheduled_tokens
    ]
    assert model_input.intermediate_prefill_mask.tolist() == [intermediate_prefill_mask]
    assert runner._decode_layout_change_removal_only is False


def test_resumed_replay_past_prompt_length_remains_prefill():
    """Resumed replay past the prompt length remains prefill."""
    # 4-token prompt, 6 previously generated tokens (num_tokens=10). Resumed
    # from preemption and already replayed its first chunk up to position 5,
    # which is past the prompt but short of num_tokens: still mid-replay.
    num_computed_tokens = 5
    batch, request = _batch_with_one_request(
        prompt_len=4,
        output_len=6,
        num_computed_tokens=num_computed_tokens,
    )
    runner = _fake_runner(batch, request)
    num_scheduled_tokens = 3

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": num_scheduled_tokens}
    scheduler_output.total_num_scheduled_tokens = num_scheduled_tokens

    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[num_computed_tokens],
        num_output_tokens=[6],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    # A decode step would set `prompt_lens=None` and sample a token.
    # This step must still be a prefill chunk (position 5..8 of a 10-token replay).
    assert model_input.prompt_lens is not None
    assert model_input.prompt_lens.tolist() == [
        num_computed_tokens + num_scheduled_tokens
    ]
    assert model_input.intermediate_prefill_mask.tolist() == [True]


# endregion Prefill classification

# region Decode input construction


@pytest.mark.parametrize("output_len", [0, 1, 2, 5])
def test_completed_cached_request_builds_decode_input(output_len: int):
    """Classify a cached request that has computed everything but its last token."""
    prompt_len = 8
    num_tokens = prompt_len + output_len
    num_computed_tokens = max(prompt_len, num_tokens - 1)
    batch, request = _batch_with_one_request(
        prompt_len=prompt_len,
        output_len=output_len,
        num_computed_tokens=num_computed_tokens,
    )
    runner = _fake_runner(batch, request)
    runner._decode_layout_changed_since_last_decode = True
    runner._decode_layout_change_removal_only = True
    num_scheduled_tokens = 1

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": num_scheduled_tokens}
    scheduler_output.total_num_scheduled_tokens = num_scheduled_tokens
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[num_computed_tokens],
        num_output_tokens=[output_len],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    assert model_input.reset_batch is True
    assert model_input.removal_only_reset is True
    assert model_input.prompt_lens is None
    assert model_input.input_positions.tolist() == [num_tokens - 1] + [-1] * (
        MAX_NUM_SEQS - 1
    )
    assert model_input.input_tokens.shape == (MAX_NUM_SEQS, 1)
    assert model_input.input_tokens[0, 0] == batch.token_ids_cpu[0, num_tokens - 1]


def test_final_one_token_prompt_chunk_stays_prefill():
    """Even a one-token prompt remainder is prefill work: the prefill path
    owns chunk bookkeeping (intermediate mask, vision rope state)."""
    batch, request = _batch_with_one_request(8, 0, num_computed_tokens=7)

    model_input = _prepare(_fake_runner(batch, request), ("r", 1, 7, 0))

    assert model_input.prompt_lens.tolist() == [8]
    assert model_input.intermediate_prefill_mask.tolist() == [False]


def test_mixed_prefill_batch_keeps_scheduled_decode_row():
    """A scheduled row must never be dropped from a prefill step: the output
    row count desyncs and the dropped token is never rescheduled."""
    batch, chunking = _batch_with_one_request(8, 0, num_computed_tokens=2)
    steady = CachedRequestState(
        req_id="r2",
        prompt_token_ids=list(range(8)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([1],),
        num_computed_tokens=8,
        output_token_ids=[100],
    )
    batch.add_request(steady)

    model_input = _prepare(
        _fake_runner(batch, chunking), ("r", 2, 2, 0), ("r2", 1, 8, 1)
    )

    # Chunk end 2+2=4 (mid-prompt) and the steady row's 8+1=9 both stay.
    assert sorted(model_input.prompt_lens.tolist()) == [4, 9]


def test_steady_block_step_builds_decode_input():
    """One whole canvas outstanding is the block steady state; dispatching it
    as prompt work re-encodes the entire session per canvas."""
    batch, request = _batch_with_one_request(8, 16, num_computed_tokens=8)
    runner = _fake_runner(batch, request)
    runner._output_tokens_per_step = 16

    model_input = _prepare(runner, ("r", 16, 8, 16))

    assert model_input.prompt_lens is None


def test_block_resume_replay_past_one_canvas_remains_prefill():
    """More than one canvas outstanding is uncomputed replay history."""
    batch, request = _batch_with_one_request(8, 16, num_computed_tokens=8)
    runner = _fake_runner(batch, request)
    runner._output_tokens_per_step = 8

    model_input = _prepare(runner, ("r", 8, 8, 16))

    assert model_input.prompt_lens.tolist() == [16]


# endregion Decode input construction

# region Output state


def test_apply_sampled_token_updates_request_state():
    try:
        import torch
    except ImportError:
        pytest.xfail("torch is required for sampled-token tensor inputs")

    prompt_len = 8
    batch, request = _batch_with_one_request(
        prompt_len=prompt_len,
        output_len=0,
        num_computed_tokens=prompt_len,
    )
    runner = _fake_runner(batch, request)
    runner.model_config.max_model_len = MAX_MODEL_LEN
    runner._apply_sampled_tokens_to_state = lambda **kwargs: (
        TTModelRunner._apply_sampled_tokens_to_state(runner, **kwargs)
    )
    runner._build_runner_output = lambda **kwargs: TTModelRunner._build_runner_output(
        runner, **kwargs
    )

    output = TTModelRunner.apply_and_build_runner_output(
        runner,
        sampled_token_ids=torch.tensor([[SAMPLED_TOKEN_ID]], dtype=torch.int32),
    )

    assert batch.num_tokens[0] == prompt_len + 1
    assert batch.token_ids_cpu[0, prompt_len] == SAMPLED_TOKEN_ID
    assert request.output_token_ids == [SAMPLED_TOKEN_ID]
    assert output.req_ids == ["r"]
    assert output.sampled_token_ids == [[SAMPLED_TOKEN_ID]]


# endregion Output state
