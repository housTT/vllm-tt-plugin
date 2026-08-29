# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_tt_plugin.input_batch import SEED_NONE_SENTINEL
from vllm_tt_plugin.model_input import TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner


def _runner(
    *,
    temperature=0.0,
    top_k=0,
    penalties=False,
    seed=SEED_NONE_SENTINEL,
    force_host_seeded_sampling=False,
):
    sampling = SimpleNamespace(
        temperature=torch.tensor([temperature]),
        top_k=torch.tensor([top_k]),
        seed=torch.tensor([seed]),
        bad_words_token_ids=[],
        has_active_logitsprocs=lambda: False,
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        no_penalties=not penalties,
        no_allowed_token_ids=True,
        sampling=sampling,
        max_num_logprobs=None,
    )
    runner = object.__new__(TTModelRunner)
    runner.sample_on_device_mode = "all"
    runner.num_devices = 2
    runner.tt_data_parallel_size = 1
    runner.supports_topk_logprobs = False
    runner.supports_device_sampling_penalties = False
    runner.device_sampling_max_top_k = 32
    runner.force_host_seeded_sampling = force_host_seeded_sampling
    runner.model_config = SimpleNamespace(logits_processors=[])
    runner.input_batch = input_batch
    return runner


def test_greedy_vllm_top_k_sentinel_remains_on_device():
    runner = _runner(temperature=0.0, top_k=0)
    assert runner.check_perform_device_sampling(False, False)


def test_unsupported_random_top_k_uses_host_compatibility():
    runner = _runner(temperature=0.8, top_k=0)
    assert not runner.check_perform_device_sampling(False, False)
    runner.input_batch.sampling.top_k.fill_(32)
    assert runner.check_perform_device_sampling(False, False)


def test_unsupported_penalties_use_host_compatibility():
    runner = _runner(temperature=0.8, top_k=20, penalties=True)
    assert not runner.check_perform_device_sampling(False, False)


def test_model_opt_in_keeps_explicit_random_seed_on_host_across_cohort_order():
    """Seeded output must not change when a host-only companion is reordered."""

    runner = _runner(
        temperature=1.0,
        top_k=20,
        seed=7,
        force_host_seeded_sampling=True,
    )
    # Alone, this row otherwise fits the bounded device sampler.  The opt-in
    # keeps it on the same vLLM host sampler it would use beside a penalties row.
    assert not runner.check_perform_device_sampling(True, False)

    runner.input_batch.num_reqs = 2
    for seeded_row in (0, 1):
        runner.input_batch.sampling.temperature = torch.ones(2)
        runner.input_batch.sampling.top_k = torch.full((2,), 20)
        runner.input_batch.sampling.seed = torch.full((2,), SEED_NONE_SENTINEL)
        runner.input_batch.sampling.seed[seeded_row] = 7
        assert not runner.check_perform_device_sampling(True, False)

    # No-seed random and seeded greedy requests retain the canonical token-out
    # path; the explicit compatibility policy is only for stochastic seed
    # reproducibility across cohort composition/order.
    runner.input_batch.sampling.seed.fill_(SEED_NONE_SENTINEL)
    assert runner.check_perform_device_sampling(True, False)
    runner.input_batch.sampling.seed.fill_(7)
    runner.input_batch.sampling.temperature.zero_()
    assert runner.check_perform_device_sampling(True, False)


def test_mixed_eligibility_cohort_uses_one_host_compatibility_mode():
    runner = _runner()
    runner.input_batch.num_reqs = 2
    runner.input_batch.sampling.temperature = torch.tensor([0.0, 0.8])
    runner.input_batch.sampling.top_k = torch.tensor([0, 99])

    # Row 0 is canonical on-device greedy, but row 1 exceeds the model's
    # declared top-k bound. The plugin has one output ABI per cohort, so both
    # rows intentionally receive logits for the explicit host sampler.
    assert not runner.check_perform_device_sampling(True, False)

    runner.input_batch.sampling.top_k[1] = 32
    assert runner.check_perform_device_sampling(True, False)


def test_two_host_only_prefill_rows_consume_stacked_torch_logits():
    runner = _runner(temperature=1.0, top_k=20)
    runner.input_batch.num_reqs = 2
    runner.input_batch.sampling.temperature = torch.tensor([1.0, 1.0])
    runner.input_batch.sampling.top_k = torch.tensor([20, 20])
    # min_p is represented by an active vLLM logits processor. One such row
    # selects the shared host-logits ABI for the complete cohort.
    runner.input_batch.sampling.has_active_logitsprocs = lambda: True
    assert not runner.check_perform_device_sampling(False, False)

    captured = {}

    def host_sampler(logits, sampling_metadata):
        del sampling_metadata
        captured["logits"] = logits.clone()
        return SimpleNamespace(
            sampled_token_ids=logits.argmax(dim=-1), logprobs_tensors=None
        )

    runner.host_sampler = host_sampler
    runner.vocab_size = 4
    logits = torch.tensor([[[-2.0, 5.0, 0.0, -1.0]], [[3.0, -2.0, 7.0, 1.0]]])
    sampling_params = TTSamplingParams(
        temperature=torch.ones(2),
        top_k=torch.full((2,), 20, dtype=torch.int32),
        top_p=torch.ones(2),
        presence_penalty=torch.zeros(2),
        frequency_penalty=torch.zeros(2),
        repetition_penalty=torch.ones(2),
        seed=torch.zeros(2, dtype=torch.int64),
        num_logprobs=torch.full((2,), -1, dtype=torch.int32),
        enable_log_probs=torch.zeros(2, dtype=torch.bool),
    )
    model_input = SimpleNamespace(
        intermediate_prefill_mask=None,
        grammar_bitmask=[None],
        generators_list=[{}],
        max_num_logprobs=[None],
        allowed_token_ids_mask_list=[None],
        bad_words_token_ids_list=[[]],
        logitsprocs_list=[None],
        input_tokens=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        prompt_lens=[2, 2],
        prompt_tokens=None,
        output_tokens=None,
    )

    sampled, logprobs = TTModelRunner._get_output_tokens(
        runner,
        logits,
        None,
        sampling_params,
        model_input,
        [2],
        False,
        False,
    )

    assert torch.equal(captured["logits"], logits[:, -1, :])
    assert sampled[0].tolist() == [[1], [2]]
    assert logprobs == [None]
