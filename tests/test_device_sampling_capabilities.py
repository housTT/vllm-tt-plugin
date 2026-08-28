# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_tt_plugin.model_runner import TTModelRunner


def _runner(*, temperature=0.0, top_k=0, penalties=False):
    sampling = SimpleNamespace(
        temperature=torch.tensor([temperature]),
        top_k=torch.tensor([top_k]),
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
