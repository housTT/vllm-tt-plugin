# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from types import SimpleNamespace

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode


def _running(is_prefill_chunk=False):
    """A stand-in for a running request, seen only through the fields TT reads."""
    return SimpleNamespace(is_prefill_chunk=is_prefill_chunk)


def _scheduler(*, running=(), waiting=0, skipped_waiting=0, mode):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    for _ in range(waiting):
        scheduler.waiting.add_request(object())
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    for _ in range(skipped_waiting):
        scheduler.skipped_waiting.add_request(object())
    scheduler.running = list(running)
    scheduler.max_num_running_reqs = 8
    scheduler._forced_mode = mode
    return scheduler


def test_forced_prefill_does_not_fallback_to_decode_per_lane(monkeypatch):
    scheduler = _scheduler(
        running=[_running()], waiting=1, mode=TTSchedulingMode.PREFILL_ONLY
    )

    monkeypatch.setattr(scheduler, "_schedule_prefill_only", SchedulerOutput.make_empty)

    def fail_local_decode_fallback():
        raise AssertionError("forced prefill must remain coordinated across lanes")

    monkeypatch.setattr(scheduler, "_schedule_decode_only", fail_local_decode_fallback)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0


def test_forced_decode_hides_and_restores_skipped_waiting(monkeypatch):
    scheduler = _scheduler(
        running=[_running()], skipped_waiting=1, mode=TTSchedulingMode.DECODE_ONLY
    )

    saved_waiting = scheduler.waiting
    saved_skipped_waiting = scheduler.skipped_waiting
    skipped_request = scheduler.skipped_waiting.peek_request()
    visible_queues = []

    def fake_base_schedule(self, throttle_prefills=False):
        visible_queues.append((bool(self.waiting), bool(self.skipped_waiting)))
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base_schedule)

    scheduler.schedule()

    assert visible_queues == [(False, False)]
    assert scheduler.waiting is saved_waiting
    assert scheduler.skipped_waiting is saved_skipped_waiting
    assert scheduler.skipped_waiting.peek_request() is skipped_request


def test_running_continuation_alone_still_schedules_prefill(monkeypatch):
    # Nothing waiting; the only work is a partial prefill in `running`. Only a
    # prefill step can advance it, so DEFAULT mode must not pick decode.
    scheduler = _scheduler(
        running=[_running(is_prefill_chunk=True)], mode=TTSchedulingMode.DEFAULT
    )

    calls = []
    monkeypatch.setattr(
        scheduler,
        "_schedule_prefill_only",
        lambda: calls.append("prefill") or SchedulerOutput.make_empty(),
    )

    def fail_decode_fallback():
        raise AssertionError("a decode step cannot advance a partial prefill")

    monkeypatch.setattr(scheduler, "_schedule_decode_only", fail_decode_fallback)

    scheduler.schedule()

    assert calls == ["prefill"]


def test_prefill_only_hides_decodes_but_keeps_continuations(monkeypatch):
    continuation = _running(is_prefill_chunk=True)
    decode = _running()
    scheduler = _scheduler(
        running=[decode, continuation], waiting=1, mode=TTSchedulingMode.PREFILL_ONLY
    )
    seen = {}

    def fake_base_schedule(self, throttle_prefills=False):
        seen["running"] = list(self.running)
        seen["max_num_running_reqs"] = self.max_num_running_reqs
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base_schedule)

    scheduler.schedule()

    assert seen["running"] == [continuation]
    # One slot is held by the hidden decode, so the waiting loop sees 8 - 1.
    assert seen["max_num_running_reqs"] == 7
    # Restored: the hidden decodes are appended back after the base pass.
    assert scheduler.running == [continuation, decode]
    assert scheduler.max_num_running_reqs == 8


def test_decode_only_hides_continuations_and_restores_them(monkeypatch):
    continuation = _running(is_prefill_chunk=True)
    decode = _running()
    scheduler = _scheduler(
        running=[decode, continuation], mode=TTSchedulingMode.DECODE_ONLY
    )
    seen = {}

    def fake_base_schedule(self, throttle_prefills=False):
        seen["running"] = list(self.running)
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base_schedule)

    scheduler.schedule()

    assert seen["running"] == [decode]
    assert scheduler.running == [decode, continuation]


def test_decode_steps_run_between_the_chunks_of_a_partial_prefill(monkeypatch):
    continuation = _running(is_prefill_chunk=True)
    decode = _running()
    scheduler = _scheduler(running=[decode, continuation], mode=TTSchedulingMode.DEFAULT)
    scheduler._chunked_prefill_decode_steps_cached = 3
    calls = []

    def fake_prefill():
        calls.append("prefill")
        output = SchedulerOutput.make_empty()
        output.total_num_scheduled_tokens = 8192
        return output

    monkeypatch.setattr(scheduler, "_schedule_prefill_only", fake_prefill)
    monkeypatch.setattr(
        scheduler, "_schedule_decode_only", lambda: calls.append("decode") or SchedulerOutput.make_empty()
    )

    for _ in range(9):
        scheduler.schedule()

    assert calls == ["prefill", "decode", "decode", "decode", "prefill", "decode", "decode", "decode", "prefill"]


def test_no_decode_steps_are_owed_when_the_prefill_finished_or_nobody_decodes(monkeypatch):
    fresh = _running()
    scheduler = _scheduler(running=[fresh], waiting=1, mode=TTSchedulingMode.DEFAULT)
    scheduler._chunked_prefill_decode_steps_cached = 3
    calls = []

    def fake_prefill():
        calls.append("prefill")
        output = SchedulerOutput.make_empty()
        output.total_num_scheduled_tokens = 512
        return output

    monkeypatch.setattr(scheduler, "_schedule_prefill_only", fake_prefill)
    monkeypatch.setattr(
        scheduler, "_schedule_decode_only", lambda: calls.append("decode") or SchedulerOutput.make_empty()
    )
    scheduler.schedule()
    scheduler.schedule()
    assert calls == ["prefill", "prefill"]

    only_partials = _scheduler(running=[_running(is_prefill_chunk=True)], mode=TTSchedulingMode.DEFAULT)
    only_partials._chunked_prefill_decode_steps_cached = 3
    seen = []
    monkeypatch.setattr(
        only_partials,
        "_schedule_prefill_only",
        lambda: seen.append("prefill") or SchedulerOutput.make_empty(),
    )
    monkeypatch.setattr(
        only_partials,
        "_schedule_decode_only",
        lambda: seen.append("decode") or SchedulerOutput.make_empty(),
    )
    only_partials.schedule()
    only_partials.schedule()
    assert seen == ["prefill", "prefill"]


def test_prefix_cache_hits_are_floored_to_the_model_alignment():
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

    blocks = [SimpleNamespace(block_id=i) for i in range(78)]
    calls = []

    def get_computed_blocks(request):
        calls.append(request)
        return KVCacheBlocks((blocks,)), 78 * 64, 0

    manager = SimpleNamespace(
        enable_caching=True,
        get_computed_blocks=get_computed_blocks,
        empty_kv_cache_blocks=KVCacheBlocks(((),)),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=64))]),
    )
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.vllm_config = SimpleNamespace(additional_config={"tt": {"prefill_chunk_alignment": 512}})
    scheduler.kv_cache_manager = manager
    scheduler._install_prefix_cache_alignment()

    trimmed, computed, boundary = manager.get_computed_blocks("req")
    assert computed == 4608 and boundary == 0
    assert [b.block_id for b in trimmed.blocks[0]] == list(range(72))
    assert calls == ["req"]

    manager.get_computed_blocks = lambda request: (KVCacheBlocks((blocks[:4],)), 4 * 64, 0)
    scheduler._install_prefix_cache_alignment()
    empty, computed, _ = manager.get_computed_blocks("short")
    assert computed == 0 and empty is manager.empty_kv_cache_blocks


def test_prefix_cache_alignment_is_left_alone_without_the_knob_or_caching():
    manager = SimpleNamespace(enable_caching=False, get_computed_blocks=lambda request: ("x", 100, 0))
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.vllm_config = SimpleNamespace(additional_config={"tt": {"prefill_chunk_alignment": 512}})
    scheduler.kv_cache_manager = manager
    scheduler._install_prefix_cache_alignment()
    assert manager.get_computed_blocks("r") == ("x", 100, 0)
    manager.enable_caching = True
    scheduler.vllm_config = SimpleNamespace(additional_config={"tt": {}})
    scheduler._install_prefix_cache_alignment()
    assert manager.get_computed_blocks("r") == ("x", 100, 0)


def test_non_final_chunks_end_on_the_alignment_and_a_budget_remnant_defers():
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.vllm_config = SimpleNamespace(additional_config={"tt": {"prefill_chunk_alignment": 512}})
    scheduler.kv_cache_manager = SimpleNamespace(enable_caching=False)
    scheduler._install_prefix_cache_alignment()
    assert scheduler.need_mamba_block_aligned_split is True

    long_prompt = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=130816, num_tokens=130816)
    assert scheduler._mamba_block_aligned_split(long_prompt, 256) == 0
    assert scheduler._mamba_block_aligned_split(long_prompt, 3192) == 3072
    assert scheduler._mamba_block_aligned_split(long_prompt, 8192) == 8192
    last_chunk = SimpleNamespace(num_computed_tokens=122880, num_prompt_tokens=130816, num_tokens=130816)
    assert scheduler._mamba_block_aligned_split(last_chunk, 7936) == 7936
    cache_hit = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=20000, num_tokens=20000)
    assert scheduler._mamba_block_aligned_split(cache_hit, 8192, num_new_local_computed_tokens=4608) == 8192
    assert scheduler._mamba_block_aligned_split(cache_hit, 8192, num_new_local_computed_tokens=4992) == 7808
    decoding = SimpleNamespace(num_computed_tokens=20000, num_prompt_tokens=20000, num_tokens=20001)
    assert scheduler._mamba_block_aligned_split(decoding, 1) == 1
