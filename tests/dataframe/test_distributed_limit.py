"""Unit tests for the distributed-Limit state machine.

Exercises `_LimitCounterImpl` directly (no Ray cluster). The interesting
invariants here are the retry-rewind path in `start_task` — failed
SwordfishTask attempts must release their claimed budget back so the retry
emits the right total. Dataframe-level limit tests don't fail tasks, so this
path is otherwise uncovered.
"""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")

from daft.execution.ray_distributed_limit import _LimitCounterImpl
from tests.conftest import get_tests_daft_runner_name


def test_claim_basic():
    actor = _LimitCounterImpl(limit=10, offset=0)
    actor.start_task("t1")
    assert actor.claim("t1", 100) == (0, 10, True)
    assert actor.claim("t1", 100) == (0, 0, True)


def test_claim_with_offset():
    actor = _LimitCounterImpl(limit=10, offset=5)
    actor.start_task("t1")
    # 5 rows go to skip, 10 to take, 5 discarded.
    assert actor.claim("t1", 20) == (5, 10, True)


def test_claim_offset_spans_multiple_calls():
    actor = _LimitCounterImpl(limit=10, offset=15)
    actor.start_task("t1")
    # First batch fully consumed by offset.
    assert actor.claim("t1", 10) == (10, 0, False)
    # Second batch: 5 more to skip, then 10 to take.
    assert actor.claim("t1", 20) == (5, 10, True)


def test_start_task_rewinds_prior_claim():
    """A retried task gets its prior take/skip refunded to the global budget."""
    actor = _LimitCounterImpl(limit=100, offset=0)
    actor.start_task("t1")
    assert actor.claim("t1", 60) == (0, 60, False)
    assert actor.remaining_take == 40

    # Simulate retry: same input_id calls start_task again.
    actor.start_task("t1")
    assert actor.remaining_take == 100, "budget should be restored after rewind"
    # The retry can now claim up to the full limit again.
    assert actor.claim("t1", 80) == (0, 80, False)


def test_start_task_rewinds_offset_claim():
    """Rewind must restore offset progress too, not just take."""
    actor = _LimitCounterImpl(limit=10, offset=20)
    actor.start_task("t1")
    assert actor.claim("t1", 15) == (15, 0, False)
    assert actor.remaining_skip == 5

    actor.start_task("t1")
    assert actor.remaining_skip == 20, "offset progress should rewind"
    assert actor.remaining_take == 10


def test_start_task_rewind_isolated_per_task():
    """Rewinding t1 must not affect t2's claims."""
    actor = _LimitCounterImpl(limit=100, offset=0)
    actor.start_task("t1")
    actor.start_task("t2")
    actor.claim("t1", 30)  # t1 takes 30
    actor.claim("t2", 40)  # t2 takes 40
    assert actor.remaining_take == 30

    # Retry t1 only.
    actor.start_task("t1")
    # t1's 30 should be refunded; t2's 40 stays claimed.
    assert actor.remaining_take == 60
    # t2's bookkeeping should be intact.
    assert actor.input_claims["t2"] == (0, 40)


def test_double_start_task_is_idempotent():
    """Calling start_task twice with no intervening claim must not rewind twice."""
    actor = _LimitCounterImpl(limit=100, offset=0)
    actor.start_task("t1")
    actor.claim("t1", 30)
    assert actor.remaining_take == 70

    actor.start_task("t1")  # first rewind: refund 30
    actor.start_task("t1")  # second call: nothing to refund, must be a no-op
    assert actor.remaining_take == 100


def test_zero_claim_entries_dropped():
    """Tasks that never consume budget shouldn't accumulate in input_claims."""
    actor = _LimitCounterImpl(limit=5, offset=0)
    actor.start_task("t1")
    actor.claim("t1", 10)  # claims all 5
    assert actor.is_done()

    # Subsequent tasks past the limit get (0, 0, True) and shouldn't be retained.
    for i in range(100):
        tid = f"past_limit_{i}"
        actor.start_task(tid)
        assert actor.claim(tid, 50) == (0, 0, True)
        assert tid not in actor.input_claims, "past-limit task should not be retained"

    # Only the one boundary task remains.
    assert set(actor.input_claims.keys()) == {"t1"}


def test_is_done_transitions():
    actor = _LimitCounterImpl(limit=10, offset=0)
    assert not actor.is_done()
    actor.start_task("t1")
    actor.claim("t1", 5)
    assert not actor.is_done()
    actor.claim("t1", 5)
    assert actor.is_done()


@pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray Runner to be in use")
def test_distributed_limit_retries_after_worker_death(tmp_path):
    """`.limit(N)` must still produce N rows when a SwordfishTask crashes mid-claim.

    Without the rewind in `_LimitCounterImpl.start_task`, the failed attempt's
    claim stays charged against the global budget while its slice never reaches
    downstream — the retry then sees a smaller budget and the output undercounts.
    """
    import os

    import daft
    from daft import DataType, col, func

    marker = str(tmp_path / "crashed_once")

    @func(return_dtype=DataType.int64(), cpus=os.cpu_count())
    def crash_once_on_zero(v: int) -> int:
        import os

        if v == 0 and not os.path.exists(marker):
            with open(marker, "w") as f:
                f.write("crashed")
            # Hard-exit the swordfish actor process. Ray surfaces this as
            # ActorDiedError → RayTaskResult.worker_died() → dispatcher marks
            # WorkerDied. flotilla's RayWorkerManager.refresh_workers loop
            # then spawns a fresh actor on this node within ~5s, onto which
            # the failed task is re-dispatched. The retry's
            # DistributedLimitSink calls start_task(input_id), which refunds
            # the crashed attempt's prior claim before claiming again.
            os._exit(1)
        return v

    df = daft.range(0, 15, partitions=15).limit(3).select(crash_once_on_zero(col("id")))
    result = df.to_pydict()

    import os

    assert os.path.exists(marker), "UDF never crashed — retry path not exercised"
    # Single-CPU serialization makes contributor order deterministic: task 0
    # retries and finishes first, then tasks 1 and 2 run in sequence. Without
    # rewind in start_task, the crashed task's row would be missing
    # (e.g. [1, 2] instead of [0, 1, 2]).
    assert result["id"] == [0, 1, 2], (
        f"expected [0, 1, 2] after retry, got {result['id']}. "
        "If 0 is missing, the limit actor failed to rewind the crashed "
        "task's claim in start_task."
    )


@pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray Runner to be in use")
def test_distributed_limit_stops_scanning_early(tmp_path):
    """The sink must stop consuming its input once the global limit is satisfied.

    Without early-stop, every scan task drains its whole input even though the
    limit actor has no budget left, which is what makes
    `read_parquet(...).where(...).limit(N)` slow over large datasets.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    import daft
    from daft import DataType, col
    from daft.series import Series

    rows_per_group = 5000
    num_groups = 400
    total_rows = rows_per_group * num_groups
    path = tmp_path / "data.parquet"
    with pq.ParquetWriter(path, pa.schema([("id", pa.int64())])) as writer:
        for g in range(num_groups):
            ids = pa.array(range(g * rows_per_group, (g + 1) * rows_per_group), type=pa.int64())
            writer.write_table(pa.table({"id": ids}))

    seen_file = tmp_path / "seen.log"

    @daft.func.batch(return_dtype=DataType.bool())
    def log_and_keep(s: Series) -> Series:
        with open(seen_file, "ab", buffering=0) as f:
            f.write(f"{len(s)}\n".encode())
        return Series.from_pylist([True] * len(s))

    # `.where` runs this UDF upstream of the distributed limit, so the log
    # records how many rows the scan actually produced.
    result = daft.read_parquet(str(path)).where(log_and_keep(col("id"))).limit(10).to_pydict()
    assert len(result["id"]) == 10

    seen = sum(int(line) for line in seen_file.read_text().splitlines() if line)
    assert seen < total_rows // 2, (
        f"distributed limit consumed {seen} of {total_rows} upstream rows; "
        "it should stop after the limit is satisfied instead of draining the scan"
    )
