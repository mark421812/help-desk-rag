from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from helpdesk.domain.errors import DomainValidationError, InvalidTransitionError
from helpdesk.domain.ingestion import (
    IngestionJob,
    IngestionJobStatus,
    LeaseLostError,
    MetadataMissingError,
    OcrRequiredError,
    SourceItem,
    SourceItemStatus,
)

NOW = datetime(2026, 11, 13, 2, 0, tzinfo=UTC)
HASH_A = hashlib.sha256(b"a").hexdigest()
HASH_B = hashlib.sha256(b"b").hexdigest()
CONTENT = hashlib.sha256(b"chunks").hexdigest()


def item(status: SourceItemStatus = SourceItemStatus.PENDING, **changes: object) -> SourceItem:
    base = SourceItem.discovered(
        "brochure.pdf",
        source_version="mtime:1;size:10",
        metadata_source_hash=HASH_A,
        file_size=10,
        modified_at=NOW,
        at=NOW,
    )
    changes.setdefault("deleted_at", NOW if status is SourceItemStatus.DELETED else None)
    return replace(base, status=status, **changes)  # type: ignore[arg-type]


def observe(
    source: SourceItem, *, version: str = "mtime:1;size:10", metadata: str = HASH_A
) -> tuple[SourceItem, bool]:
    return source.observe(
        source_version=version,
        metadata_source_hash=metadata,
        file_size=10,
        modified_at=NOW,
        at=NOW + timedelta(days=1),
    )


def test_a_processed_item_completes_and_remembers_what_it_indexed() -> None:
    completed = (
        item()
        .begin_processing()
        .complete(content_hash=CONTENT, mime_type="application/pdf", at=NOW)
    )

    assert completed.status is SourceItemStatus.COMPLETED
    assert completed.last_successful_content_hash == CONTENT
    assert completed.last_successful_metadata_hash == HASH_A


def test_an_unchanged_indexed_file_needs_no_job_but_a_changed_one_does() -> None:
    indexed = item().begin_processing().complete(content_hash=CONTENT, mime_type="x", at=NOW)

    unchanged, unchanged_needs = observe(indexed)
    new_content, content_needs = observe(indexed, version="mtime:2;size:11")
    new_metadata, metadata_needs = observe(indexed, metadata=HASH_B)

    assert (unchanged.status, unchanged_needs) == (SourceItemStatus.COMPLETED, False)
    assert (new_content.status, content_needs) == (SourceItemStatus.PENDING, True)
    assert (new_metadata.status, metadata_needs) == (SourceItemStatus.PENDING, True)


def test_a_rejected_file_is_not_retried_until_it_changes() -> None:
    rejected = item().begin_processing().reject(MetadataMissingError(("title",)))

    again, needs = observe(rejected)

    assert rejected.status is SourceItemStatus.METADATA_MISSING
    assert rejected.error_code == "metadata_missing"
    assert again.status is SourceItemStatus.METADATA_MISSING
    # 掃描仍會嘗試排 job，但同一個指紋的 job key 已存在，所以實際上不會再排。
    assert needs is True
    assert again.discovery_fingerprint == rejected.discovery_fingerprint


def test_missing_is_a_counter_that_keeps_the_last_status_and_returning_bumps_the_generation() -> (
    None
):
    indexed = item().begin_processing().complete(content_hash=CONTENT, mime_type="x", at=NOW)

    gone_twice = indexed.mark_missing().mark_missing()
    back, needs = observe(gone_twice)

    assert (gone_twice.status, gone_twice.missing_scan_count) == (SourceItemStatus.COMPLETED, 2)
    assert gone_twice.is_missing
    assert (back.status, back.missing_scan_count, needs) == (SourceItemStatus.PENDING, 0, True)
    assert back.change_generation == indexed.change_generation + 1


def test_a_deleted_file_that_reappears_is_processed_again() -> None:
    deleted = item(SourceItemStatus.COMPLETED).mark_missing().mark_deleted(NOW)

    back, needs = observe(deleted)

    assert deleted.deleted_at == NOW and deleted.last_successful_content_hash is None
    assert (back.status, back.deleted_at, needs) == (SourceItemStatus.PENDING, None, True)


@pytest.mark.parametrize(
    ("status", "move"),
    [
        (SourceItemStatus.PENDING, "complete"),
        (SourceItemStatus.COMPLETED, "complete"),
        (SourceItemStatus.DELETED, "begin_processing"),
        (SourceItemStatus.DELETED, "mark_deleted"),
        (SourceItemStatus.FAILED, "reject"),
        (SourceItemStatus.QUARANTINED, "fail"),
    ],
)
def test_moves_outside_the_item_transition_table_are_rejected(
    status: SourceItemStatus, move: str
) -> None:
    source = item(status)
    moves: dict[str, Callable[[], SourceItem]] = {
        "complete": lambda: source.complete(content_hash=CONTENT, mime_type="x", at=NOW),
        "begin_processing": source.begin_processing,
        "mark_deleted": lambda: source.mark_deleted(NOW),
        "reject": lambda: source.reject(OcrRequiredError("no text")),
        "fail": lambda: source.fail("processing_error", "boom"),
    }

    with pytest.raises(InvalidTransitionError):
        moves[move]()


def test_an_item_rejects_inconsistent_rows() -> None:
    with pytest.raises(DomainValidationError):
        item(SourceItemStatus.COMPLETED, deleted_at=NOW)
    with pytest.raises(DomainValidationError):
        item(error_code="failed", error_message=None)


def job(**changes: object) -> IngestionJob:
    return replace(IngestionJob.process(item(), max_attempts=3, now=NOW), **changes)  # type: ignore[arg-type]


def test_every_observed_change_gets_its_own_job_even_when_it_reverts() -> None:
    def key(source: SourceItem) -> str:
        return IngestionJob.process(source, max_attempts=3, now=NOW).idempotency_key

    original = item().begin_processing().complete(content_hash=CONTENT, mime_type="x", at=NOW)
    unchanged, _ = observe(original)
    edited, _ = observe(original, metadata=HASH_B)
    reverted, _ = observe(
        edited.begin_processing().complete(content_hash=CONTENT, mime_type="x", at=NOW)
    )

    assert key(unchanged) == key(original)
    assert len({key(original), key(edited), key(reverted)}) == 3
    assert reverted.discovery_fingerprint == original.discovery_fingerprint


def test_a_lease_is_exclusive_until_it_expires() -> None:
    lease = timedelta(minutes=10)
    claimed = job().claim("worker-1", NOW, lease)

    with pytest.raises(LeaseLostError):
        claimed.claim("worker-2", NOW + timedelta(minutes=5), lease)
    with pytest.raises(LeaseLostError):
        claimed.complete("worker-2", NOW)
    taken_over = claimed.claim("worker-2", NOW + lease, lease)
    with pytest.raises(LeaseLostError):
        taken_over.complete("worker-1", NOW + lease)

    assert taken_over.complete("worker-2", NOW + lease).status is IngestionJobStatus.COMPLETED


def test_retryable_failures_back_off_exponentially_until_attempts_run_out() -> None:
    lease, base = timedelta(minutes=10), timedelta(seconds=30)
    current = job()
    waits = []
    for attempt in range(3):
        now = current.available_at
        current = current.claim("w", now, lease).fail(
            "w", now, code="processing_error", message="boom", retryable=True, base_delay=base
        )
        waits.append(current.available_at - now if attempt < 2 else None)

    assert waits[:2] == [base, base * 2]
    assert (current.status, current.attempt_count) == (IngestionJobStatus.FAILED, 3)


def test_a_permanent_failure_ends_the_job_at_once_and_only_failed_jobs_can_be_retried() -> None:
    claimed = job().claim("w", NOW, timedelta(minutes=1))
    failed = claimed.fail(
        "w", NOW, code="metadata_missing", message="x", retryable=False, base_delay=timedelta(1)
    )
    retried = failed.retry_manually(NOW)

    assert (failed.status, failed.attempt_count) == (IngestionJobStatus.FAILED, 1)
    assert (retried.status, retried.attempt_count, retried.error_code) == (
        IngestionJobStatus.PENDING,
        0,
        None,
    )
    with pytest.raises(InvalidTransitionError):
        claimed.complete("w", NOW).retry_manually(NOW)


def test_a_job_row_needs_a_lease_exactly_while_processing() -> None:
    with pytest.raises(DomainValidationError):
        job(status=IngestionJobStatus.PROCESSING)
    with pytest.raises(DomainValidationError):
        job(leased_by="w", lease_expires_at=NOW)
