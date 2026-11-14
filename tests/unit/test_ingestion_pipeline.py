"""掃描、處理與刪除規則，用正式的檔案、parser、切塊與 YAML adapter，加上記憶體版的資料庫與索引。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from helpdesk.domain.documents import Document, VersionId
from helpdesk.domain.ingestion import (
    IngestionJobStatus,
    ScanRejectedError,
    SourceFileNotFoundError,
    SourceItemStatus,
)
from tests.ingestion_fakes import Pipeline

POLICY = """# 請假規則

員工每年有七天特休，應事先提出申請並經主管核准後才能休假，未經核准視為曠職處理。

## 病假

病假需檢附醫院證明，連續三天以上需另附診斷書，並於復職後三日內補辦請假手續。
"""
REVISED = POLICY.replace("七天", "十天")


def write(root: Path, name: str, text: str = POLICY, *, mtime: int | None = None) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def current_version(pipeline: Pipeline, name: str) -> VersionId | None:
    document = pipeline.store.document_of(name)
    return None if document is None else document.current_version_id


@pytest.fixture
def pipeline(tmp_path: Path) -> Pipeline:
    return Pipeline(root=tmp_path)


async def test_a_new_document_is_parsed_chunked_embedded_and_indexed(pipeline: Pipeline) -> None:
    write(pipeline.root, "leave.md")

    report = await pipeline.scanner().run()
    handled = await pipeline.drain()

    item = pipeline.store.item("leave.md")
    version = current_version(pipeline, "leave.md")
    assert version is not None
    chunks = pipeline.store.chunks_of(version)
    assert (report.discovered, report.queued, handled) == (1, 1, 1)
    assert item.status is SourceItemStatus.COMPLETED
    assert chunks and all(chunk.qdrant_point_id == chunk.id for chunk in chunks)
    assert {chunk.section_path for chunk in chunks} == {"請假規則", "請假規則 > 病假"}
    assert sorted(pipeline.index.points) == sorted(str(chunk.id) for chunk in chunks)
    assert all(point.is_current for point in pipeline.index.points.values())
    assert pipeline.index.points[str(chunks[0].id)].title == "leave"


async def test_rescanning_an_unchanged_folder_queues_nothing_and_changes_nothing(
    pipeline: Pipeline,
) -> None:
    write(pipeline.root, "leave.md")
    await pipeline.scanner().run()
    await pipeline.drain()
    points = dict(pipeline.index.points)
    embedded = len(pipeline.embedder.embedded)

    report = await pipeline.scanner().run()

    assert (report.queued, await pipeline.drain()) == (0, 0)
    assert pipeline.index.points == points
    assert len(pipeline.embedder.embedded) == embedded


async def test_reprocessing_identical_content_reuses_the_version_without_embedding_again(
    pipeline: Pipeline,
) -> None:
    path = write(pipeline.root, "leave.md", mtime=1_700_000_000)
    await pipeline.scanner().run()
    await pipeline.drain()
    version = current_version(pipeline, "leave.md")
    embedded = len(pipeline.embedder.embedded)

    os.utime(path, (1_800_000_000, 1_800_000_000))
    report = await pipeline.scanner().run()
    await pipeline.drain()

    assert report.queued == 1
    assert current_version(pipeline, "leave.md") == version
    assert len(pipeline.embedder.embedded) == embedded
    assert len(pipeline.index.points) == len(pipeline.store.chunks)


async def test_a_new_version_becomes_current_only_after_its_points_are_written(
    pipeline: Pipeline,
) -> None:
    write(pipeline.root, "leave.md", mtime=1_700_000_000)
    await pipeline.scanner().run()
    await pipeline.drain()
    old_version = current_version(pipeline, "leave.md")
    seen_during_upsert: list[tuple[VersionId | None, int]] = []

    def record(document: Document) -> None:
        old_points = [p for p in pipeline.index.points.values() if p.version_id == str(old_version)]
        seen_during_upsert.append((current_version(pipeline, "leave.md"), len(old_points)))

    pipeline.index.on_upsert = record
    write(pipeline.root, "leave.md", REVISED, mtime=1_800_000_000)
    await pipeline.scanner().run()
    await pipeline.drain()

    new_version = current_version(pipeline, "leave.md")
    assert new_version not in (None, old_version)
    [(current_while_writing, old_points_while_writing)] = seen_during_upsert
    assert current_while_writing == old_version
    assert old_points_while_writing > 0
    assert set(pipeline.store.versions) == {new_version}
    assert {point.version_id for point in pipeline.index.points.values()} == {str(new_version)}
    assert all(point.is_current for point in pipeline.index.points.values())
    assert any("十天" in chunk.content for chunk in pipeline.store.chunks.values())


async def test_a_crash_after_writing_points_converges_on_retry_without_duplicates(
    pipeline: Pipeline,
) -> None:
    write(pipeline.root, "leave.md")
    pipeline.index.fail_publish = 1
    await pipeline.scanner().run()
    await pipeline.drain()

    failed = pipeline.store.item("leave.md")
    job = next(iter(pipeline.store.jobs.values()))
    assert (failed.status, job.status, job.error_code) == (
        SourceItemStatus.FAILED,
        IngestionJobStatus.PENDING,
        "processing_error",
    )

    pipeline.clock.advance(seconds=10)
    await pipeline.drain()

    version = current_version(pipeline, "leave.md")
    assert version is not None
    assert pipeline.store.item("leave.md").status is SourceItemStatus.COMPLETED
    assert len(pipeline.store.versions) == 1
    assert sorted(pipeline.index.points) == sorted(
        str(chunk.id) for chunk in pipeline.store.chunks_of(version)
    )


async def test_a_file_seen_missing_once_is_kept_and_twice_is_deleted(pipeline: Pipeline) -> None:
    for name in ("a.md", "b.md", "c.md", "d.md", "e.md"):
        write(pipeline.root, name)
    await pipeline.scanner().run()
    await pipeline.drain()
    (pipeline.root / "a.md").unlink()

    first = await pipeline.scanner().run()
    await pipeline.drain()
    kept = pipeline.store.item("a.md")
    second = await pipeline.scanner().run()
    await pipeline.drain()

    assert (first.missing, first.deletions_queued) == (1, 0)
    assert (kept.status, kept.missing_scan_count) == (SourceItemStatus.COMPLETED, 1)
    assert (second.missing, second.deletions_queued) == (1, 1)
    assert pipeline.store.item("a.md").status is SourceItemStatus.DELETED
    assert pipeline.store.document_of("a.md") is None
    assert len(pipeline.index.current_points()) == len(pipeline.store.chunks)
    assert len({point.document_id for point in pipeline.index.points.values()}) == 4


async def test_a_file_that_comes_back_after_one_miss_is_never_deleted(pipeline: Pipeline) -> None:
    for name in ("a.md", "b.md", "c.md", "d.md", "e.md"):
        write(pipeline.root, name)
    await pipeline.scanner().run()
    await pipeline.drain()
    moved = (pipeline.root / "a.md").read_text(encoding="utf-8")
    (pipeline.root / "a.md").unlink()
    await pipeline.scanner().run()

    write(pipeline.root, "a.md", moved)
    await pipeline.scanner().run()
    await pipeline.scanner().run()
    await pipeline.drain()

    item = pipeline.store.item("a.md")
    assert (item.status, item.missing_scan_count) == (SourceItemStatus.COMPLETED, 0)
    assert pipeline.store.document_of("a.md") is not None


async def test_a_deleted_file_that_returns_is_indexed_again(pipeline: Pipeline) -> None:
    for name in ("a.md", "b.md", "c.md", "d.md", "e.md"):
        write(pipeline.root, name)
    await pipeline.scanner().run()
    await pipeline.drain()
    (pipeline.root / "a.md").unlink()
    await pipeline.scanner().run()
    await pipeline.scanner().run()
    await pipeline.drain()

    write(pipeline.root, "a.md")
    await pipeline.scanner().run()
    await pipeline.drain()

    assert pipeline.store.item("a.md").status is SourceItemStatus.COMPLETED
    assert current_version(pipeline, "a.md") is not None


@pytest.mark.parametrize("removed", [2, 5])
async def test_a_scan_that_loses_too_many_files_is_refused(
    pipeline: Pipeline, removed: int
) -> None:
    names = ["a.md", "b.md", "c.md", "d.md", "e.md"]
    for name in names:
        write(pipeline.root, name)
    await pipeline.scanner().run()
    await pipeline.drain()
    for name in names[:removed]:
        (pipeline.root / name).unlink()

    with pytest.raises(ScanRejectedError):
        await pipeline.scanner().run()

    assert all(item.missing_scan_count == 0 for item in pipeline.store.items.values())
    forced = await pipeline.scanner().run(allow_mass_deletion=True)
    assert forced.missing == removed


async def test_acl_on_rejects_a_document_without_metadata(tmp_path: Path) -> None:
    pipeline = Pipeline(root=tmp_path, access_control=True)
    write(tmp_path, "BookRAG.md")

    await pipeline.scanner().run()
    await pipeline.drain()

    item = pipeline.store.item("BookRAG.md")
    assert (item.status, item.error_code) == (SourceItemStatus.METADATA_MISSING, "metadata_missing")
    assert pipeline.store.document_of("BookRAG.md") is None
    assert pipeline.index.points == {}
    assert (await pipeline.scanner().run()).queued == 0


async def test_turning_acl_on_withdraws_documents_indexed_without_metadata(tmp_path: Path) -> None:
    open_pipeline = Pipeline(root=tmp_path)
    write(tmp_path, "BookRAG.md")
    write(tmp_path, "brochure.md")
    (tmp_path / "brochure.md.metadata.yaml").write_text(
        "title: 簡章\ndocument_type: policy\nowner_department: HR\ndocument_owner: hr\n"
        "status: published\nexternal_llm_allowed: true\n",
        encoding="utf-8",
    )
    await open_pipeline.scanner().run()
    await open_pipeline.drain()
    assert {p.title for p in open_pipeline.index.points.values()} == {"BookRAG", "簡章"}

    acl_pipeline = Pipeline(
        root=tmp_path,
        access_control=True,
        store=open_pipeline.store,
        index=open_pipeline.index,
        clock=open_pipeline.clock,
    )
    report = await acl_pipeline.scanner().run()
    await acl_pipeline.drain()

    assert report.queued == 2
    assert acl_pipeline.store.item("BookRAG.md").status is SourceItemStatus.METADATA_MISSING
    assert acl_pipeline.store.document_of("BookRAG.md") is None
    assert {p.title for p in acl_pipeline.index.points.values()} == {"簡章"}


async def test_switching_acl_on_and_off_again_reindexes_the_document(tmp_path: Path) -> None:
    write(tmp_path, "BookRAG.md")
    shared = Pipeline(root=tmp_path)
    await shared.scanner().run()
    await shared.drain()

    def toggled(access_control: bool) -> Pipeline:
        return Pipeline(
            root=tmp_path,
            access_control=access_control,
            store=shared.store,
            index=shared.index,
            clock=shared.clock,
            embedder=shared.embedder,
        )

    await toggled(True).scanner().run()
    await toggled(True).drain()
    rejected = shared.store.item("BookRAG.md").status
    await toggled(False).scanner().run()
    await toggled(False).drain()

    assert rejected is SourceItemStatus.METADATA_MISSING
    assert shared.store.item("BookRAG.md").status is SourceItemStatus.COMPLETED
    assert {point.title for point in shared.index.current_points()} == {"BookRAG"}


async def test_permanent_rejections_record_their_status_and_code(tmp_path: Path) -> None:
    pipeline = Pipeline(root=tmp_path, max_file_size_bytes=2_000)
    write(tmp_path, "huge.md", POLICY * 20)
    write(tmp_path, "empty.txt", "   \n")
    write(tmp_path, "bad.md")
    (tmp_path / "bad.md.metadata.yaml").write_text("status: approved\n", encoding="utf-8")

    await pipeline.scanner().run()
    await pipeline.drain()

    outcomes = {
        name: (pipeline.store.item(name).status, pipeline.store.item(name).error_code)
        for name in ("huge.md", "empty.txt", "bad.md")
    }
    assert outcomes == {
        "huge.md": (SourceItemStatus.QUARANTINED, "file_too_large"),
        "empty.txt": (SourceItemStatus.QUARANTINED, "invalid_document"),
        "bad.md": (SourceItemStatus.QUARANTINED, "invalid_metadata"),
    }
    assert all(job.status is IngestionJobStatus.FAILED for job in pipeline.store.jobs.values())
    assert pipeline.index.points == {}


async def test_ingest_queues_one_file_without_marking_others_missing(pipeline: Pipeline) -> None:
    write(pipeline.root, "a.md")
    write(pipeline.root, "nested/b.md")

    report = await pipeline.ingester().run("nested/b.md")
    again = await pipeline.ingester().run("nested/b.md")
    await pipeline.drain()

    assert (report.queued, again.queued) == (True, False)
    assert [item.relative_path for item in pipeline.store.items.values()] == ["nested/b.md"]
    assert pipeline.store.item("nested/b.md").status is SourceItemStatus.COMPLETED
    with pytest.raises(SourceFileNotFoundError):
        await pipeline.ingester().run("nested/missing.md")


async def test_a_job_whose_file_changed_since_the_scan_is_retried_not_indexed(
    pipeline: Pipeline,
) -> None:
    write(pipeline.root, "leave.md", mtime=1_700_000_000)
    await pipeline.scanner().run()
    write(pipeline.root, "leave.md", REVISED, mtime=1_800_000_000)

    await pipeline.drain()

    item = pipeline.store.item("leave.md")
    assert (item.status, item.error_code) == (SourceItemStatus.FAILED, "source_changed")
    assert pipeline.index.points == {}
