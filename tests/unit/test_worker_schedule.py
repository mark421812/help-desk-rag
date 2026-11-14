from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from helpdesk.presentation.worker.cli import build_parser
from helpdesk.presentation.worker.runner import DailySchedule
from tests.ingestion_fakes import Pipeline

TAIPEI_TWO_AM = DailySchedule(hour=2, minute=0, timezone="Asia/Taipei")


def test_the_next_run_is_two_am_taipei_today_or_tomorrow() -> None:
    before = datetime(2026, 11, 12, 17, 0, tzinfo=UTC)  # 01:00 台北
    after = datetime(2026, 11, 12, 19, 0, tzinfo=UTC)  # 03:00 台北

    assert TAIPEI_TWO_AM.next_run(before) == datetime(2026, 11, 12, 18, 0, tzinfo=UTC)
    assert TAIPEI_TWO_AM.next_run(after) == datetime(2026, 11, 13, 18, 0, tzinfo=UTC)


def test_a_restart_after_a_missed_run_catches_up_once() -> None:
    now = datetime(2026, 11, 13, 1, 0, tzinfo=UTC)  # 09:00 台北

    assert TAIPEI_TWO_AM.missed(None, now)
    assert TAIPEI_TWO_AM.missed(datetime(2026, 11, 12, 10, 0, tzinfo=UTC), now)
    assert not TAIPEI_TWO_AM.missed(datetime(2026, 11, 12, 18, 5, tzinfo=UTC), now)
    assert not TAIPEI_TWO_AM.missed(None, datetime(2026, 11, 12, 17, 0, tzinfo=UTC))


async def test_a_single_file_ingest_does_not_suppress_the_catch_up_scan(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\n第一份文件。", encoding="utf-8")
    pipeline = Pipeline(root=tmp_path)
    pipeline.clock.now = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)  # 11/12 17:00 台北
    await pipeline.scanner().run()

    pipeline.clock.now = datetime(2026, 11, 13, 9, 0, tzinfo=UTC)  # 11/13 17:00 台北
    await pipeline.ingester().run("a.md")

    assert TAIPEI_TWO_AM.missed(await pipeline.scanner().last_completed_at(), pipeline.clock.now)

    await pipeline.scanner().run()

    assert not TAIPEI_TWO_AM.missed(
        await pipeline.scanner().last_completed_at(), pipeline.clock.now
    )


def test_the_cli_offers_the_planned_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["scan"]).allow_mass_deletion is False
    assert parser.parse_args(["ingest", "nested/a.pdf"]).relative_path == "nested/a.pdf"
    assert parser.parse_args(["items", "--status", "metadata_missing"]).status == "metadata_missing"
    assert str(parser.parse_args(["retry", "6a1f0d2e-0000-4000-8000-000000000000"]).job_id)
    assert parser.parse_args(["run", "--until-idle"]).until_idle is True
