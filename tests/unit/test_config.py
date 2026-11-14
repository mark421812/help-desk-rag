from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpdesk.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
COMPOSE_FILE = PROJECT_ROOT / "compose.yaml"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the developer's shell variables and repository `.env` out of these tests."""

    monkeypatch.chdir(tmp_path)
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def test_both_toggles_default_to_enabled() -> None:
    settings = Settings(llm_provider="fake")

    assert settings.access_control_enabled is True
    assert settings.sensitive_data_scan_enabled is True


def test_a_local_env_file_can_turn_both_toggles_off(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=fake\nACCESS_CONTROL_ENABLED=false\nSENSITIVE_DATA_SCAN_ENABLED=false\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.access_control_enabled is False
    assert settings.sensitive_data_scan_enabled is False


def test_prompt_budget_must_fit_a_full_evidence_budget() -> None:
    with pytest.raises(ValidationError, match="MAX_PROMPT_TOKENS"):
        Settings(llm_provider="fake", max_context_tokens=6_000, max_prompt_tokens=12_000)
    with pytest.raises(ValidationError, match="MAX_PROMPT_TOKENS"):
        Settings(llm_provider="fake", max_context_tokens=6_000, max_prompt_tokens=44_191)

    smallest = Settings(llm_provider="fake", max_context_tokens=6_000, max_prompt_tokens=44_192)
    assert smallest.max_prompt_tokens == 44_192


@pytest.mark.parametrize("key", [None, "", "   "])
def test_settings_load_without_a_groq_key_so_the_worker_does_not_need_one(
    key: str | None,
) -> None:
    assert Settings(llm_provider="groq", groq_api_key=key).groq_api_key is None


def test_recovery_must_not_fail_turns_that_may_still_be_running() -> None:
    with pytest.raises(ValidationError, match="TURN_STALE_AFTER_SECONDS"):
        Settings(
            llm_provider="fake",
            rewrite_timeout_seconds=10,
            groq_total_timeout_seconds=75,
            turn_stale_after_seconds=169,
        )

    smallest = Settings(
        llm_provider="fake",
        rewrite_timeout_seconds=10,
        groq_total_timeout_seconds=75,
        turn_stale_after_seconds=170,
    )
    assert smallest.turn_stale_after_seconds == 170


def test_groq_with_a_key_starts_and_keeps_the_key_out_of_its_repr() -> None:
    settings = Settings(llm_provider="groq", groq_api_key="gsk_synthetic_value")

    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "gsk_synthetic_value"
    assert "gsk_synthetic_value" not in repr(settings)


def test_groq_base_url_must_be_an_allowed_https_host() -> None:
    with pytest.raises(ValidationError, match="GROQ_ALLOWED_HOSTS"):
        Settings(
            llm_provider="groq",
            groq_api_key="gsk_synthetic_value",
            groq_base_url="https://attacker.example/openai/v1",
        )


def test_chunk_overlap_must_be_smaller_than_the_target() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        Settings(llm_provider="fake", chunk_target_tokens=350, chunk_overlap_tokens=350)


def test_env_example_documents_every_setting() -> None:
    documented = {
        line.partition("=")[0].strip()
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    compose_variables = set(re.findall(r"\$\{([A-Z_]+)", COMPOSE_FILE.read_text(encoding="utf-8")))

    assert documented == {name.upper() for name in Settings.model_fields} | compose_variables


def test_a_copied_env_example_loads_with_both_toggles_on(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    settings = Settings()

    assert settings.llm_provider == "groq"
    assert settings.groq_api_key is None
    assert settings.groq_allowed_hosts == ["api.groq.com"]
    assert settings.access_control_enabled is True
    assert settings.sensitive_data_scan_enabled is True
