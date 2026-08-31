from __future__ import annotations

import re
from pathlib import Path

from resnet_repro.config import load_frozen_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "cifar10_plain20_resnet20_frozen.yaml"


def _approved_decisions_from_log() -> set[str]:
    approved: set[str] = set()
    text = (ROOT / "docs" / "decision_log.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 7 and fields[1].startswith("DEC-") and fields[5] == "APPROVED_BY_USER":
            approved.add(fields[1])
    return approved


def test_frozen_decisions_exactly_match_approved_decision_log() -> None:
    config_decisions = set(load_frozen_config(CONFIG).data["approval"]["decision_ids"])
    assert config_decisions == _approved_decisions_from_log()
    assert len(config_decisions) == 18


def test_phase0_closeout_decisions_are_synchronized_to_all_required_docs() -> None:
    paths = [
        ROOT / "AGENTS.md",
        ROOT / "docs" / "decision_log.md",
        ROOT / "docs" / "assumptions.md",
        ROOT / "docs" / "approval_summary.md",
        ROOT / "docs" / "implementation_plan.md",
        ROOT / "docs" / "test_specification.md",
        ROOT / "docs" / "source_traceability.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "DEC-BN-001C" in text, path
        assert "DEC-RNG-002" in text, path


def test_approval_summary_records_exact_frozen_config_sha256() -> None:
    config = load_frozen_config(CONFIG)
    summary = (ROOT / "docs" / "approval_summary.md").read_text(encoding="utf-8")
    assert f"Current SHA-256：`{config.sha256}`" in summary


def test_test_specification_ids_are_unique() -> None:
    text = (ROOT / "docs" / "test_specification.md").read_text(encoding="utf-8")
    test_ids = re.findall(r"^\| ([A-Z]+-\d{3}) \|", text, flags=re.MULTILINE)
    assert test_ids
    duplicates = {test_id for test_id in test_ids if test_ids.count(test_id) > 1}
    assert not duplicates
