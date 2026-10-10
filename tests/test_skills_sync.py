from pathlib import Path


def test_agent_skill_manifests_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    codex = root / ".agents" / "skills"
    claude = root / ".claude" / "skills"
    codex_manifests = sorted(p.relative_to(codex) for p in codex.glob("*/SKILL.md"))
    claude_manifests = sorted(p.relative_to(claude) for p in claude.glob("*/SKILL.md"))
    assert codex_manifests
    assert codex_manifests == claude_manifests
