---
name: code-change-verification
description: Verify AdaLiFT code, tests, configs, packaging, and agent-skill synchronization after implementation changes.
---

Use after runtime, test, configuration, CI, packaging, or documentation changes that affect developer workflows.

1. Inspect the diff for accidental research-semantic changes.
2. Run targeted tests first, then `make verify` for substantial changes.
3. Confirm `.agents/skills/` and `.claude/skills/` are synchronized.
4. Check deterministic CPU coverage for new core behavior.
5. Ensure optional integrations did not become core dependencies.
6. Check README/docs if public behavior changed.
7. Summarize what was verified and any unverified GPU/scale behavior.
