# Public release review — 2026-10-12

This is the **pre-publication review record**. Its pending findings describe the
working tree and GitHub settings at review time. The public RC4 release includes
this reviewed source; use the GitHub release notes and live protection API
responses for final commit, CI, artifact identity, and activation evidence.

## Recommendation

The current software is suitable for a public **research alpha** after the final source is committed, exact-commit hosted CI passes, and the prepared protections are activated and verified. Do not describe the currently private GitHub repository as fully protected or the uncommitted fixes as published. No new release-blocking runtime defect was found in the checkpoint follow-up.

This is an independent Codex review of the local tree in `~/Downloads/adalift`, not a Claude/Fable model attribution. Fable's three changed files (`checkpoint.py`, `test_checkpoint_transaction.py`, `CHANGELOG.md`) were inspected but not edited; their SHA-256 fingerprints remained unchanged. HEAD remains `a4a4cf3dea56df57e26a76fe6d9746ddb83f393c`, with both existing RC tags untouched.

## Findings, ranked by release impact

| Priority | Finding | Status / evidence |
| --- | --- | --- |
| P2 | Branch protection and tag rules cannot currently be enforced on this private repository/account. | Pending. GitHub protection and ruleset endpoints returned HTTP 403 requesting GitHub Pro or public visibility. Prepared payloads are [.github/protection/main.json](../.github/protection/main.json), [main-review.json](../.github/protection/main-review.json), and [release-tags.json](../.github/protection/release-tags.json); they are not live rules. |
| P2 | The complete candidate is uncommitted; successful hosted CI checks an earlier source. | Pending. [Hosted run](https://github.com/ahmedjawedaj/adalift/actions/runs/38163424180) verifies `a4a4cf3`, excluding the follow-up, presentation work, license correction, and contributor/CI updates. Require green checks on the final reviewed commit. |
| P2 | LICENSE differed from canonical Apache-2.0, including the parenthetical patent-litigation wording. | Resolved locally; not yet pushed. GitHub's current license detection reports `Other` / `NOASSERTION`. Restored the [canonical Apache text](https://www.apache.org/licenses/LICENSE-2.0), preserving the declared license and contributor copyright. Recheck GitHub detection after push. |
| P3 | Maintainer identity, code-owner routing, conduct contact, and the external PR workflow were incomplete. | Resolved locally in [GOVERNANCE.md](../GOVERNANCE.md), [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md), [CONTRIBUTING.md](../CONTRIBUTING.md), [.github/CODEOWNERS](../.github/CODEOWNERS), and the issue/PR templates. |
| P3 | Dependency audits ran only weekly/manually, and merge cleanup and SHA pinning were not enforced as repository settings. | Local CI now audits PRs and pushes as well as weekly, uses CPU Torch and a pinned audit tool, and checkout does not persist credentials. Live GitHub settings now enable vulnerability alerts, squash-only merges, branch cleanup, and full-SHA Actions enforcement. |

Private vulnerability reporting returned 404 while private. Public fork approval settings returned 422 because the repository is private. Both are documented as pending public-only activation in the [maintainer guide](MAINTAINER_GUIDE.md); the private security email route already exists.

## Checkpoint follow-up acceptance

- [Step-counter validation](../src/adalift/training/checkpoint.py#L248) rejects booleans, complex values, NaN, infinity, negative values, and tensors with more than one element before application. Valid Python numeric and one-element tensor counters remain accepted.
- [Extra-state validation](../src/adalift/training/checkpoint.py#L193) permits safe module extra state, checks tensor/non-tensor compatibility, and delegates content validation to the owning module. The invalid-dictionary test exercises rollback after `set_extra_state` rejects it.
- The integer-counter test now binds the restored optimizer to the restored model. It checks weights, counter value, successful stepping, and finite parameters. Numeric-to-tensor conversion by PyTorch is correctly treated as valid behavior.
- Known optimizer moments must match their parameter shapes, including scalar parameters; factored optimizer statistics remain accepted.
- `weights_only=True` and the existing model/optimizer/EMA/RNG rollback remain in place. Custom hooks with irreversible external side effects remain outside the stated guarantee; optimizer-specific unknown state keys are not exhaustively validated.

Independent reproduction: the current 22-test checkpoint transaction file gives **12 failed, 10 passed on unpatched `a4a4cf3`**, and **22 passed on the patched source**. Source overrides were confined to a temporary fixture; the working tree was not reverted or patched for that comparison.

## Verification

| Check run independently | Result |
| --- | --- |
| `make verify`, macOS, Python 3.12.11 / Torch 2.14.1 | Pass: 94 passed, 1 CUDA skipped, 89.88% coverage |
| `make verify`, macOS, Python 3.11.14 / Torch 2.14.1 | Pass: 94 passed, 1 CUDA skipped, 89.88% coverage |
| Full pytest/coverage, Python 3.11.14 / Torch 2.13.0 | Pass: 94 passed, 1 CUDA skipped, 89.88% coverage |
| `verify_distribution.py`, Python 3.12 | Wheel and sdist clean installs, full suites, and config CLI smoke pass |
| Installed-package dependency audit, pip-audit 2.10.1 | No known vulnerabilities among audited dependencies; 54 distributions enumerated, local unpublished AdaLiFT skipped |
| Basic secret-pattern scan | No matches in 153 historical text blobs and 107 candidate files at scan time; this is not a comprehensive secret detector |
| Workflow/static checks | Read-only token, full-SHA external Actions, no `pull_request_target`, checkout credentials disabled; protection JSON parses; reviewed relative links resolve |

Verified imports in all three local environments resolve to the Downloads source. The optional NumPy initialization warning is present; NumPy is not required for the core checks. External coverage and mypy caches were used. CUDA, MPS, distributed behavior, ImageNet/FID reproduction, and matched-quality adaptive speed or compute advantages remain unverified.

The distribution check preceded these final review notes. Its tested runtime, packaging rules, examples, and CI source are unchanged by the notes. Final release artifacts must be rebuilt from the final committed/tagged source and hashed; local build success does not establish release identity.

## Contribution and release handoff

The [maintainer guide](MAINTAINER_GUIDE.md) records exact commands and verification responses for main/tag protection, public fork approvals, and private vulnerability reporting. The updated main policy requires one independent trusted approval for external contributions, with an owner/admin PR-only approval bypass authorized by the owner after this review. Separate baseline protection keeps all three actual Actions quality checks, an up-to-date branch, resolved conversations, linear history, and no force pushes/deletion mandatory for administrators too. CODEOWNERS routes review requests.

GitHub does not permit self-approval. The owner can instead merge their own PR through the review bypass after self-review and green CI; a second maintainer is optional. The bypass is administrator-role based and technically applies to any administrator-merged PR, so external contributions should still receive normal owner review. No user was invited or granted access by this review. The release tag ruleset has a documented administrator emergency bypass, so tags are not claimed to be absolutely immutable.

For the next implementation/release pass:

1. Accept the checkpoint patch and review the complete documentation, canonical license, workflow, and metadata diff. No additional checkpoint change is requested by this review.
2. Choose a fresh RC version/tag; update central version, citation, and changelog consistently. Do not move the existing RC tags.
3. Commit and push the complete reviewed candidate, then check hosted CI on that exact SHA, including the updated dependency audit and clean distributions.
4. With explicit publication authorization, either enable private protection through GitHub Pro first, or make public and immediately apply/read back the prepared protections and public-only settings. Avoid external merges during activation.
5. Upload the social preview, verify the public contribution/issue templates and license detection, rebuild release artifacts, and record source/tag and SHA-256 hashes.

The result can be presented as a tested research implementation open to contributions. It cannot be presented as an official LiFT implementation, a reproduced paper benchmark, production-ready software, or validated adaptive acceleration.
