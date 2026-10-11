# Maintainer guide

## Publication policy and live verification

GitHub Free supports these protections for public repositories; private repositories require a supporting paid plan. Protection payloads are stored below, but their presence in Git does not enable a GitHub setting. Verify the live API responses before merging or releasing. Publication evidence is recorded in the GitHub release notes. See [GitHub's protection availability](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

GitHub repository settings now allow squash merges only and delete finished branches automatically. Issues and forks are enabled. Dependabot vulnerability alerts are enabled. Actions has a read-only default token, cannot approve PR reviews, and requires external Actions to be pinned to full commit SHAs. Recheck these live settings at publication; a file in the repository does not apply a GitHub setting.

The current sole administrator/code owner is `@ahmedjawedaj`. The owner has authorized a review bypass so they can merge their own PRs without another approving reviewer. GitHub does not allow authors to approve their own PRs; use the documented bypass after self-review. A second maintainer is optional under this policy.

## Main branch policy

Two configurations work together. **Apply both.** The baseline protects CI and history without an administrator bypass; a separate review ruleset permits the owner/admin to bypass approvals only through a PR merge.

### Mandatory CI and history

[.github/protection/main.json](../.github/protection/main.json) requires:

- All three quality checks, bound to the observed GitHub Actions application (ID `15368`): `quality (3.11, locked)`, `quality (3.12, locked)`, and `quality (3.11, 2.13.0)`.
- A branch up to date with `main`, resolved review conversations, and linear history.
- No force pushes or branch deletion. Apply these restrictions to administrators too.

This payload has `required_pull_request_reviews: null`: review requirements live in the separate ruleset below. These check names and application ID were read from the successful hosted run for `a4a4cf3`. If workflow job names change, update the protection payload before merging the rename.

Inspect existing rules before applying the payload; the PUT endpoint replaces branch protection. Once the repository is public or the account supports private protection:

```bash
gh api repos/ahmedjawedaj/adalift/branches/main/protection
gh api --method PUT repos/ahmedjawedaj/adalift/branches/main/protection \
  --input .github/protection/main.json
gh api repos/ahmedjawedaj/adalift/branches/main/protection
```

A first GET can return 404 for an unprotected branch. A 403 is not evidence that protection is active. Verify checks, strict status checking, administrator enforcement, conversation resolution, and force-push/deletion restrictions in the final response.

### Required PRs and owner review bypass

[.github/protection/main-review.json](../.github/protection/main-review.json) requires a PR and one approving review, dismisses stale approvals, and targets `refs/heads/main`. The administrator role (ID `5`) has `bypass_mode: pull_request`. This permits an administrator to bypass the review requirement when merging a PR, while requiring a PR trail. It does not exempt them from the separate CI/history protection.

```bash
gh api repos/ahmedjawedaj/adalift/rulesets
gh api --method POST repos/ahmedjawedaj/adalift/rulesets \
  --input .github/protection/main-review.json
gh api repos/ahmedjawedaj/adalift/rulesets
```

Create this ruleset once, or update its existing ID rather than duplicating it. Inspect the returned ID with `gh api repos/ahmedjawedaj/adalift/rulesets/RULESET_ID`. Verify active enforcement, the exact branch target, one required approval, and the administrator PR-only bypass. Confirm both this ruleset and baseline protection are applied before accepting external contributions.

Only `@ahmedjawedaj` currently has administrator access. GitHub's bypass is role-based, applies to any PR merged by an administrator, and is not limited by PR authorship. Use the exception for owner-authored work; review external contributions normally. Granting another user administrator access also grants this bypass. CODEOWNERS still requests owner review on changes.

To merge your own PR: inspect the diff, wait for all required CI checks on the latest source, bring the branch up to date, resolve review conversations, then use GitHub's bypass option for squash merge. Do not attempt a self-approval or use an administrative merge option to skip the separate CI gate. Review bypass support is documented in [GitHub's ruleset instructions](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository).

## Release tags

[.github/protection/release-tags.json](../.github/protection/release-tags.json) defines an active ruleset for `v*` tags that restricts updates and deletion. Administrators have an explicit emergency bypass; contributors do not. This is not an absolute immutability guarantee. Normal releases create a new version/tag instead of rewriting an existing tag.

```bash
gh api repos/ahmedjawedaj/adalift/rulesets
gh api --method POST repos/ahmedjawedaj/adalift/rulesets \
  --input .github/protection/release-tags.json
gh api repos/ahmedjawedaj/adalift/rulesets
```

Create the ruleset once. If a matching ruleset exists, inspect and update its ID rather than creating a duplicate. After creation, inspect that ID with `gh api repos/ahmedjawedaj/adalift/rulesets/RULESET_ID` to verify its pattern, rules, bypass, and enforcement.

## Public fork CI and security

- Approve workflow runs from fork contributors only after checking the diff, including workflow changes, dependencies, build scripts, and executable hooks. Use the public fork approval policy `all_external_contributors` at publication.
- Keep `GITHUB_TOKEN` read-only and automatic Actions PR approvals disabled. Do not pass secrets into fork jobs or use `pull_request_target` to execute contributor code.
- Pin external Actions to full commit SHAs. Checkout does not persist credentials. The quality matrix includes minimum and locked PyTorch, smoke training, and installed-distribution checks.
- Dependency audits run on pushes to `main`, PRs, and weekly schedules. Monitor audit failures; the online vulnerability database can change independently of a PR. The audit is not currently a required branch check; maintainers must review a failed audit before merging or releasing.
- Keep Dependabot updates enabled for Python packages and Actions. Review updates rather than merging automatically.
- Once public, enable private vulnerability reporting and verify it succeeds. Until then, [SECURITY.md](../SECURITY.md) provides the private email route. Review secret-scanning and push-protection availability in GitHub Security settings; verify their enabled state rather than assuming public visibility enables every control.

Public-only API activation commands:

```bash
gh api --method PUT \
  repos/ahmedjawedaj/adalift/actions/permissions/fork-pr-contributor-approval \
  -f approval_policy=all_external_contributors
gh api repos/ahmedjawedaj/adalift/actions/permissions/fork-pr-contributor-approval
gh api --method PUT repos/ahmedjawedaj/adalift/private-vulnerability-reporting
gh api repos/ahmedjawedaj/adalift/private-vulnerability-reporting
```

GitHub rejects the fork approval endpoint on private repositories. Run these commands for a public repository and verify their responses; documentation alone does not activate the controls.

## Release procedure

1. Review the complete diff, including checkpoint fixes, documentation, CI, canonical license, and metadata. Run `make verify` and `python scripts/verify_distribution.py` on the intended source.
2. Choose a fresh RC version/tag for the final reviewed candidate. Existing `v0.1.0rc2` and `v0.1.0rc3` tags identify earlier private review candidates; the first public candidate is `v0.1.0rc4`. Do not rewrite previous tags. Update the central package version, citation, and changelog together.
3. Commit and push the reviewed tree through a PR. Require hosted checks on that exact commit, not the earlier baseline. Until protections can be enforced, manually follow the same review and CI policy.
4. Publish only with the owner's explicit authorization. On GitHub Free, immediately activate and read back main/tag protections and the public-only security settings when visibility changes. Avoid merging external PRs during that activation window.
5. Verify the public README, examples, issue/PR templates, license detection, citation, and social preview. Use the [discovery checklist](DISCOVERABILITY.md).
6. Tag the verified commit with the new version, build wheel/sdist, test clean installations, and record artifact SHA-256 hashes plus source commit in the release evidence. Publish an alpha release with the known GPU and research limitations.

No production stability, ImageNet/FID reproduction, or adaptive speed/quality advantage is implied by passing the software checks. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
