# Release verification checklist

Copy this checklist into each release's evidence record and close it against the
specific source commit. These unchecked items are a reusable template; the
published release notes record completed checks and live protection status.

- [ ] Independent review findings resolved or explicitly accepted.
- [ ] Final version, citation, changelog, tag and source commit agree; existing tags are unchanged.
- [ ] Hosted CI green on the exact commit: Python 3.11/3.12 and minimum PyTorch.
- [ ] Current dependency audit reviewed; endpoint, compaction, count and checkpoint regressions pass.
- [ ] Wheel/sdist clean installs, full suites, CLI smoke runs and metadata checks pass.
- [ ] README, examples, canonical Apache-2.0 license, provenance and citation reviewed.
- [ ] Contribution/issue/PR templates, named maintainer, conduct contact and code-owner routing present.
- [ ] Main baseline protection and review ruleset read back: owner PR-only review bypass, mandatory Actions checks for everyone, current branch, resolved conversations, linear history, no force push/deletion.
- [ ] Release-tag ruleset read back, including its documented administrator emergency bypass.
- [ ] Fork workflow approval, private vulnerability reporting, dependency alerts and Actions settings verified.
- [ ] Public README, clone/install commands, issue forms, license detection and social preview checked.
- [ ] Release artifact hashes, source commit, CI links, and limitations recorded.

Software release readiness is separate from research reproduction. CUDA, MPS,
distributed behavior, ImageNet/FID results and adaptive quality/latency advantages
remain unvalidated. Do not present synthetic tests as benchmark reproduction.

The 2026-10-12 pre-publication follow-up passed 94 CPU tests with one CUDA skip and
89.88% coverage, including independent clean-install distribution checks. See
[PUBLIC_RELEASE_REVIEW.md](PUBLIC_RELEASE_REVIEW.md) for the historical review and
[MAINTAINER_GUIDE.md](MAINTAINER_GUIDE.md) for activation and verification commands.
