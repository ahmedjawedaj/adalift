# Governance

AdaLiFT uses a maintainer-led, evidence-first governance model during research alpha.

The current maintainer and release owner is [Ahmed Jawed (@ahmedjawedaj)](https://github.com/ahmedjawedaj). Contact: `ahmed.jawed32@gmail.com`. Repository permissions, protection settings, and the publication procedure are documented in [the maintainer guide](docs/MAINTAINER_GUIDE.md).

- Maintainers decide roadmap and release scope.
- Significant architecture changes require an issue or design note before merge.
- Scientific disagreements are resolved by reproducible experiments whenever possible.
- Compatibility-breaking changes are allowed before 1.0 but must be documented in `CHANGELOG.md`.
- External contributions require one approving review from a trusted reviewer other than the author, passing required CI, and resolved review conversations. Review must cover scientific assumptions for model, loss, sampler, and experiment changes.
- The owner/admin may bypass the approval requirement to merge their own PR after self-review. A PR, passing required CI, and resolved conversations remain mandatory. This is an explicit owner-authorized exception; a second reviewer is optional for owner-authored work. Code owners route review requests. An agent's local report is not a GitHub approval.
- Maintainers merge accepted PRs using squash merge. Release authority remains with the named release owner. Grant write access only to trusted maintainers; contributors normally work through forks.
- Security reports follow `SECURITY.md` and should not be opened publicly before triage.

As the contributor base grows, add named maintainers and explicit nomination rules. Document changes to decision-making before adopting them. Research alpha does not promise a fixed response time or long-term API stability.
