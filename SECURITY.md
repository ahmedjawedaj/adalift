# Security Policy

## Supported versions

Only the latest release and `main` are supported during research alpha.

## Reporting

Report vulnerabilities privately to the maintainer at **ahmed.jawed32@gmail.com**.
Include the affected commit/version and a minimal reproduction. Do not disclose
executable payloads in public issues. Once public private-vulnerability reporting
is enabled, use the repository's Security tab as an alternative route.

## Checkpoint safety

AdaLiFT requires PyTorch 2.13 or newer and loads state dictionaries with
`weights_only=True`, without an unsafe fallback. The loader rejects unapproved
pickle globals; this does not make arbitrary downloaded files safe against
resource exhaustion or all framework vulnerabilities. Keep PyTorch patched,
verify artifact provenance and hashes, and keep checkpoint metadata to plain
Python values. Public weights should use `safetensors` when practical.

The dependency floor was raised during the 2026-10-11 review because the
dependency audit reported later loader and memory-safety vulnerabilities in
older PyTorch versions. Both the minimum 2.13.0 and locked 2.14.1 reported no
known vulnerabilities at that check. Dependency audits are time-specific.
