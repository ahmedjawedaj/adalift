# Security Policy

## Supported versions

Only the latest release and `main` are supported during research alpha.

## Reporting

Do not open public issues for vulnerabilities involving arbitrary code execution, unsafe checkpoint loading, credential leakage, CI secrets, or dependency compromise. Contact the maintainers privately through the repository security advisory mechanism once the GitHub repository is created.

## Checkpoint safety

PyTorch pickle checkpoints can execute code when loaded from untrusted sources. AdaLiFT's training checkpoints are intended for trusted local artifacts. Public weight releases should use `safetensors` wherever possible and publish hashes.
