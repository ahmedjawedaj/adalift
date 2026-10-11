# Review candidate and public release checklist

- [x] Independent review of the RC2 source ZIP hash and implementation.
- [ ] RC3 follow-up commit/tag and source ZIP identity verified after cleanup.
- [ ] GitHub CI green on Python 3.11 and 3.12, including installed distributions.
- [ ] Minimum PyTorch 2.13 compatibility checked; dependencies audited.
- [ ] Endpoint, heterogeneous compaction, order and actual count regressions pass.
- [ ] Safe checkpoint, optimizer/EMA resume and precision regressions pass.
- [ ] Wheel/sdist build, clean install, suite and CLI smoke runs pass.
- [ ] Repository metadata, security contact, citation and provenance reviewed.
- [ ] Review findings resolved or explicitly accepted before making the repo public.

The repository remains private during review. Software release readiness is
separate from research reproduction. CUDA/GPU behavior, ImageNet metrics and
adaptive quality/compute advantages remain unvalidated.
