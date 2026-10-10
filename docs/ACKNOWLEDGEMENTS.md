# Research and implementation provenance

This repository describes itself as an independent implementation from published
methods. The review remediation does not establish historical code provenance
or compatibility with unreleased author code. No external implementation was
copied during this remediation.

- [LiFT: Loop Flow Transformers](https://arxiv.org/abs/2610.05538): recurrent architecture,
  continuous depth supervision and experimental protocol.
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
  (Peebles and Xie): latent patch transformer and adaLN-zero conditioning conventions.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747): flow-matching framework.
- [RoFormer](https://arxiv.org/abs/2104.09864): rotary positional embeddings.
- [SiT](https://arxiv.org/abs/2401.08740): related flow/interpolant transformer research.

The LiFT paper identifies SpeedrunDiT as its block-design source. Exact parity
with that implementation remains unverified. Citations acknowledge research
ideas; they do not imply copied code or automatically confer code-license rights.
Before importing any third-party implementation, record its source, license,
modifications, and any required notices. The software is licensed Apache-2.0;
the original research papers and runtime dependencies retain their own licenses.
