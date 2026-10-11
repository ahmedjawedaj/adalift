# Repository discovery and presentation

This document records the repository's public presentation and the remaining publication steps. Discovery depends on useful, accessible content and community interest; descriptions, topics, and visuals do not guarantee rankings, AI citations, stars, or contributions.

## Canonical description

> PyTorch research implementation of LiFT (Loop Flow Transformers) with adaptive recurrent depth, endpoint-preserving inference, active batching, and reproducible CPU examples.

Use the repository URL as the canonical project destination: <https://github.com/ahmedjawedaj/adalift>. Avoid creating competing pages with contradictory release or performance claims.

## Repository topics

`pytorch`, `flow-matching`, `diffusion-transformers`, `loop-flow-transformers`, `flow-transformers`, `recurrent-transformers`, `adaptive-computation`, `test-time-compute`, `adaptive-inference`, `generative-models`, `deep-learning`, `machine-learning`, `reproducible-research`, `computer-vision`, `transformers`.

These describe actual implementation components and research questions. Package metadata carries a smaller matching set. Do not add unrelated trending tags or claims of validated acceleration.

## Search and answer visibility

- **SEO:** the README names the method, framework, problem, implementation status, and runnable entry points. GitHub description and topics supply repository search metadata.
- **AEO:** the FAQ gives direct answers to likely questions about the method, CPU use, official status, pretrained weights, and adaptive speed claims.
- **GEO:** clear definitions, links to the original paper, code, protocols, and limitations give readers and answer systems traceable evidence. Keep facts consistent across releases.
- Diagrams have descriptive alt text and accompanying prose. Important explanations also appear as text.
- Contribution entry points and issue templates help a visitor take a concrete next step. Invite useful participation; avoid automated starring, spam promotion, or keyword stuffing.

Google's [guidance for AI search experiences](https://developers.google.com/search/docs/appearance/ai-features) describes the continued relevance of normal search fundamentals. GitHub documents [repository topics](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics) and [README presentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes).

## Visual assets

- [Social preview PNG](assets/adalift-social-preview.png): 1280 × 640, solid background; also the README hero.
- [Editable social preview SVG](assets/adalift-social-preview.svg): source for the PNG.
- [Adaptive inference SVG](assets/adaptive-inference.svg): endpoint completion and active-row compaction.

These are explanatory diagrams, not benchmark plots. To regenerate the PNG with an external SVG renderer, preserve its 1280 × 640 dimensions and opaque background. The renderer is a documentation tool, not a package dependency.

## Publication checklist

1. Complete the [release checklist](RELEASE_CHECKLIST.md), independent review, and hosted CI for the final commit.
2. Review and commit the documentation, example, metadata, and visual changes alongside the intended release work.
3. Push the reviewed commit. Confirm the GitHub description and topics match the canonical values above.
4. Make the repository public only when publication is authorized. While private, search and cloning are limited to users with access; public discovery cannot be evaluated.
5. Under **Settings → General → Social preview**, upload `docs/assets/adalift-social-preview.png`. GitHub currently allows a first social preview upload only for public repositories; private repositories require a previously uploaded image. See [GitHub's preview instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview).
6. Check the public README, diagram rendering, clone/install commands, issue templates, and preview card. Ensure release links and citation metadata identify the reviewed version.
7. Share a concise technical introduction with the paper relationship, CPU demo, limitations, and specific requests for review or experiments. Publish only claims backed by recorded evidence.

Revisit the FAQ and status paragraph when checkpoints, GPU validation, or reproduction results become available. Keep performance claims tied to a config, commit, hardware, quality protocol, and measured compute.
