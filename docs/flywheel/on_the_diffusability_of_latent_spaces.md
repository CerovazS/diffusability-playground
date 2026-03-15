# On the Diffusability of Latent Spaces

## Summary

Recent work has shown that standard VAE latent spaces are often poorly suited for diffusion models due to misalignment with semantic representations and unfavorable perception-rate-distortion trade-offs. Current improvements mainly focus on aligning latent spaces with pretrained representation models and shaping their frequency structure, as evidence suggests that latents dominated by lower frequencies lead to better semantic structure and faster diffusion convergence. In this context, *diffusability* informally refers to how easily diffusion transformers can model a latent space.

However, the concept remains poorly defined and insufficiently studied. Existing work is largely vision-focused, task-specific, and lacks a rigorous geometric interpretation of why certain latent structures improve diffusion. Key open questions include whether semantic alignment is fundamentally geometric, whether probing methods truly measure semanticity or diffusability, how these properties emerge during training, and how diffusability relates to memorization and generalization in diffusion transformers.

This work proposes a systematic study of latent space diffusability, starting with controllable synthetic datasets. By generating point-cloud datasets with adjustable properties (intrinsic dimension, multimodality, curvature, and distribution tails), diffusion models can be evaluated independently of decoders using precise geometric and distributional metrics. Insights from these controlled experiments will then be tested on real audio autoencoders, leveraging existing latent VAE checkpoints. The goal is to move toward a clearer definition and measurable understanding of diffusability in generative latent spaces.
