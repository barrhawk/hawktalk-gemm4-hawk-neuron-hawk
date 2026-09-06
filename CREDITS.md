# Credits & Attribution

This repository stands on other people's published work. The list below is not a
formality — the working ports here would not exist without the recipe and toolchain
credited first, and naming them precisely is how a serious shop works.

## Acknowledgments

With thanks to the **[internet.dev](https://internet.dev)** collective and community for
their support and camaraderie. Good work is easier, and better, in good company.

Special thanks to **Jimmy Lee** ([@wwwjim](https://x.com/wwwjim)) — internet.dev's
janitor-in-chief — for the encouragement and for keeping the lights on for builders.

## William McLean — `xbill9`

**The gemma4-on-Inferentia2 wrapper recipe is his.**

- GitHub: https://github.com/xbill9
- Hugging Face: https://huggingface.co/xbill9
- Google Developer Expert and AWS Community Builder

William McLean (xbill9) published the original gemma4-on-Inferentia2 wrapper recipe under
**Apache-2.0** — the wrap-HF-eager approach: wrap the reference eager implementation,
scatter KV into device-resident buffers via `input_output_aliases`, hand-load the
`layer_scalar` buffers, keep softcap and embeddings off-device, plus his `DenseExperts`
and `SPMD` scatter routing for the MoE. Every gemma4 model we run coherently on Neuron
sits on that recipe.

HawkTalk's contribution is the layer built on top: the sliding-window chunked-prefill
long-context path, the correctness discipline (SEQ_MATCH-at-many-positions), and the 12B
dense and 26B-A4B MoE per-model adaptations. His original, unmodified scripts are included
verbatim under `reference/xbill9/` for reference and attribution — see
`reference/README.md`. Those files carry his copyright and his Apache-2.0 license, and
were not authored by us. Any downstream release, write-up, or talk that comes out of this
work names him first.

Public sources: `xbill9/gemma-4-E2B-it-inferentia2` and the E4B repository on Hugging Face.

## Google — gemma4 base weights

The base model weights are **Google gemma4**.

- Model cards: https://ai.google.dev/gemma
- Weights are governed by Google's Gemma license and terms of use.

This repository ships **no weights** — only recipes and scripts. You obtain the weights
yourself from Google under Google's terms. The Apache-2.0 `LICENSE` in this repository
covers our code only, never the weights.

## AWS Neuron SDK

The toolchain that makes any of this run on Inferentia2 / Trainium:

- **AWS Neuron SDK** — https://awsdocs-neuron.readthedocs-hosted.com/
- **NxD Inference (NeuronX Distributed Inference)** — the serving framework
- **neuronx-cc** — the Neuron compiler

### Filed compiler contribution

An `islpy` dependency-drift compiler failure encountered during this work was root-caused
and filed upstream as **[aws-neuron/aws-neuron-sdk#1391](https://github.com/aws-neuron/aws-neuron-sdk/issues/1391)**.
See `docs/islpy-neuronxcc-2.27-bug.md` for the root-cause analysis and the one-line fix.

## Summary of what is whose

| Component                            | Owner / License                                     |
|--------------------------------------|-----------------------------------------------------|
| Recipes & scripts in this repo       | HawkTalk — Apache-2.0 (`LICENSE`)                   |
| Wrapper recipe (the core technique)  | William McLean / xbill9 — Apache-2.0                |
| `reference/xbill9/` scripts          | William McLean / xbill9 — Apache-2.0, his copyright |
| gemma4 weights                       | Google — Gemma license (not shipped here)           |
| Neuron SDK / NxDI / neuronx-cc       | AWS — see AWS Neuron licensing                      |
