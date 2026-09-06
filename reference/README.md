# reference/

This directory holds third-party material included for **reference and
attribution**. Nothing in `reference/` was authored by this project.

## reference/xbill9/

`reference/xbill9/` contains **William McLean's (xbill9) original,
published scripts, included verbatim** — the gemma-4-on-Inferentia2
wrapper recipe this project builds on.

- Author: William McLean — https://github.com/xbill9 — https://huggingface.co/xbill9
- Source: HuggingFace repos `xbill9/gemma-4-*-inferentia2`,
  branch `gemma4-inf2-nxd-kvshare`
- License: **Apache-2.0**, William McLean's copyright

These files are here **unmodified** so that:

1. His original recipe is preserved and properly attributed.
2. Readers can diff our ports against the exact source they build on.

**Do not edit files in `reference/xbill9/`.** They are a snapshot of his
published work, not our code. Our scripts (the 12B and 26B ports, the SWA
sliding-window + chunked-prefill work, the serving backend) live at the
top level of the repo and are licensed under our own Apache-2.0 `LICENSE`.

See `../CREDITS.md` for the full attribution.
