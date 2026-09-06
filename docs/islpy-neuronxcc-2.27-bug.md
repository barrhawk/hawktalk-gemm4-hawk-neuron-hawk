# neuronx-cc 2.27 compiler crash at seq_len > 4096: `islpy` dependency drift breaks `BasicSet.is_subset()`

**TL;DR** — If `neuronx-cc` 2.27 fails to compile any graph whose sequence length exceeds 4096 with an internal `is_subset()` assertion (`NCC_ISMP902`), you have the wrong `islpy` installed. The compiler pins `islpy~=2026.1`, but pip resolves the newer `islpy==2026.2.1`, which removed an implicit type downcast the compiler relies on. Pin it back:

```bash
pip install islpy==2026.1
```

That is the entire fix. The rest of this document explains why, so you can recognize it and so it can be fixed upstream.

Tracked upstream as **[aws-neuron/aws-neuron-sdk#1391](https://github.com/aws-neuron/aws-neuron-sdk/issues/1391)** (reporter: barrhawk).

---

## Symptom

Compiling a model with `neuronx-cc` (via NxD Inference / `torch_neuronx.trace`) succeeds for short sequence buckets and fails only once the compiled sequence length crosses **4096**. Small buckets (e.g. `seq2048/ctx1024`) compile cleanly; a larger bucket aborts.

The failure surfaces inside the compiler's integer-set-manipulation (ISMP) pass, not in your model code, with a message of the form:

```
NCC_ISMP902: internal compiler error in is_subset()
  ...
TypeError: BasicSet.is_subset() argument must be a BasicSet, not Set
```

Key tells that this is the drift bug and not a real modeling error:

- The **same source and same flags** compile fine at seq ≤ 4096 and fail only above it.
- The crash is inside the compiler's polyhedral/affine-set machinery (`is_subset`, `BasicSet`, `Set`), never in a Neuron op or your traced graph.
- It appeared without any change to your code, `neuronx-cc`, or the Neuron SDK — i.e. after a fresh `pip install` into a new venv where pip pulled a newer `islpy`.

## Why 4096 is the threshold

`neuronx-cc` only invokes the `islpy` (Integer Set Library) code path for the larger affine domains that appear when the sequence dimension is big enough to force multi-tile / chunked loop nests. Below ~4096 the schedule stays within simpler bounds that never hit the `is_subset()` comparison in question, so the wrong `islpy` version is harmless. Above it, the compiler asks `islpy` whether one integer set is a subset of another — and that call is where the API changed.

## Root cause: a dropped implicit downcast in `islpy` 2026.2.1

`islpy` is the Python binding for ISL. It exposes two related set types:

- `BasicSet` — a single convex integer polyhedron.
- `Set` — a union of `BasicSet`s (a `BasicSet` is the one-piece special case).

`neuronx-cc` 2.27 calls, in effect:

```python
basic_set.is_subset(some_set)   # basic_set: BasicSet, some_set: Set
```

- In **`islpy` 2026.1** (the version the compiler was built and tested against), `BasicSet.is_subset(Set)` **auto-downcast the receiver**: it promoted the `BasicSet` `self` to a `Set` and performed the comparison. The call returned a `bool`.
- In **`islpy` 2026.2.1** (published *after* neuronx-cc 2.27 shipped), that convenience downcast was **removed**. `BasicSet.is_subset()` now requires its argument to match, and the mixed `BasicSet`/`Set` call raises `TypeError` instead of quietly coercing. The compiler's uncaught call becomes the `NCC_ISMP902` internal error.

## The dependency drift

The bug is not really in `islpy` — the new, stricter behavior is defensible. It is a **version-pin drift**:

- `neuronx-cc` 2.27 declares `islpy~=2026.1`.
- The `~=2026.1` compatible-release specifier permits **any** `2026.*` that is `>= 2026.1` — including `2026.2.1`.
- `islpy 2026.2.1` was published **after** neuronx-cc 2.27, so at build/test time the compiler only ever saw `2026.1`. A later fresh install resolves the newest compatible release and silently upgrades the compiler's ISL binding to one with a changed API.

Result: an unchanged compiler binary gets a subtly different `islpy` under it, and breaks only on the code path (seq > 4096) that exercises the changed method.

## Diagnosis

Check the installed version:

```bash
python -c "import islpy; print(islpy.__version__)"
```

- `2026.1` → not this bug.
- `2026.2.1` (or any `2026.2.*`) alongside `neuronx-cc` 2.27 → this bug.

## Workaround (one line)

Pin `islpy` back to the version the compiler was built against, in the venv that holds `neuronx-cc`:

```bash
pip install islpy==2026.1
```

Then recompile. No source, flag, or SDK change is needed. Verified: the exact same graph and flags that aborted with `NCC_ISMP902` at seq > 4096 under `islpy 2026.2.1` compile cleanly under `islpy 2026.1`.

To make it stick across fresh environments, pin it explicitly in your requirements ahead of `neuronx-cc`'s loose spec:

```
islpy==2026.1
```

## Suggested upstream fix

Any one of these closes the drift permanently; the first is the smallest and most correct:

1. **Tighten the pin** in `neuronx-cc` 2.27's dependency metadata from `islpy~=2026.1` to `islpy==2026.1` (or `>=2026.1,<2026.2`), so pip cannot pull a post-release `islpy` the compiler never saw.
2. **Make the compiler call type-agnostic** — coerce the receiver before comparing, so it works under both the old and new `islpy` API:
   ```python
   # instead of basic_set.is_subset(other)
   other.to_set()  # or wrap: isl.Set.from_basic_set(basic_set).is_subset(other)
   ```
   i.e. promote the `BasicSet` to a `Set` explicitly rather than relying on the removed implicit downcast.
3. **Catch and report** the `TypeError` in the ISMP pass with an actionable message naming the `islpy` version mismatch, instead of surfacing a bare `NCC_ISMP902` internal error.

## Reference

- Upstream issue: **aws-neuron/aws-neuron-sdk#1391** — `neuronx-cc 2.27 NCC_ISMP902 is_subset() failure at seq_len>4096 due to islpy 2026.2.1 drift` (reporter: barrhawk).
- Affected toolchain: AWS Neuron SDK, NxD Inference, `neuronx-cc` 2.27, on `inf2`.
- `islpy` project: the Python binding for the Integer Set Library (ISL).

---

*This writeup is offered so any Neuron user who hits `NCC_ISMP902` at long sequence lengths can identify and fix it in one command. The one-line pin (`pip install islpy==2026.1`) is the fix until the compiler's dependency spec is tightened upstream.*
