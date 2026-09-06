# coding=utf-8
"""
Gemma4 (text-only) for NxD Inference on Inferentia2.

Port of `Gemma4ForConditionalGeneration.language_model` (transformers 5.x) to the
NxDI NeuronBaseModel contract. Text decoder only: no vision tower, no audio tower,
no MoE (enable_moe_block=false in this checkpoint), no AltUp/LAUREL.

Architecture deltas vs. the NxDI gemma3 module this file is patterned on:
  * mixed head_dim: sliding layers 256, full-attention layers 512 (global_head_dim)
  * NoPE-padded "proportional" RoPE on full-attention layers (partial_rotary_factor=0.25)
  * qk-norm with softmax scaling 1.0 (HF `Gemma4TextAttention.scaling = 1.0`)
  * scale-free v_norm on values
  * per-layer embeddings (PLE) feeding a third residual block in every decoder layer
  * KV sharing: layers >= 15 have no live k/v path; they reuse layer 13/14's K/V
  * double-wide MLP (intermediate 12288) on the KV-shared layers
  * final logit softcapping at 30.0
  * plain-weight RMSNorm: out = norm(x) * w   -- NOT gemma3's (1 + w) convention.
    Nothing in this file or its convert function adds +1.0 to any norm weight.

Traceability: every branch in a forward() below is decided by Python constants fixed
at construction (layer type, donor/consumer role). No branching on tensor values,
no dynamic shapes.
"""

import copy
import math
import logging
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from neuronx_distributed.parallel_layers.layers import (  # noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.flashdecode.utils import get_cache_size
# NxDI 0.8 added create_sampler(); 0.6 (this DLAMI) ships only Sampler.
# The factory just picks DataParallelSampler when sampling_dp_degree>1 and
# batch>1, otherwise returns Sampler(neuron_config). serve16 runs
# sampling_dp_degree=1, so plain Sampler is the same object either way.
try:
    from neuronx_distributed_inference.modules.generation.sampling import create_sampler
except ImportError:  # NxDI < 0.8
    from neuronx_distributed_inference.modules.generation.sampling import Sampler as _Sampler

    def create_sampler(neuron_config, lm_head_tp_degree=None, do_sample=None):
        try:
            return _Sampler(neuron_config, do_sample)
        except TypeError:
            return _Sampler(neuron_config)
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager
from neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager import BlockKVCacheManager
from neuronx_distributed_inference.modules.kvcache.utils import get_kv_shapes

from transformers.activations import ACT2FN

# --- AutoConfig stub for model_type=gemma4 (agent patch, 2026-08-23) ---------
# The DLAMI transformers (4.57.x) predates gemma4, so hf_adapter.load_config
# -> AutoConfig.from_pretrained raises KeyError(gemma4). Gemma4InferenceConfig
# flattens text_config itself and only needs a PretrainedConfig carrying the
# raw json attrs, so a stub registration is sufficient and touches nothing in
# NxDI. On a transformers that already ships gemma4 the register() call raises
# and is skipped.
from transformers import AutoConfig as _HFAutoConfig
from transformers import PretrainedConfig as _HFPretrainedConfig


class _Gemma4StubHFConfig(_HFPretrainedConfig):
    model_type = "gemma4"


try:
    _HFAutoConfig.register("gemma4", _Gemma4StubHFConfig)
except Exception:
    pass
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

SLIDING = "sliding_attention"
FULL = "full_attention"


# -----------------------------------------------------------------------------
# Norms
# -----------------------------------------------------------------------------


class NeuronGemma4RMSNorm(nn.Module):
    """Gemma4 RMSNorm: out = x * (mean(x^2) + eps)**-0.5 * w, computed in fp32.

    Unlike gemma2/gemma3, the weight is used as-is (HF inits it to ones and
    multiplies directly). Do NOT add 1.0 to these weights anywhere."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x):
        out = x.float()
        # torch.pow(mean_squared, -0.5), NOT torch.rsqrt: HF's Gemma4RMSNorm carries
        # an explicit comment that rsqrt/sqrt lower differently across compilers.
        # neuronx-cc maps rsqrt to a hardware reciprocal-sqrt approximation, and
        # these norm weights are large (norm.weight max 118.5), which amplifies the
        # post-scale error. Keep pow.
        mean_squared = out.pow(2).mean(-1, keepdim=True) + self.eps
        out = out * torch.pow(mean_squared, -0.5)
        out = out * self.weight.float()
        return out.type_as(x)


class Gemma4ScaleFreeRMSNorm(nn.Module):
    """RMSNorm without a learned scale (HF Gemma4RMSNorm(with_scale=False)).
    Used for v_norm; the checkpoint has no v_norm weights."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        out = x.float()
        mean_squared = out.pow(2).mean(-1, keepdim=True) + self.eps
        out = out * torch.pow(mean_squared, -0.5)
        return out.type_as(x)


# -----------------------------------------------------------------------------
# Rotary embeddings
# -----------------------------------------------------------------------------


class Gemma4GlobalRotaryEmbedding(RotaryEmbedding):
    """RoPE for full-attention layers (head_dim 512, theta 1e6, partial factor 0.25).

    Mirrors HF `_compute_proportional_rope_parameters`: only the first
    `int(0.25 * 512 // 2) = 64` frequency channels are real; the remaining 192 of
    the 256 inv_freq entries are ZERO. Zero frequency => cos = 1, sin = 0, so
    RoPE is the identity on those channels ("NoPE" padding) with fully static
    shapes -- no slicing or masking needed at trace time.

    NOTE the denominator: HF computes inv_freq[k] = theta ** (-2k / head_dim)
    with head_dim = 512 (NOT 2 * rope_angles = 128). Keep it that way.

    With NxDI's rotate_half convention (x[..., :256] pairs with x[..., 256:]),
    the rotated channels end up being 0-63 and 256-319, matching HF exactly
    (HF uses the same cat(freqs, freqs) + rotate_half layout).
    """

    def __init__(self, dim, max_position_embeddings=131072, base=1000000.0,
                 partial_rotary_factor=0.25):
        super().__init__(dim, max_position_embeddings=max_position_embeddings, base=base)
        self.rope_angles = int(partial_rotary_factor * dim // 2)

    def get_inv_freqs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        freq_indices = torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float, device=device)
        inv_freq = 1.0 / (self.base ** (freq_indices / self.dim))
        nope = torch.zeros(self.dim // 2 - self.rope_angles, dtype=torch.float32, device=device)
        return torch.cat((inv_freq, nope), dim=0)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


class Gemma4NeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronGemma4Attention


class Gemma4InferenceConfig(InferenceConfig):
    """The ONE definition of the gemma4 inference config.

    `config.py` re-exports this class; there is no second copy. `get_config_cls()`
    returns this class, so every path -- compile, save, reload-from-json -- gets
    the same validation.

    __init__ ordering is deliberate and load-bearing:
      1. load_config()          HF wrapper config attributes land on self
      2. _flatten_text_config() text_config.* copied up to the top level
      3. **kwargs               explicit overrides WIN over the flattened values.
                                This is also the reload path: from_json_string
                                feeds every serialized attribute back through
                                kwargs, and they must not be re-derived.
      4. _apply_defaults()      fill anything still absent
      5. add_derived_config() / validate_config()

    Getting 2 and 3 the other way round (kwargs first, then flatten) silently
    discards explicit overrides -- do not swap them.
    """

    # Copied verbatim from text_config onto the flattened top level.
    # tie_word_embeddings MUST be here or the lm_head tied-weights clone hook
    # never fires (the checkpoint has no lm_head key).
    ATTRIBUTES = [
        "head_dim",
        "global_head_dim",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "num_kv_shared_layers",
        "use_double_wide_mlp",
        "sliding_window",
        "layer_types",
        "rms_norm_eps",
        "hidden_act",
        "vocab_size",
        "vocab_size_per_layer_input",
        "hidden_size_per_layer_input",
        "max_position_embeddings",
        "final_logit_softcapping",
        "tie_word_embeddings",
        "pad_token_id",
        "local_rope_theta",
        "global_rope_theta",
        "partial_rotary_factor",
    ]

    VALID_LAYER_TYPES = {SLIDING, FULL}

    def __init__(self, neuron_config: NeuronConfig, fused_spec_config=None, load_config=None,
                 metadata=None, **kwargs):
        self.attributes = list(self.ATTRIBUTES)

        self.neuron_config = neuron_config
        self.fused_spec_config = fused_spec_config

        if load_config is not None:
            load_config(self)
        else:
            self.load_config()

        self.metadata = metadata

        self._flatten_text_config()

        # kwargs win over flattened values (see class docstring).
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._apply_defaults()
        self.add_derived_config()
        self.validate_config()

    # ------------------------------------------------------------------
    @staticmethod
    def _get(obj, name, default=None):
        """Read `name` from either a dict or an attribute-style config object.

        text_config arrives as an HF config OBJECT on the compile path and as a
        plain DICT on the reload-from-json path. `hasattr` is False for every key
        of a dict, so an attribute-only reader silently no-ops on reload.
        """
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _flatten_text_config(self):
        """Copy what we need from the multimodal config's text_config onto self.

        The wrapper config from the checkpoint dir is `Gemma4ForConditionalGeneration`
        shaped; every model-side consumer in NxDI reads flat attributes.
        """
        text_config = getattr(self, "text_config", None)
        if text_config is None:
            # Reload-from-json path: kwargs carry every already-flattened value.
            return

        _get = self._get
        missing = object()
        for attribute in self.attributes:
            value = _get(text_config, attribute, missing)
            if value is not missing:
                # Note: copied even when the value is None (e.g. a checkpoint with
                # final_logit_softcapping: null) so validate_config sees the truth
                # instead of an absent attribute.
                setattr(self, attribute, value)

        # HF calls it hidden_activation; NxDI modules (NeuronLlamaMLP) read hidden_act.
        hidden_act = _get(text_config, "hidden_activation", None)
        if hidden_act is not None:
            self.hidden_activation = hidden_act
            self.hidden_act = hidden_act

        # rope_parameters is a per-layer-type dict:
        #   sliding_attention: {rope_theta: 1e4, rope_type: default}
        #   full_attention:    {rope_theta: 1e6, rope_type: proportional,
        #                       partial_rotary_factor: 0.25}
        # These are NOT defaulted. A wrong-but-plausible RoPE table on the 7
        # full-attention layers is exactly the silent-degradation failure this
        # port is most exposed to, so absence is a hard error.
        rope_params = _get(text_config, "rope_parameters", None)
        if not rope_params:
            raise ValueError(
                "text_config.rope_parameters is missing. This port refuses to guess "
                "RoPE parameters: the full-attention layers build a NoPE-padded "
                "proportional table whose rope_theta and partial_rotary_factor come "
                "only from here."
            )
        sliding = _get(rope_params, SLIDING, None)
        full = _get(rope_params, FULL, None)
        if not sliding or not full:
            raise ValueError(
                f"text_config.rope_parameters must define both {SLIDING!r} and "
                f"{FULL!r}; got keys {sorted(rope_params) if isinstance(rope_params, dict) else rope_params}"
            )

        def _require(d, key, where):
            value = _get(d, key, None)
            if value is None:
                raise ValueError(
                    f"text_config.rope_parameters.{where}.{key} is missing; this port "
                    "refuses to default RoPE parameters"
                )
            return float(value)

        self.local_rope_theta = _require(sliding, "rope_theta", SLIDING)
        self.global_rope_theta = _require(full, "rope_theta", FULL)
        self.partial_rotary_factor = _require(full, "partial_rotary_factor", FULL)

        full_rope_type = _get(full, "rope_type", None)
        if full_rope_type != "proportional":
            raise ValueError(
                f"rope_parameters.{FULL}.rope_type={full_rope_type!r}; "
                "Gemma4GlobalRotaryEmbedding implements 'proportional' only"
            )
        sliding_rope_type = _get(sliding, "rope_type", None)
        if sliding_rope_type not in ("default", None):
            raise ValueError(
                f"rope_parameters.{SLIDING}.rope_type={sliding_rope_type!r}; "
                "sliding layers use NxDI's stock RotaryEmbedding ('default') only"
            )

    def get_text_config(self):
        """Agent patch 2026-08-23: the base InferenceConfig.get_text_config
        returns self.text_config when present -- but this port keeps the RAW HF
        text_config (object or dict) around after flattening, so NxDI's
        NeuronBaseForCausalLM.__init__ would read vocab_size off a dict and
        crash. All flattened values live on self; return self."""
        return self

    def _apply_defaults(self):
        """Fill only what neither the checkpoint nor kwargs provided.

        Runs AFTER kwargs so an explicit override is never clobbered. RoPE values
        are deliberately NOT defaulted here -- see _flatten_text_config.
        """
        if getattr(self, "hidden_act", None) is None:
            self.hidden_act = getattr(self, "hidden_activation", None)
        if getattr(self, "hidden_activation", None) is None:
            self.hidden_activation = getattr(self, "hidden_act", None)
        if getattr(self, "pad_token_id", None) is None:
            self.pad_token_id = 0

    # ------------------------------------------------------------------
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if hasattr(self, "num_hidden_layers") and hasattr(self, "num_kv_shared_layers"):
            # First layer index whose attention reuses shared KV (15 for E2B: 35 - 20).
            self.first_kv_shared_layer_idx = (
                self.num_hidden_layers - self.num_kv_shared_layers
            )
            if hasattr(self, "layer_types"):
                # Last non-shared layer of each type: the KV donors.
                # E2B: {"sliding_attention": 13, "full_attention": 14}.
                donors = {}
                for i in range(self.first_kv_shared_layer_idx):
                    donors[self.layer_types[i]] = i
                self.kv_donor_layers = donors

    def get_required_attributes(self) -> List[str]:
        return list(self.attributes)

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Gemma4NeuronConfig]:
        return Gemma4NeuronConfig

    # ------------------------------------------------------------------
    def validate_config(self):
        """Gemma4 structural invariants.

        This runs on EVERY construction path including reload-from-json, because
        get_config_cls() returns this class.
        """
        super().validate_config()  # required-attribute presence + base checks

        n = self.num_hidden_layers
        if len(self.layer_types) != n:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, expected "
                f"num_hidden_layers={n}"
            )
        bad = set(self.layer_types) - self.VALID_LAYER_TYPES
        if bad:
            raise ValueError(f"Unknown layer_types values: {sorted(bad)}")
        if not (0 < self.num_kv_shared_layers < n):
            raise ValueError(
                f"num_kv_shared_layers={self.num_kv_shared_layers} must be in (0, {n})"
            )
        shared_types = set(self.layer_types[self.first_kv_shared_layer_idx:])
        missing_donors = shared_types - set(self.kv_donor_layers)
        if missing_donors:
            raise ValueError(
                f"KV-shared layers of type {sorted(missing_donors)} have no donor "
                f"layer before index {self.first_kv_shared_layer_idx}"
            )
        if SLIDING in self.layer_types:
            if not self.sliding_window or self.sliding_window < 2:
                raise ValueError(
                    f"sliding_window={self.sliding_window!r} is unusable: the decode "
                    "ring buffer indexes modulo (sliding_window - 1)"
                )
        if not self.tie_word_embeddings:
            raise ValueError(
                "tie_word_embeddings must be True: the checkpoint has no lm_head "
                "weight, so the lm_head is cloned from embed_tokens"
            )
        if self.hidden_act != "gelu_pytorch_tanh":
            raise ValueError(
                f"hidden_act={self.hidden_act!r}; this port assumes gelu_pytorch_tanh"
            )
        if getattr(self, "enable_moe_block", False):
            raise ValueError("enable_moe_block is set - MoE is not ported (v1 cut list)")
        softcap = self.final_logit_softcapping
        if softcap is not None and float(softcap) <= 0.0:
            raise ValueError(
                f"final_logit_softcapping={softcap!r} must be > 0 or None (no softcap)"
            )
        if self.num_attention_heads % self.neuron_config.tp_degree != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"tp_degree={self.neuron_config.tp_degree}"
            )


def get_updated_configs(config: Gemma4InferenceConfig):
    """Per-layer config clones.

    - full-attention layers get sliding_window=None (their NeuronAttentionBase
      takes the standard causal path; gemma3 pattern)
    - KV-shared layers (>= first_kv_shared_layer_idx) get intermediate_size
      doubled, which is the entire `use_double_wide_mlp` implementation:
      NeuronLlamaMLP just reads config.intermediate_size.
    """
    updated_configs = []
    first_shared = config.first_kv_shared_layer_idx
    for i in range(config.num_hidden_layers):
        updated_config = copy.deepcopy(config)
        if config.layer_types[i] != SLIDING:
            updated_config.sliding_window = None
        if config.use_double_wide_mlp and i >= first_shared > 0:
            updated_config.intermediate_size = config.intermediate_size * 2
        updated_configs.append(updated_config)
    return updated_configs


# -----------------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------------


class NeuronGemma4Attention(NeuronAttentionBase):
    """Gemma4 attention on top of NeuronAttentionBase.

    Base class handles: qkv/o parallel projections (with GQA.REPLICATE_TO_TP_DEGREE
    replicating the single KV head across both cores -- 2 % 1 == 0, zero code, the
    KV cache is simply duplicated per core which is fine in a 16GB/core budget),
    q/k per-head RMSNorm via move_heads_front, RoPE, windowed vs standard forward,
    mask application, and cache read/write plumbing.

    This subclass adds, via a ~10 line prep_qkv_tensors override:
      1. scale-free v_norm on values (HF applies it before caching/attending)
      2. the entire KV-sharing mechanism (see below)
    """

    def __init__(self, config: Gemma4InferenceConfig, layer_idx: int = 0):
        self.layer_type = config.layer_types[layer_idx]
        is_sliding = self.layer_type == SLIDING
        head_dim = config.head_dim if is_sliding else config.global_head_dim  # 256 / 512

        if is_sliding:
            rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.local_rope_theta,
            )
        else:
            rotary_emb = Gemma4GlobalRotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.global_rope_theta,
                partial_rotary_factor=config.partial_rotary_factor,
            )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,   # 8
            num_key_value_heads=config.num_key_value_heads,   # 1
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,  # we use per-head q/k_layernorm below instead
            sliding_window=config.sliding_window if is_sliding else None,
            # Installed NxDI 0.8.16251 has NO softmax_scale kwarg (the pinned
            # snapshot did); every score path hardcodes / math.sqrt(head_dim).
            # HF Gemma4 needs net scale 1.0 (qk-norm replaces 1/sqrt(d)), so Q
            # is pre-multiplied by sqrt(head_dim) in prep_qkv_tensors below to
            # cancel it exactly. Agent patch 2026-08-23.
        )

        # Exact attribute names matter: convert_hf_to_neuron_state_dict renames
        # q_norm/k_norm -> q_layernorm/k_layernorm. Applied pre-RoPE on [B,S,H,D]
        # inside move_heads_front, same order as HF (q_norm -> rope).
        norm_dtype = config.neuron_config.torch_dtype
        self.q_layernorm = NeuronGemma4RMSNorm(head_dim, eps=config.rms_norm_eps,
                                               dtype=norm_dtype)
        self.k_layernorm = NeuronGemma4RMSNorm(head_dim, eps=config.rms_norm_eps,
                                               dtype=norm_dtype)
        # Scale-free: no weights in the checkpoint for this module.
        self.v_norm = Gemma4ScaleFreeRMSNorm(head_dim, eps=config.rms_norm_eps)

        # ---- KV sharing roles (all Python constants; each layer traces to its
        # own static subgraph, the dict below is pure dataflow plumbing) ----
        first_shared = config.first_kv_shared_layer_idx
        self.is_kv_shared = layer_idx >= first_shared > 0
        # Donor = last non-shared layer of this layer's type (13 sliding, 14 full
        # for E2B). Matches HF's store_full_length_kv computation exactly.
        prev_layers = list(config.layer_types[:first_shared])
        self.is_kv_donor = (
            not self.is_kv_shared
            and len(prev_layers) > 0
            and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
        )
        # One dict shared by all 35 attention modules, injected by the text model
        # right after layer construction.
        self.shared_state = None

    def prep_qkv_tensors(self, *args, **kwargs):
        """This override is the entire KV-sharing mechanism.

        Layers 15-34 have no *trained* K/V path -- HF never executes their
        k_proj/v_proj (the checkpoint tensors are dead weights). Their only
        correct K/V source is the last non-shared layer of the same type. This
        is a correctness requirement, not an optimization: skipping it produces
        garbage.

        Donors (13 sliding / 14 full) publish their post-norm, post-RoPE (K, V)
        for the current step. Consumers substitute the donor's (K, V) for their
        own (which was computed from dead weights and is discarded -- ~0.4% of
        layer FLOPs, kept so the stock GQA/preshard machinery stays untouched).

        Everything downstream is the UNMODIFIED base path: each consumer writes
        the donor-derived K/V into its own cache buffer and attends over it.
        Donor and consumer share layer type, position_ids, window trim, and
        ring-buffer math, so consumer caches are bit-identical copies of the
        donor's -- correct by construction, and the base model's 35-entry KV
        collection loop needs no changes. (v2 can dedupe the buffers; that
        requires forking the base KV loop and is deliberately not done here.)

        Works for both prefill (full-sequence K/V shared *before* the sliding
        window trim, which happens later in windowed_attention_forward) and
        decode (single-token K/V shared), with RoPE positions matching because
        donor and consumer run in the same forward at the same positions.
        Layer execution order guarantees donors run before consumers.
        """
        Q, K, V, cos_cache, sin_cache, residual = super().prep_qkv_tensors(*args, **kwargs)

        # Net-1.0 softmax scale (agent patch 2026-08-23): this SDK's
        # NeuronAttentionBase divides Q (or the QK scores) by
        # math.sqrt(self.head_dim) in every eager score path and offers no
        # scale hook. Q feeds ONLY the score matmuls, and scores are linear in
        # Q, so pre-multiplying here cancels that division exactly (bit-exact
        # for head_dim=256; one bf16 rounding for head_dim=512).
        Q = Q * math.sqrt(self.head_dim)

        # HF applies v_norm to values before caching/attending. Norm is over the
        # last (head_dim) axis so BHSD here vs HF's BSHD is equivalent.
        V = self.v_norm(V)

        if self.is_kv_donor:
            self.shared_state[self.layer_type] = (K, V)
        if self.is_kv_shared:
            K, V = self.shared_state[self.layer_type]

        return Q, K, V, cos_cache, sin_cache, residual


# -----------------------------------------------------------------------------
# Decoder layer
# -----------------------------------------------------------------------------


class NeuronGemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4InferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_sliding = config.layer_types[layer_idx] == SLIDING
        dtype = config.neuron_config.torch_dtype
        eps = config.rms_norm_eps

        self.self_attn = NeuronGemma4Attention(config, layer_idx)
        # NeuronLlamaMLP covers use_double_wide_mlp for free: this layer's config
        # clone already carries intermediate_size 6144 or 12288 (see
        # get_updated_configs). Activation is gelu_pytorch_tanh via ACT2FN.
        self.mlp = NeuronLlamaMLP(config)

        self.input_layernorm = NeuronGemma4RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.post_attention_layernorm = NeuronGemma4RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.pre_feedforward_layernorm = NeuronGemma4RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.post_feedforward_layernorm = NeuronGemma4RMSNorm(config.hidden_size, eps=eps, dtype=dtype)

        # ---- PLE (third residual block). Small matrices (1536x256 / 256x1536),
        # replicated on both cores -- not worth sharding.
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.act_fn = ACT2FN[config.hidden_act]
        self.per_layer_input_gate = nn.Linear(
            config.hidden_size, config.hidden_size_per_layer_input, bias=False
        ).to(dtype)
        self.per_layer_projection = nn.Linear(
            config.hidden_size_per_layer_input, config.hidden_size, bias=False
        ).to(dtype)
        self.post_per_layer_input_norm = NeuronGemma4RMSNorm(config.hidden_size, eps=eps, dtype=dtype)

        # Loaded from the checkpoint (key layers.{i}.layer_scalar, shape [1]).
        # nn.Parameter, NOT register_buffer: every checkpoint-loaded scalar in the
        # NxDI tree is a Parameter (see modules/attention/sink.py LearnedSink.sink),
        # and every register_buffer in that tree is persistent=False, i.e. a
        # computed constant that is deliberately NOT fed from a checkpoint. If the
        # NxD sharding/loading path enumerates parameters rather than the full
        # state_dict, a buffer here would silently keep its torch.ones() initializer
        # and every layer would drop its scale correction -- the measured values are
        # 0.0178 (L0) .. 0.87, i.e. up to a 56x per-layer error, which produces
        # fluent garbage that the smoke test cannot detect. requires_grad=False
        # matches LearnedSink.
        self.layer_scalar = nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        per_layer_input: Optional[torch.Tensor] = None,
        adapter_ids=None,
        active_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # [SWA_PATCH_G5] Sliding layers get the windowed mask; full layers the
        # global causal mask. All branches below are decided by Python constants
        # per traced graph (is_sliding, is_for_context_encoding bool,
        # past_key_value None-ness).
        mask = attention_mask
        act_mask = active_mask
        if self.is_sliding and local_mask is not None:
            is_prefix_cte = bool(kwargs.get("is_for_context_encoding", False)) and past_key_value is not None
            if is_prefix_cte:
                # Block-KV prefix CTE (may_have_prefix in get_model_output):
                #  - attention_mask stays the 2-D (B, prefix_bucket) validity mask
                #    that _create_context_attn_mask returns raw when prefix_size != 0.
                #    perform_prefix_prefill_windowed_attn derives P from it.
                #  - the chunk-local windowed-causal mask ([MOD 4]) replaces the
                #    causal-tril active_mask, so the active x active block is
                #    sliding-masked too.
                act_mask = local_mask
            else:
                # Flat layout (unchanged demo path) and block-KV TKG / no-prefix CTE.
                mask = local_mask

        # NOTE: unlike the NxDI gemma3 layer, the sqrt(hidden_size) embedding
        # scale is NOT applied here -- it is applied once in get_model_output,
        # because the PLE projection must consume the scaled embeddings (HF
        # feeds the same scaled inputs_embeds to both the layer stack and
        # per_layer_model_projection).

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, _cos, _sin = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            active_mask=act_mask,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # PLE block: h = residual + norm(proj(act(gate(h)) * per_layer_input))
        residual = hidden_states
        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar

        # NxDI 5-tuple protocol. cos/sin caches are returned as None ON PURPOSE:
        # the base loop threads layer N's cos/sin into layer N+1, which is wrong
        # here because sliding (dim 256, theta 1e4) and full (dim 512, theta 1e6)
        # layers have different tables. Returning None forces each layer to
        # compute its own; the compiler CSEs the duplicate computations.
        return (hidden_states, present_key_value, None, None, None)


# -----------------------------------------------------------------------------
# KV cache manager: mixed cache lengths AND mixed head dims
# -----------------------------------------------------------------------------


class Gemma4KVCacheManager(KVCacheManager):
    """Stock KVCacheManager assumes one head_dim for every layer; gemma4 mixes
    256 (sliding) and 512 (full). It also applies its sliding-window ring-buffer
    write globally; gemma4 needs it only on sliding layers.

    Allocation table (per layer):
        sliding_attention (28 layers): length sliding_window (512), head_dim 256
        full_attention     (7 layers): length max_length,          head_dim 512

    Read/write paths (get_kv_by_layer_id per-layer seq_len override via v_shapes,
    fill_prefix on prefill, scatter on decode, continuous-batching seq_id
    handling) are inherited untouched -- they index per-layer buffers and never
    assume cross-layer shape uniformity beyond allocation.
    """

    def _init_kv_shape(self, config: InferenceConfig,
                       layer_to_cache_size_mapping: Optional[List[int]] = None):
        assert layer_to_cache_size_mapping, \
            "Gemma4KVCacheManager requires layer_to_cache_size_mapping (per-layer cache lengths)"
        assert not self.neuron_config.apply_seq_ids_mask, \
            "apply_seq_ids_mask not supported in gemma4 v1"

        max_batch_size = (
            config.neuron_config.kv_cache_batch_size + config.neuron_config.kv_cache_padding_size
        )
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config)

        self.layer_types = list(config.layer_types)
        self.layer_head_dims = [
            config.head_dim if t == SLIDING else config.global_head_dim for t in self.layer_types
        ]

        self.padded_layer_ids = []
        self.k_shapes = []
        self.v_shapes = []
        for idx, cache_len in enumerate(layer_to_cache_size_mapping):
            k_shape, v_shape = get_kv_shapes(
                cache_len,
                max_batch_size,
                num_kv_heads_per_rank,
                self.layer_head_dims[idx],
                self.k_cache_transposed,
                self.is_kv_cache_tiled,
            )
            self.k_shapes.append(k_shape)
            self.v_shapes.append(v_shape)

    def _get_index_to_update_new_position(self, seq_ids, scatter_index, position_ids,
                                          full_k, transposed: bool, layer_idx: int):
        """Per-layer decode write index. Base class branches on self.sliding_window
        globally; here only sliding layers use the ring buffer.

        Sliding ring is mod (window - 1), matching _create_windowed_attn_mask_tkg,
        which reserves the cache's final slot (always masked) so the attention
        path has a place for the active token. Full layers write at the absolute
        position -- their cache is allocated at max_length so position_ids is
        already in range.
        """
        assert not self.is_medusa, "medusa not supported in gemma4 v1"
        if self.layer_types[layer_idx] == SLIDING:
            position_ids = position_ids % (self.sliding_window - 1)
        index = position_ids
        view_shape = (-1, 1, index.shape[-1], 1) if not transposed else (-1, 1, 1, index.shape[-1])
        return index.view(*view_shape).expand_as(full_k)


class Gemma4BlockKVCacheManager(BlockKVCacheManager):
    """[SWA_PATCH_G4] Block-KV (prefix-caching) layout with per-layer head_dim:
    256 on the 28 sliding layers, 512 on the 7 full layers. Every layer --
    sliding included -- stores the FULL sequence in blocks; the 512 window is
    enforced by the mask (attention_base perform_prefix_prefill_windowed_attn
    and [MOD 4] below), not by the flat 512-slot ring buffer.

    Stock BlockKVCacheManager._init_kv_shape ignores layer_to_cache_size_mapping
    and uses one _get_hidden_dim_per_head(config) == config.head_dim == 256 for
    all layers, so the full layers' (B,1,C,512) K/V would fail the index_put.
    layer_to_cache_size_mapping must still be TRUTHY so KVCacheManager.__init__
    allocates from self.k_shapes / self.v_shapes; its lengths are ignored here.
    """

    def _init_kv_shape(self, config: InferenceConfig, layer_to_cache_size_mapping=None):
        nc = config.neuron_config
        assert nc.is_prefix_caching and not nc.is_chunked_prefill, \
            "Gemma4BlockKVCacheManager is the prefix-caching block layout only"
        assert not nc.apply_seq_ids_mask, "apply_seq_ids_mask not supported in gemma4"
        assert not nc.attn_block_tkg_nki_kernel_enabled, "attn TKG kernel not supported in gemma4"
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config)
        num_blocks = nc.pa_num_blocks + self._NUM_EXTRA_RESERVED_BLOCK  # +1 pad block for slot -1 writes
        block_size = nc.pa_block_size
        # Never tile (stock tiles when max_length/block_size < 128). The 4-D
        # (blocks, H, block, D) layout is read/written via cache.shape, so mixed
        # head_dim needs nothing else.
        self.block_tiling = False
        self.block_tiling_factor = -1
        self.padded_layer_ids = []
        self.layer_types = list(config.layer_types)
        self.k_shapes, self.v_shapes = [], []
        for t in self.layer_types:
            d = config.head_dim if t == SLIDING else config.global_head_dim
            shape = (num_blocks, num_kv_heads_per_rank, block_size, d)
            self.k_shapes.append(shape)
            self.v_shapes.append(shape)
        self.k_shape = self.v_shape = self.k_shapes[0]  # base-parity only; unused


# -----------------------------------------------------------------------------
# LM head with final logit softcapping
# -----------------------------------------------------------------------------


class Gemma4SoftcapLMHead(ColumnParallelLinear):
    """lm_head with final_logit_softcapping: out = cap * tanh(x @ W / cap).

    tanh is elementwise, so it commutes with the vocab-parallel column split --
    correct with gather_output=False + on-device sampling too. This is the whole
    softcapping story; nothing else in NxDI needs touching.

    softcap=None is legal (HF guards `if final_logit_softcapping is not None`);
    this checkpoint sets 30.0, but a future export may not. With None the class
    degrades to a plain ColumnParallelLinear, decided once at construction so
    there is no data-dependent branch in the traced graph."""

    def __init__(self, *args, softcap=30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.softcap = None if softcap is None else float(softcap)

    def forward(self, hidden_states):
        output = super().forward(hidden_states)
        if isinstance(output, tuple):  # (output, bias) form when skip_bias_add
            output = output[0]
        if self.softcap is None:
            return output
        return torch.tanh(output / self.softcap) * self.softcap


# -----------------------------------------------------------------------------
# Text model
# -----------------------------------------------------------------------------


def _assert_v1_feature_set(neuron_config):
    """Fail fast on NxDI features whose code paths were removed from the
    get_model_output copy below. Each is a deliberate v1 cut.

    Absence of a flag is treated as a FAILURE, not as a pass: this guard exists
    because the corresponding branches were deleted from the get_model_output
    copy, so a flag that NxDI renames or removes must stop the compile loudly
    rather than silently report "supported"."""
    unsupported = {
        "layer_boundary_markers": False,
        "flash_decoding_enabled": False,
        "sequence_parallel_enabled": False,
        "is_medusa": False,
        "is_eagle_draft": False,
        "is_eagle3": False,
        "enable_eagle_speculation": False,
        "enable_fused_speculation": False,
        "fused_qkv": False,
        "attn_block_tkg_nki_kernel_enabled": False,
        "k_cache_transposed": False,
        "kv_cache_tiling": False,
        "qkv_kernel_fuse_residual_add": False,
        "is_chunked_prefill": False,
        # Read at get_model_output to decide update_kv_per_layer; that path
        # (cache writes inside the attention module, list-shaped
        # next_decoder_cache) is untested here.
        "attn_block_tkg_nki_kernel_cache_update": False,
        "qkv_kernel_enabled": False,
        "mlp_kernel_enabled": False,
    }
    for name, expected in unsupported.items():
        if not hasattr(neuron_config, name):
            raise AssertionError(
                f"neuron_config has no attribute {name!r}. This guard lists the "
                "NxDI features whose code paths were deleted from this port's "
                "get_model_output; a missing flag means the SDK moved and this "
                "port must be re-checked against it, not that the feature is off."
            )
        actual = getattr(neuron_config, name)
        if actual not in (expected, None):
            raise AssertionError(f"gemma4 v1 does not support neuron_config.{name}={actual}")

    # [SWA_PATCH_G1] block-KV + prefix caching (= emulated chunked prefill).
    # Allowed ONLY as a pair. is_chunked_prefill stays banned (other block path).
    blk = bool(getattr(neuron_config, "is_block_kv_layout", False))
    pfx = bool(getattr(neuron_config, "is_prefix_caching", False))
    assert blk == pfx, (
        "gemma4: is_block_kv_layout and is_prefix_caching must be set together "
        "(block layout is only wired for the prefix-caching read/write path)")
    if blk:
        bs = neuron_config.pa_block_size
        assert neuron_config.max_length % bs == 0, "max_length must be a multiple of pa_block_size"
        for b in (getattr(neuron_config, "prefix_buckets", None) or []):
            assert b % bs == 0, f"prefix bucket {b} not a multiple of pa_block_size={bs}"
        ceb = getattr(neuron_config, "context_encoding_buckets", None)
        assert ceb is None or min(ceb) >= 128, (
            "prefix-caching CTE buckets must be >= 128 active tokens: attention_base "
            "tells TKG from prefix-CTE by q_len < 128")
        assert getattr(neuron_config, "attn_kernel_enabled", None) is False, (
            "block-KV path needs attn_kernel_enabled=False: the prior-mask patches live "
            "only in perform_prefix_prefill's native branch")
    if getattr(neuron_config, "enable_bucketing", False):
        assert blk, (
            "enable_bucketing requires is_block_kv_layout: with the flat cache the 7 "
            "full-attention layers are pinned to max_length while the TKG mask is "
            "n_positions wide (get_model_output assert)")

    assert getattr(neuron_config, "cp_degree", 1) in (1, None), "cp_degree > 1 unsupported in v1"
    assert getattr(neuron_config, "attention_dp_degree", 1) in (1, None), \
        "attention DP unsupported in v1"
    assert getattr(neuron_config, "lora_config", None) is None, "LoRA unsupported in v1"
    assert getattr(neuron_config, "speculation_length", 0) in (0, 1), \
        "speculation unsupported in v1"
    assert getattr(neuron_config, "windowed_context_encoding_size", None) is None, \
        "windowed context encoding unsupported in v1"

    # Same broadcast constraint as enable_bucketing, reached by the other door.
    # [SWA_PATCH_G2] flat layout only: full-attention cache pinned to max_length.
    if not getattr(neuron_config, "is_block_kv_layout", False):
        assert getattr(neuron_config, "token_generation_buckets", None) is None, \
            ("token_generation_buckets forces multiple TKG n_positions values; the "
             "full-attention KV cache is pinned to max_length, so only a single "
             "bucket == max_length works in v1")
        assert getattr(neuron_config, "context_encoding_buckets", None) is None, \
            "context_encoding_buckets unsupported in v1 (single CTE bucket only)"

    # KVCacheManager takes an `elif self.batch_size < self.kv_cache_batch_size:`
    # branch BEFORE the `else` that calls our _get_index_to_update_new_position
    # override. In that branch position_ids are used raw as scatter indices, so
    # the sliding layers' % (sliding_window - 1) ring wrap never happens (any
    # position >= 512 scatters out of range on a 512-slot cache) and the garbage
    # row lands on absolute position seq_len - 1, which for the full layers is a
    # live, attended position. Defaults make these equal (kv_cache_batch_size
    # defaults to batch_size), but setting kv_cache_batch_size explicitly is a
    # normal serving move, so this is asserted rather than assumed.
    #
    # Scoped to the token-generation graph ONLY. That branch lives inside the
    # `else:` of `if is_for_context_encoding:` in update_kv_by_layer_id, so it is
    # never traced into the prefill graph -- and continuous batching legitimately
    # compiles prefill at ctx_batch_size=1 against a kv_cache_batch_size of 16.
    # Asserting unconditionally here would ban continuous batching, which is the
    # entire reason this port exists.
    # [SWA_PATCH_G2] flat KVCacheManager scatter-index concerns only; the block
    # manager writes by slot_mapping and never reaches _get_index_to_update_new_position.
    if not getattr(neuron_config, "is_block_kv_layout", False):
        if neuron_config.is_prefill_stage is not True:
            assert neuron_config.batch_size == neuron_config.kv_cache_batch_size, (
                f"token-generation graph has batch_size={neuron_config.batch_size} != "
                f"kv_cache_batch_size={neuron_config.kv_cache_batch_size}: the "
                "KVCacheManager path that handles this bypasses gemma4's per-layer "
                "ring-buffer write index and corrupts both cache families. Set "
                "tkg_batch_size == kv_cache_batch_size."
            )
        assert not getattr(neuron_config, "kv_cache_padding_size", 0), \
            "kv_cache_padding_size unsupported in v1 (same scatter-index path as above)"

    # attention_base.get_flash_attention_strategy auto-selects UNSHARDED_KERNEL
    # for any prefill q_len >= 4096 unless attn_kernel_enabled is explicitly
    # False. perform_prefill then passes sliding_window into the kernel and
    # asserts get_platform_target() != "trn1", which the 28 sliding layers would
    # trip on trn1-family silicon. Whether inf2 reports as trn1 here is
    # UNVERIFIED, so this refuses the combination rather than guessing.
    max_ctx = getattr(neuron_config, "max_context_length", 0) or 0
    if max_ctx >= 4096 and getattr(neuron_config, "attn_kernel_enabled", None) is not False:
        raise AssertionError(
            f"max_context_length={max_ctx} >= 4096 auto-enables the flash-attention "
            "kernel, which asserts on trn1-family targets when sliding_window is "
            "set. Pass attn_kernel_enabled=False explicitly (or keep prefill under "
            "4096) until the kernel path is validated on the box."
        )


class NeuronGemma4TextModel(NeuronBaseModel):

    def setup_attr_for_model(self, config: Gemma4InferenceConfig):
        _assert_v1_feature_set(config.neuron_config)

        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

        # Mixed attention wiring: with sliding_window set AND has_mixed_attn=True,
        # NeuronBaseModel.forward builds BOTH masks (global attn_mask + windowed
        # local_attn_mask) and threads them to every layer; each layer picks its
        # own (see NeuronGemma4DecoderLayer.forward).
        self.sliding_window = config.sliding_window  # 512
        self.has_mixed_attn = True

        # Per-layer cache lengths; widths are handled in Gemma4KVCacheManager.
        # NOTE full layers use max_length. At token-gen the global mask width is
        # n_positions, so v1 requires the TKG graph to run with a single bucket
        # where n_positions == max_length (default unless bucketing is forced).
        self.layer_to_cache_size_mapping = [
            config.sliding_window if t == SLIDING else config.neuron_config.max_length
            for t in config.layer_types
        ]

    def init_model(self, config: Gemma4InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        dtype = config.neuron_config.torch_dtype

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=dtype,
            shard_across_embedding=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        # PLE token-identity table: [150554, 35*256=8960] ~= 2.7GB bf16,
        # sharded across the embedding dim -> ~1.35GB/core.
        self.embed_tokens_per_layer = ParallelEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            self.padding_idx,
            dtype=dtype,
            shard_across_embedding=True,
        )

        # PLE context projection: [8960, 1536], column-parallel with the output
        # gathered because the following reshape+norm needs the full 35*256 axis.
        self.per_layer_model_projection = ColumnParallelLinear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
            gather_output=True,
            dtype=dtype,
        )
        self.per_layer_projection_norm = NeuronGemma4RMSNorm(
            config.hidden_size_per_layer_input, eps=config.rms_norm_eps, dtype=dtype
        )

        self.lm_head = Gemma4SoftcapLMHead(
            config.hidden_size,
            config.vocab_size,
            softcap=config.final_logit_softcapping,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=dtype,
        )

        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [NeuronGemma4DecoderLayer(conf, idx) for idx, conf in enumerate(updated_configs)]
        )

        # ONE dict shared by all 35 attention modules -- the KV-sharing channel.
        shared_state = {}
        for layer in self.layers:
            layer.self_attn.shared_state = shared_state

        self.norm = NeuronGemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def init_inference_optimization(self, config: Gemma4InferenceConfig):
        # Same as the base version, but plugs in Gemma4KVCacheManager
        # (gpt_oss precedent for swapping the manager here).
        if self.on_device_sampling:
            lm_head_tp_degree = None
            if hasattr(self, "lm_head") and hasattr(self.lm_head, "tensor_parallel_group"):
                lm_head_tp_degree = self.lm_head.tensor_parallel_group.size()
            self.sampler = create_sampler(config.neuron_config, lm_head_tp_degree)

        if config.neuron_config.is_block_kv_layout:
            # [SWA_PATCH_G4] base picks BlockKVCacheManager here; that class has
            # one head_dim for all layers -> Gemma4 subclass.
            self.kv_mgr = Gemma4BlockKVCacheManager(
                config,
                num_kv_head=self.num_key_value_heads,
                sliding_window=self.sliding_window,
                layer_to_cache_size_mapping=self.layer_to_cache_size_mapping,  # truthy; lengths ignored
            )
        else:
            self.kv_mgr = Gemma4KVCacheManager(
                config,
                num_kv_head=self.num_key_value_heads,
                global_rank=self.rank_util,
                sliding_window=self.sliding_window,
                layer_to_cache_size_mapping=self.layer_to_cache_size_mapping,
            )

    def _compute_scaled_embeds_and_per_layer_inputs(self, input_ids, inputs_embeds):
        """PLE pipeline, transcribed from HF Gemma4TextModel line-by-line.

        Ordering that matters. Scale-rounding matters too, and is NOT uniform:
        the two embedding scales go through HF's Gemma4TextScaledWordEmbedding,
        which does `.to(self.weight.dtype)` -- so they are rounded to the
        activation dtype here as well. The projection and combine scales are raw
        python floats in HF and are kept raw here.
          1. inputs_embeds = embed(input_ids) * sqrt(hidden_size)   [scaled]
          2. lookup = embed_per_layer(input_ids) * sqrt(ple_dim)    [scaled]
             -> reshape [B, S, L, 256]
          3. proj = per_layer_model_projection(SCALED inputs_embeds)
                    * hidden_size**-0.5 -> reshape -> RMSNorm
             (HF feeds the same scaled inputs_embeds it sends into the layer
              stack -- see Gemma4TextModel.forward + project_per_layer_inputs.)
          4. per_layer_inputs = (proj + lookup) * 2**-0.5

        Everything here is gather + matmul + reshape: fully traceable.
        """
        dtype = inputs_embeds.dtype
        bsz, seq_len = input_ids.shape[:2]
        num_layers = self.num_hidden_layers
        ple_dim = self.hidden_size_per_layer_input

        embed_scale = torch.tensor(float(self.hidden_size) ** 0.5, dtype=torch.float32).to(dtype)
        inputs_embeds = inputs_embeds * embed_scale

        ple_scale = torch.tensor(float(ple_dim) ** 0.5, dtype=torch.float32).to(dtype)
        per_layer_lookup = self.embed_tokens_per_layer(input_ids) * ple_scale
        per_layer_lookup = per_layer_lookup.reshape(bsz, seq_len, num_layers, ple_dim)

        # RAW python float, NOT rounded to `dtype`. HF's per_layer_model_projection_scale
        # is a plain `config.hidden_size**-0.5` python float and torch does not round a
        # python scalar down to bf16 before the multiply. (Contrast the two embed_scale
        # multiplies above, where HF's Gemma4TextScaledWordEmbedding really does
        # `.to(self.weight.dtype)` -- those are correctly rounded.)
        proj_scale = float(self.hidden_size) ** -0.5
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * proj_scale
        per_layer_projection = per_layer_projection.reshape(bsz, seq_len, num_layers, ple_dim)
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        # RAW python float again: HF's per_layer_input_scale is `2.0**-0.5`.
        combine_scale = 2.0 ** -0.5
        per_layer_inputs = (per_layer_projection + per_layer_lookup) * combine_scale

        return inputs_embeds, per_layer_inputs

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        active_mask=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_cache: bool = False,
        is_for_context_encoding: bool = False,
        vision_embeddings=None,
        vision_mask=None,
        deepstack_vision_embeds=None,
        local_attn_mask: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Copy of NeuronBaseModel.get_model_output (NxD Inference, pinned tree in
        this repo's scratchpad snapshot) with the v1-unsupported branches removed
        (markers / eagle / medusa / SP-slice / vision / lora -- all asserted off
        in setup_attr_for_model) and exactly three gemma4 modifications:
          [MOD 1] sqrt(hidden_size) embedding scale after embed_tokens
          [MOD 2] PLE computation before the layer loop
          [MOD 3] per_layer_input threaded into every layer call (static slice)
        The KV collection loop is byte-equivalent to the base version: every one
        of the 35 layers returns a present_key_value (shared layers return the
        donor-derived K/V they attended with), so kv_mgr.update_cache sees the
        35 entries it expects.
        """
        if not is_for_context_encoding and not self.neuron_config.is_block_kv_layout:
            # [SWA_PATCH_G3] block-KV: TKG prior is gathered from active_block_table
            # and is exactly prefix_bucket == n_positions wide, so no pin needed.
            # Python-level check on ints, evaluated once per traced bucket -- not a
            # tensor branch. model_wrapper sets self.n_positions per bucket before
            # tracing. The 7 full-attention layers' KV cache is max_length long
            # (setup_attr_for_model) and KVCacheManager.get_kv_by_layer_id forces
            # that width on every read, while the TKG mask is n_positions wide.
            assert self.n_positions == self.neuron_config.max_length, (
                f"token generation traced with n_positions={self.n_positions} != "
                f"max_length={self.neuron_config.max_length}. The full-attention "
                "KV cache is pinned to max_length, so the mask and the cache read "
                "would not broadcast. This means bucketing got enabled somewhere; "
                "see _assert_v1_feature_set."
            )

        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # [MOD 1] + [MOD 2]: scale embeddings, compute PLE from the scaled embeds.
        inputs_embeds, per_layer_inputs = self._compute_scaled_embeds_and_per_layer_inputs(
            input_ids, inputs_embeds
        )

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # [MOD 4] [SWA_PATCH_G6] block-KV sliding-window masks. Python-static branches only.
        if self.neuron_config.is_block_kv_layout and self.sliding_window:
            W = self.sliding_window
            dev = position_ids.device
            if is_for_context_encoding:
                # NeuronBaseModel.forward built local_attn_mask n_positions wide
                # (= prefix_bucket + chunk under prefix caching). Rebuild it
                # chunk-local: same rule as _create_windowed_attn_mask_cte.
                i = torch.arange(seq_length, device=dev).unsqueeze(1)
                j = torch.arange(seq_length, device=dev).unsqueeze(0)
                m = (j <= i) & (j >= i - W + 1)
                local_attn_mask = m[None, None, :, :].expand(batch_size, 1, seq_length, seq_length)
            else:
                # TKG: the prior is the block-gathered prefix, n_positions wide,
                # but the base built a W-wide ring-buffer mask. Derive the window
                # from absolute positions on top of the global validity mask
                # (B,1,1,n_positions).
                prior_pos = torch.arange(self.n_positions, device=dev)[None, None, None, :]
                q_pos = position_ids[:, None, :, None]  # (B,1,1,1) for n_active_tokens == 1
                local_attn_mask = attention_mask.to(torch.bool) & (prior_pos > (q_pos - W))

        # SP is asserted off; this is a no-op passthrough kept for base parity.
        hidden_states = self.process_sequence_parallel_hidden_states(
            inputs_embeds, seq_length, kwargs.get("active_block_table", None)
        )

        update_kv_per_layer = update_cache and (
            self.neuron_config.layer_boundary_markers
            or (
                self.neuron_config.attn_block_tkg_nki_kernel_cache_update
                and not is_for_context_encoding
            )
        )

        # decoder layers
        next_decoder_cache = [] if update_kv_per_layer else ()

        cache_size = (
            get_cache_size(self.n_positions, self.num_cores_per_group, is_for_context_encoding)
            if self.neuron_config.flash_decoding_enabled
            else self.n_positions
        )
        if self.sliding_window:
            # Base behavior: seq_len hint becomes the window. Harmless for full
            # layers -- Gemma4KVCacheManager.v_shapes overrides seq_len per layer
            # on every cache read (get_kv_by_layer_id).
            cache_size = self.sliding_window

        get_kv_per_layer = False
        active_block_table = kwargs.get("active_block_table", None)
        empty_active_block_table = True if active_block_table is None \
            else len(active_block_table.shape) == 1
        may_have_prefix = (
            self.is_prefix_caching and is_for_context_encoding and not empty_active_block_table
        )
        if may_have_prefix or not is_for_context_encoding \
                or windowed_context_encoding_window_idx >= 1:
            past_key_values = self.kv_mgr.get_cache(
                seq_ids=seq_ids,
                seq_len=cache_size,
                is_for_context_encoding=is_for_context_encoding,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                # Each layer computes its own cos/sin (mixed rope dims): see
                # NeuronGemma4DecoderLayer.forward.
                cos_cache=None,
                sin_cache=None,
                rotary_position_ids=rotary_position_ids,
                kv_mgr=self.kv_mgr,
                get_kv_per_layer=get_kv_per_layer,
                update_kv_per_layer=update_kv_per_layer,
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                residual=residual,
                local_mask=local_attn_mask,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                padding_mask=padding_mask,
                # [MOD 3] constant idx => static slice, one 256-wide signal per layer
                per_layer_input=per_layer_inputs[:, :, idx, :],
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            kv = layer_outputs[1]
            if update_kv_per_layer:
                next_decoder_cache += kv
            else:
                next_decoder_cache += (kv,)
            residual = layer_outputs[4]

        if update_cache and not update_kv_per_layer:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        self.full_hidden_states = None

        return (hidden_states, next_decoder_cache)


# -----------------------------------------------------------------------------
# CausalLM entry point
# -----------------------------------------------------------------------------


class NeuronGemma4ForCausalLM(NeuronBaseForCausalLM):
    """Text-only Gemma4 for NxD Inference."""

    _model_cls = NeuronGemma4TextModel

    # Verified against this checkpoint's safetensors header: text keys live under
    # model.language_model.* (NOT language_model.model.*).
    _STATE_DICT_MODEL_PREFIX = "model.language_model."

    # NOTE: no enable_context_encoding / enable_token_generation overrides.
    # gemma3 and qwen3_moe override them to stash a `compile_tag` that their
    # get_compiler_args branches on; this port's get_compiler_args emits one flag
    # string for both graphs, so overriding would only risk dropping the base
    # signatures' **model_init_kwargs / enable_wlt_optimization.

    def get_compiler_args(self):
        # gemma3's flag string, verbatim.
        optimization_level = "-O1"
        compiler_args = (
            "--enable-saturate-infinity --enable-mixed-precision-accumulation "
            f"--model-type transformer {optimization_level}"
        )
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        compiler_args += " --auto-cast=none"
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """Remap the (prefix-stripped) HF state dict to Neuron module names.

        The base loader has already stripped `model.language_model.` from every
        text key. Ordering of operations below matters only in that tower
        deletion must see the raw keys.

        NO +1.0 is added to any norm weight: Gemma4RMSNorm multiplies its weight
        directly (unlike gemma2/3's (1+w) convention). Copying gemma3's offsets
        here would be a silent-wrongness bug.
        """
        neuron_config = config.neuron_config
        assert not neuron_config.fused_qkv, "fused_qkv unsupported in gemma4 v1"

        # 1. Drop the multimodal towers. After the prefix strip, every surviving
        #    text key is bare (embed_tokens..., layers...., norm.weight); anything
        #    still under "model." is a tower/projector key.
        for key in list(state_dict.keys()):
            if key.startswith("model.") or "audio" in key or "vision" in key:
                del state_dict[key]

        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree

        for i in range(num_layers):
            # SPMDRank has no checkpoint entry; synthesize (mandatory).
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            # q_norm/k_norm -> q_layernorm/k_layernorm (plain rename, no offset).
            # All 35 layers have both, including the KV-shared layers 15-34 whose
            # k_norm (and k/v_proj) are dead weights in HF: we deliberately load
            # and compute them, then discard the result in prep_qkv_tensors --
            # this keeps the stock GQA preshard machinery untouched (~0.4% FLOPs).
            state_dict[f"layers.{i}.self_attn.q_layernorm.weight"] = (
                state_dict.pop(f"layers.{i}.self_attn.q_norm.weight").detach().clone()
            )
            state_dict[f"layers.{i}.self_attn.k_layernorm.weight"] = (
                state_dict.pop(f"layers.{i}.self_attn.k_norm.weight").detach().clone()
            )

            # v_norm is scale-free; the checkpoint has no v_norm weights, but be
            # defensive in case a future export adds them.
            state_dict.pop(f"layers.{i}.self_attn.v_norm.weight", None)

            # Everything else keeps its name:
            #  - self_attn.{q,k,v,o}_proj.weight: GroupQueryAttention preshard hooks
            #    re-prefix to self_attn.qkv_proj.* / o_proj.*, replicate the single
            #    KV head to tp_degree=2 (REPLICATE_TO_TP_DEGREE, since 2 % 1 == 0),
            #    and split the 8 Q heads 4/4.
            #  - mlp.{gate,up,down}_proj.weight: NeuronLlamaMLP preshard hooks.
            #  - *_layernorm/*_norm weights: loaded as-is (no offsets).
            #  - per_layer_input_gate / per_layer_projection / layer_scalar: as-is.

        # To facilitate rank usage in the base model.
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        # tie_word_embeddings=True and the checkpoint has no lm_head key.
        # Requires tie_word_embeddings to be present on the flattened config,
        # or the application base never calls this hook.
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Gemma4InferenceConfig
