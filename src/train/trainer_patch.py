"""
Flash Attention Patch for Qwen-VL models
Based on original: qwenvl/train/trainer.py
"""
from typing import Optional

import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple:
    """Custom flash attention forward for Qwen-VL."""
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
        )

    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError("Tensor query has shape with a zero dimension.")

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    cu_seqlens = attention_mask

    # Convert list to tensor if needed (e.g., attention_mask=[1296] from collator)
    if isinstance(cu_seqlens, (list, tuple)):
        cu_seqlens = torch.tensor([0] + list(cu_seqlens), dtype=torch.int32, device=query.device)

    # Debug: check cu_seqlens validity
    if cu_seqlens is None or cu_seqlens.numel() == 0 or cu_seqlens.dim() != 1:
        raise ValueError(f"Invalid cu_seqlens: type={type(cu_seqlens)}, shape={getattr(cu_seqlens, 'shape', None)}")

    with torch.no_grad():
        max_seqlen = max([
            cu_seqlens[idx + 1] - cu_seqlens[idx]
            for idx in range(cu_seqlens.size(0) - 1)
        ]).item()

    attn_output = flash_attn_varlen_func(
        query, key, value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    attn_output = attn_output.unsqueeze(0)
    return attn_output, None


def qwen2vl_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple:
    """Qwen2VL attention forward with flash attention."""
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, sliding_window=self.sliding_window,
        position_ids=position_ids, **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def qwen3vl_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple:
    """Qwen3VL attention forward with flash attention."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def return_mask(config, inputs_embeds, attention_mask=None, cache_position=None, *, past_key_values=None, position_ids=None, **kwargs):
    return attention_mask


# Store original forward functions for restoration
_original_forwards = {}

def replace_qwen2_vl_attention_class():
    """Replace attention classes with flash attention forward."""
    import transformers

    # Qwen2VL
    _original_forwards['qwen2_vl_attn'] = transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward
    _original_forwards['qwen2_vl_create_causal_mask'] = transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask
    _original_forwards['qwen2_vl_create_sliding'] = transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = qwen2vl_forward
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = return_mask
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = return_mask

    # Qwen2.5VL
    _original_forwards['qwen2_5_vl_attn'] = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward
    _original_forwards['qwen2_5_vl_create_causal_mask'] = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask
    _original_forwards['qwen2_5_vl_create_sliding'] = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2vl_forward
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = return_mask
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = return_mask

    # Qwen3VL - patch both forward and create_causal_mask (same as original)
    _original_forwards['qwen3_vl_attn'] = transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward
    _original_forwards['qwen3_vl_create_causal_mask'] = transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = qwen3vl_forward
    transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = return_mask

    # Qwen3VL MoE - patch both forward and create_causal_mask (same as original)
    _original_forwards['qwen3_vl_moe_attn'] = transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward
    _original_forwards['qwen3_vl_moe_create_causal_mask'] = transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = qwen3vl_forward
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = return_mask


def restore_original_attention_class():
    """Restore original attention classes (for validation)."""
    import transformers

    if 'qwen2_vl_attn' in _original_forwards:
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = _original_forwards['qwen2_vl_attn']
        transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = _original_forwards['qwen2_vl_create_causal_mask']
        transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = _original_forwards['qwen2_vl_create_sliding']

    if 'qwen2_5_vl_attn' in _original_forwards:
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = _original_forwards['qwen2_5_vl_attn']
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = _original_forwards['qwen2_5_vl_create_causal_mask']
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = _original_forwards['qwen2_5_vl_create_sliding']

    if 'qwen3_vl_attn' in _original_forwards:
        transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = _original_forwards['qwen3_vl_attn']
        transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = _original_forwards['qwen3_vl_create_causal_mask']

    if 'qwen3_vl_moe_attn' in _original_forwards:
        transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = _original_forwards['qwen3_vl_moe_attn']
        transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = _original_forwards['qwen3_vl_moe_create_causal_mask']
