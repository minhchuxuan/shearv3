"""Minimal HuggingFace export utilities for BGE-M3 pruned models"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoModel, AutoConfig, AutoTokenizer


def get_layer_count_from_state_dict(state_dict):
    """Extract number of layers from state dict keys"""
    max_layer = -1
    for key in state_dict.keys():
        if "encoder.layer." in key:
            try:
                layer_num = int(key.split("encoder.layer.")[1].split(".")[0])
                max_layer = max(max_layer, layer_num)
            except (ValueError, IndexError):
                continue
    return max_layer + 1 if max_layer >= 0 else 0


def create_hf_config_from_backbone(backbone):
    """Create HuggingFace config from pruned backbone"""
    config = backbone.config
    
    # Get actual dimensions from pruned model
    actual_layers = len(backbone.encoder.layer)
    actual_vocab_size = backbone.embeddings.word_embeddings.weight.shape[0]
    actual_max_pos = backbone.embeddings.position_embeddings.weight.shape[0]
    actual_type_vocab = backbone.embeddings.token_type_embeddings.weight.shape[0]
    
    # CRITICAL: Keep original hidden_size from embeddings (not attention output size)
    actual_hidden_size = backbone.embeddings.word_embeddings.weight.shape[1]
    
    # Determine uniform intermediate size across layers (use max to allow padding)
    if actual_layers > 0:
        intermediate_sizes = [layer.intermediate.dense.out_features for layer in backbone.encoder.layer]
        actual_intermediate = max(intermediate_sizes) if len(intermediate_sizes) > 0 else config.intermediate_size
        
        # Get actual attention dimensions from the first layer
        first_layer = backbone.encoder.layer[0]
        actual_heads = first_layer.attention.num_attention_heads
        actual_all_head_size = first_layer.attention.query.weight.shape[0]
        
        print(f"Config dimensions: layers={actual_layers}, heads={actual_heads}, all_head_size={actual_all_head_size}, hidden_size={actual_hidden_size}, intermediate(max)={actual_intermediate}")
    else:
        actual_heads = config.num_attention_heads
        actual_intermediate = config.intermediate_size
    
    return {
        "architectures": ["XLMRobertaModel"],
        "attention_probs_dropout_prob": getattr(config, "attention_probs_dropout_prob", 0.1),
        "bos_token_id": 0,
        "classifier_dropout": None,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": getattr(config, "hidden_dropout_prob", 0.1),
        "hidden_size": actual_hidden_size,
        "initializer_range": getattr(config, "initializer_range", 0.02),
        "intermediate_size": actual_intermediate,
        "layer_norm_eps": getattr(config, "layer_norm_eps", 1e-05),
        "max_position_embeddings": actual_max_pos,
        "model_type": "xlm-roberta",
        "num_attention_heads": actual_heads,
        "num_hidden_layers": actual_layers,
        "output_past": True,
        "pad_token_id": 1,
        "position_embedding_type": "absolute",
        "type_vocab_size": actual_type_vocab,
        "use_cache": True,
        "vocab_size": actual_vocab_size
    }


def _pad_rows(t: torch.Tensor, target_rows: int) -> torch.Tensor:
    if t.size(0) == target_rows:
        return t
    out = torch.zeros((target_rows, t.size(1)), dtype=t.dtype)
    out[: t.size(0), : t.size(1)] = t
    return out


def _pad_cols(t: torch.Tensor, target_cols: int) -> torch.Tensor:
    if t.size(1) == target_cols:
        return t
    out = torch.zeros((t.size(0), target_cols), dtype=t.dtype)
    out[:, : t.size(1)] = t
    return out


def _pad_vec(v: torch.Tensor, target_len: int) -> torch.Tensor:
    if v.numel() == target_len:
        return v
    out = torch.zeros((target_len,), dtype=v.dtype)
    out[: v.numel()] = v
    return out


def pad_state_dict_uniform(hf_state_dict: dict) -> dict:
    """Pad tensors to match HuggingFace expected dimensions.
    - Attention query/key/value: pad output dimension to hidden_size 
    - MLP intermediate: pad to max intermediate size
    """
    # Get hidden size from embeddings
    hidden_size = None
    if "embeddings.word_embeddings.weight" in hf_state_dict:
        hidden_size = hf_state_dict["embeddings.word_embeddings.weight"].shape[1]
    elif "embeddings.position_embeddings.weight" in hf_state_dict:
        hidden_size = hf_state_dict["embeddings.position_embeddings.weight"].shape[1]
    
    if hidden_size is None:
        return hf_state_dict
    
    # Determine max intermediate size across layers for MLP padding
    inter_sizes = []
    for k, v in hf_state_dict.items():
        if k.endswith("intermediate.dense.weight"):
            inter_sizes.append(v.shape[0])
    
    inter_max = max(inter_sizes) if inter_sizes else hidden_size * 4
    new_sd = dict(hf_state_dict)
    
    # Collect layer indices from keys present
    layer_indices = set()
    for k in hf_state_dict.keys():
        if k.startswith("encoder.layer."):
            try:
                idx = int(k.split("encoder.layer.")[1].split(".")[0])
                layer_indices.add(idx)
            except Exception:
                pass
    
    for i in layer_indices:
        prefix = f"encoder.layer.{i}"
        
        # CRITICAL FIX: Pad attention query/key/value layers to hidden_size
        # HuggingFace expects these to have output_size = hidden_size
        for name in ["query", "key", "value"]:
            w_key = f"{prefix}.attention.self.{name}.weight"
            b_key = f"{prefix}.attention.self.{name}.bias"
            if w_key in new_sd:
                current_output_size = new_sd[w_key].shape[0]
                if current_output_size < hidden_size:
                    print(f"Padding {w_key} from {current_output_size} to {hidden_size}")
                    new_sd[w_key] = _pad_rows(new_sd[w_key], hidden_size)
            if b_key in new_sd:
                current_size = new_sd[b_key].shape[0]
                if current_size < hidden_size:
                    new_sd[b_key] = _pad_vec(new_sd[b_key], hidden_size)
        
        # MLP intermediate
        int_w = f"{prefix}.intermediate.dense.weight"
        int_b = f"{prefix}.intermediate.dense.bias"
        if int_w in new_sd:
            new_sd[int_w] = _pad_rows(new_sd[int_w], inter_max)
        if int_b in new_sd:
            new_sd[int_b] = _pad_vec(new_sd[int_b], inter_max)
        
        # MLP output (from intermediate to hidden)
        out_w = f"{prefix}.output.dense.weight"
        out_b = f"{prefix}.output.dense.bias"
        if out_w in new_sd:
            new_sd[out_w] = _pad_cols(new_sd[out_w], inter_max)
    
    return new_sd


def convert_backbone_to_hf_state_dict(backbone_state_dict):
    """Convert backbone state dict keys to HuggingFace format"""
    hf_state_dict = {}
    
    for key, value in backbone_state_dict.items():
        # Convert attention layer keys: .attention.query → .attention.self.query
        if ".attention.query." in key:
            new_key = key.replace(".attention.query.", ".attention.self.query.")
        elif ".attention.key." in key:
            new_key = key.replace(".attention.key.", ".attention.self.key.")
        elif ".attention.value." in key:
            new_key = key.replace(".attention.value.", ".attention.self.value.")
        elif ".attention.out_proj." in key:
            new_key = key.replace(".attention.out_proj.", ".attention.output.dense.")
        else:
            new_key = key
            
        hf_state_dict[new_key] = value
    
    # Add missing attention.output.dense layers as identity matrices
    # The backbone concatenates attention heads without learned output projection
    for key in list(hf_state_dict.keys()):
        if ".attention.self.query.weight" in key:
            # Extract layer number and create identity mapping
            layer_prefix = key.replace(".attention.self.query.weight", "")
            
            # Get dimensions after potential padding
            current_query_output_size = hf_state_dict[key].shape[0]
            
            # Get original hidden_size from embeddings
            original_hidden_size = None
            for emb_key in hf_state_dict.keys():
                if "embeddings.word_embeddings.weight" in emb_key:
                    original_hidden_size = hf_state_dict[emb_key].shape[1]
                    break
            
            # Create identity weight and zero bias for missing output projection
            output_weight_key = f"{layer_prefix}.attention.output.dense.weight"
            output_bias_key = f"{layer_prefix}.attention.output.dense.bias"
            
            if output_weight_key not in hf_state_dict:
                # Create identity matrix: query output size → hidden size
                hidden_size = original_hidden_size or current_query_output_size
                
                # If query layers were padded to hidden_size, output projection is just identity
                if current_query_output_size == hidden_size:
                    identity_weight = torch.eye(hidden_size, dtype=hf_state_dict[key].dtype)
                else:
                    # Should not happen with our current padding approach, but handle gracefully
                    identity_weight = torch.zeros((hidden_size, current_query_output_size), dtype=hf_state_dict[key].dtype)
                    min_dim = min(hidden_size, current_query_output_size)
                    identity_weight[:min_dim, :min_dim] = torch.eye(min_dim, dtype=hf_state_dict[key].dtype)
                
                hf_state_dict[output_weight_key] = identity_weight
                    
            if output_bias_key not in hf_state_dict:
                # Bias should match the output size (hidden_size)
                hidden_size = original_hidden_size or current_query_output_size
                hf_state_dict[output_bias_key] = torch.zeros(hidden_size, dtype=hf_state_dict[key].dtype)
    
    return hf_state_dict


def save_backbone_as_hf_model(backbone, save_path, base_model_name="BAAI/bge-m3"):
    """Save pruned backbone as HuggingFace XLM-RoBERTa model"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Update backbone config to match actual pruned dimensions
    actual_layers = len(backbone.encoder.layer)
    if actual_layers > 0:
        first_layer = backbone.encoder.layer[0]
        actual_intermediate = first_layer.intermediate.dense.out_features
        actual_heads = first_layer.attention.num_attention_heads
        
        # CRITICAL: Keep original hidden_size from embeddings (don't change to attention output size)
        embedding_hidden_size = backbone.embeddings.word_embeddings.weight.shape[1]
        
        # Update only the dimensions that actually changed
        backbone.config.num_hidden_layers = actual_layers
        backbone.config.num_attention_heads = actual_heads
        backbone.config.intermediate_size = actual_intermediate
        # DO NOT change backbone.config.hidden_size - it should match embeddings
        
        print(f"Config: {actual_layers} layers, {actual_heads} heads, hidden_size={embedding_hidden_size} (kept from embeddings), intermediate={actual_intermediate}")
        
        # Update layer attention configs for consistency
        for layer in backbone.encoder.layer:
            layer.attention.num_attention_heads = actual_heads
            # Note: all_head_size may be different from hidden_size after pruning
    
    # Save backbone (handles conversion and padding internally)
    backbone.save_pretrained(save_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(save_path)
    
    print(f"✅ Model saved to {save_path}")
    return save_path
