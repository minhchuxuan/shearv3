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
    
    # Use first layer to get actual head count and intermediate size
    if actual_layers > 0:
        first_layer = backbone.encoder.layer[0]
        actual_intermediate = first_layer.intermediate.dense.out_features
        
        # Infer actual head count from weight dimensions (more reliable than stored config)
        query_weight_shape = first_layer.attention.query.weight.shape
        head_dim = config.hidden_size // config.num_attention_heads  # Original head dimension
        actual_all_head_size = query_weight_shape[0]
        actual_heads = actual_all_head_size // head_dim
        
        print(f"Config dimensions: layers={actual_layers}, heads={actual_heads}, intermediate={actual_intermediate}")
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
        "hidden_size": config.hidden_size,
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
            hidden_size = hf_state_dict[key].shape[0]  # Should equal all_head_size
            
            # Create identity weight and zero bias for missing output projection
            output_weight_key = f"{layer_prefix}.attention.output.dense.weight"
            output_bias_key = f"{layer_prefix}.attention.output.dense.bias"
            
            if output_weight_key not in hf_state_dict:
                hf_state_dict[output_weight_key] = torch.eye(hidden_size, dtype=hf_state_dict[key].dtype)
            if output_bias_key not in hf_state_dict:
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
        query_weight_shape = first_layer.attention.query.weight.shape
        head_dim = backbone.config.hidden_size // backbone.config.num_attention_heads
        actual_heads = query_weight_shape[0] // head_dim
        
        backbone.config.num_hidden_layers = actual_layers
        backbone.config.num_attention_heads = actual_heads
        backbone.config.intermediate_size = actual_intermediate
        
        print(f"Config: {actual_layers} layers, {actual_heads} heads, {actual_intermediate} intermediate")
        
        # Update layer attention configs
        for layer in backbone.encoder.layer:
            layer.attention.num_attention_heads = actual_heads
            layer.attention.all_head_size = actual_heads * head_dim
    
    # Save backbone (handles conversion internally)
    backbone.save_pretrained(save_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(save_path)
    
    print(f"✅ Model saved to {save_path}")
    return save_path
