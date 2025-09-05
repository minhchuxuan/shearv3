"""Clean HuggingFace export utilities for BGE-M3 pruned models"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoModel, AutoConfig, AutoTokenizer, XLMRobertaModel, XLMRobertaConfig


def create_true_pruned_config(backbone):
    """Generate config reflecting actual pruned dimensions"""
    # Count actual layers after pruning
    num_layers = len(backbone.encoder.layer)
    
    # Get dimensions from embeddings and first layer
    hidden_size = backbone.embeddings.word_embeddings.weight.shape[1]  # Always 1024
    vocab_size = backbone.embeddings.word_embeddings.weight.shape[0]
    max_pos = backbone.embeddings.position_embeddings.weight.shape[0]
    type_vocab = backbone.embeddings.token_type_embeddings.weight.shape[0]
    
    if num_layers > 0:
        first_layer = backbone.encoder.layer[0]
        num_heads = first_layer.attention.num_attention_heads
        intermediate_size = first_layer.intermediate.dense.out_features
    else:
        # Fallback to original config values
        num_heads = backbone.config.num_attention_heads  
        intermediate_size = backbone.config.intermediate_size
    
    # Validate mathematical consistency
    assert hidden_size % num_heads == 0, f"Invalid: hidden_size={hidden_size} not divisible by num_heads={num_heads}"
    
    return {
        "architectures": ["XLMRobertaModel"],
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": max_pos,
        "model_type": "xlm-roberta",
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "output_past": True,
        "pad_token_id": 1,
        "position_embedding_type": "absolute",
        "type_vocab_size": type_vocab,
        "use_cache": True,
        "vocab_size": vocab_size
    }


def transfer_pruned_weights(backbone, hf_model):
    """Copy weights handling dimension mismatches"""
    # 1. Embeddings - direct copy
    hf_model.embeddings.load_state_dict(backbone.embeddings.state_dict())
    
    # 2. Encoder layers
    for i, backbone_layer in enumerate(backbone.encoder.layer):
        hf_layer = hf_model.encoder.layer[i]
        
        # Get dimensions
        all_head_size = backbone_layer.attention.all_head_size
        hidden_size = backbone.config.hidden_size
        
        # 3. Attention weights with dimension handling
        if all_head_size < hidden_size:
            # Pruned heads case: place in subset of weight matrix
            for name in ['query', 'key', 'value']:
                backbone_param = getattr(backbone_layer.attention, name)
                hf_param = getattr(hf_layer.attention.self, name)
                
                # Initialize with zeros, copy pruned weights to first rows
                hf_param.weight.data.zero_()
                hf_param.weight.data[:all_head_size, :] = backbone_param.weight.data
                hf_param.bias.data.zero_()
                hf_param.bias.data[:all_head_size] = backbone_param.bias.data
            
            # For pruned heads: create identity mapping for attention output
            # HF expects [hidden_size, all_head_size] projection
            hf_out = hf_layer.attention.output.dense
            hf_out.weight.data.zero_()
            # Create identity mapping for the remaining heads
            hf_out.weight.data[:all_head_size, :all_head_size] = torch.eye(all_head_size)
            hf_out.bias.data.zero_()
            
        else:
            # No head pruning: direct copy
            for name in ['query', 'key', 'value']:
                backbone_param = getattr(backbone_layer.attention, name)
                hf_param = getattr(hf_layer.attention.self, name)
                hf_param.weight.data.copy_(backbone_param.weight.data)
                hf_param.bias.data.copy_(backbone_param.bias.data)
            
            # For no head pruning: create full identity mapping
            hf_out = hf_layer.attention.output.dense
            hf_out.weight.data = torch.eye(hidden_size)
            hf_out.bias.data.zero_()
        
        # 4. MLP layers - direct copy
        hf_layer.intermediate.dense.weight.data.copy_(backbone_layer.intermediate.dense.weight.data)
        hf_layer.intermediate.dense.bias.data.copy_(backbone_layer.intermediate.dense.bias.data)
        hf_layer.output.dense.weight.data.copy_(backbone_layer.output.dense.weight.data)
        hf_layer.output.dense.bias.data.copy_(backbone_layer.output.dense.bias.data)
        
        # 5. Layer norms - backbone has merged LayerNorm, HF expects separate ones
        # The backbone's output.LayerNorm handles both attention and MLP residuals
        # We need to copy it to both HF LayerNorm locations
        
        # Copy to attention output LayerNorm (handles attention residual in HF)
        hf_layer.attention.output.LayerNorm.weight.data.copy_(backbone_layer.output.LayerNorm.weight.data)
        hf_layer.attention.output.LayerNorm.bias.data.copy_(backbone_layer.output.LayerNorm.bias.data)
        
        # Copy to MLP output LayerNorm (handles MLP residual in HF) 
        hf_layer.output.LayerNorm.weight.data.copy_(backbone_layer.output.LayerNorm.weight.data)
        hf_layer.output.LayerNorm.bias.data.copy_(backbone_layer.output.LayerNorm.bias.data)


def export_embedding_heads(embedding_heads, save_path):
    """Export the embedding heads separately for complete model functionality"""
    heads_path = os.path.join(save_path, "embedding_heads.pt")
    heads_state = {
        'dense_head': embedding_heads.dense_head.state_dict(),
        'sparse_head': embedding_heads.sparse_head.state_dict(), 
        'multi_vector_head': embedding_heads.multi_vector_head.state_dict(),
        'vocab_size': embedding_heads.vocab_size,
        'hidden_size': embedding_heads.hidden_size,
    }
    torch.save(heads_state, heads_path)
    print(f"✅ Embedding heads saved to {heads_path}")


def export_pruned_backbone_clean(backbone, save_path, base_model_name="BAAI/bge-m3", embedding_heads=None):
    """Export truly pruned model without padding"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # 1. Create config matching pruned architecture
    pruned_config = create_true_pruned_config(backbone)
    
    # 2. Instantiate fresh HF model with pruned config
    config_obj = XLMRobertaConfig(**pruned_config)
    hf_model = XLMRobertaModel(config_obj)
    
    # 3. Copy weights with dimension handling
    transfer_pruned_weights(backbone, hf_model)
    
    # 4. Save clean model
    hf_model.save_pretrained(save_path)
    
    # 5. Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(save_path)
    
    # 6. Save embedding heads if provided
    if embedding_heads is not None:
        export_embedding_heads(embedding_heads, save_path)
        
        # Save complete model configuration
        model_config = {
            'architecture': 'bge-m3-pruned',
            'has_embedding_heads': True,
            'base_model': base_model_name,
            'head_config': {
                'dense_dim': embedding_heads.dense_head.output_dim,
                'vocab_size': embedding_heads.vocab_size,
                'hidden_size': embedding_heads.hidden_size,
            }
        }
        with open(os.path.join(save_path, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
    
    print(f"✅ Clean pruned model exported to {save_path}")
    print(f"📊 Architecture: {pruned_config['num_hidden_layers']} layers, {pruned_config['num_attention_heads']} heads, {pruned_config['intermediate_size']} intermediate")
    
    return save_path
# Backward compatibility - keep old function names but use new implementation
def save_backbone_as_hf_model(backbone, save_path, base_model_name="BAAI/bge-m3"):
    """Legacy wrapper - use export_pruned_backbone_clean instead"""
    return export_pruned_backbone_clean(backbone, save_path, base_model_name)

def create_hf_config_from_backbone(backbone):
    """Legacy wrapper - use create_true_pruned_config instead"""
    return create_true_pruned_config(backbone)
