#!/usr/bin/env python3
"""
Evaluate pruned BGE-M3 model on STS dataset
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
import torch.nn.functional as F
from tqdm import tqdm
import os


def load_embedding_head(model_path):
    """Load the embedding head from pruned model"""
    heads_path = os.path.join(model_path, "embedding_heads.pt")
    
    if not os.path.exists(heads_path):
        print("⚠️ No embedding heads found, will use raw mean pooling")
        return None
    
    print(f"Loading embedding heads from {heads_path}")
    heads_data = torch.load(heads_path, map_location='cpu')
    
    # Recreate dense head matching the original BGE-M3 architecture
    class DenseEmbeddingHead(nn.Module):
        def __init__(self, hidden_size: int, output_dim: int = None):
            super().__init__()
            self.output_dim = output_dim or hidden_size
            self.dense = nn.Linear(hidden_size, self.output_dim)
            self.activation = nn.Tanh()
            
        def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
            # Mean pooling with attention mask (same as BGE-M3)
            if attention_mask is not None:
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled = hidden_states[:, 0]  # CLS token fallback
            
            dense_embedding = self.dense(pooled)
            dense_embedding = self.activation(dense_embedding)
            return F.normalize(dense_embedding, p=2, dim=-1)
    
    # Create and load the dense head
    hidden_size = heads_data['hidden_size']
    dense_head = DenseEmbeddingHead(hidden_size)
    dense_head.load_state_dict(heads_data['dense_head'])
    print("✅ Loaded pre-trained embedding head with learned transformations")
    return dense_head


def load_pruned_model(model_path):
    """Load the pruned model, tokenizer, and embedding head"""
    print(f"Loading pruned model from {model_path}")
    
    # Load backbone
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load embedding head
    embedding_head = load_embedding_head(model_path)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    if embedding_head is not None:
        total_params += sum(p.numel() for p in embedding_head.parameters())
    
    print(f"Model loaded: {total_params:,} parameters")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Attention heads: {model.config.num_attention_heads}")
    print(f"Intermediate size: {model.config.intermediate_size}")
    print(f"Using embedding head: {embedding_head is not None}")
    
    return model, tokenizer, embedding_head


def encode_texts(texts, model, tokenizer, embedding_head=None, batch_size=32, max_length=512):
    """Encode texts using the model with proper embedding head"""
    model.eval()
    if embedding_head is not None:
        embedding_head.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Get embeddings from backbone
            outputs = model(**inputs)
            
            # Use embedding head if available, otherwise raw mean pooling
            if embedding_head is not None:
                # Use the loaded embedding head (includes pooling, transform, activation, normalization)
                batch_embeddings = embedding_head(outputs.last_hidden_state, inputs['attention_mask'])
            else:
                # Fallback: Raw mean pooling (SUBOPTIMAL - will hurt performance)
                print("⚠️ Using raw mean pooling - performance will be degraded!")
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                
                embeddings_masked = token_embeddings * input_mask_expanded
                sum_embeddings = torch.sum(embeddings_masked, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # Normalize
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            embeddings.append(batch_embeddings.cpu())
    
    return torch.cat(embeddings, dim=0)


def evaluate_sts(model, tokenizer, embedding_head=None, dataset_name="mteb/stsbenchmark-sts"):
    """Evaluate on STS dataset"""
    print(f"Loading STS dataset: {dataset_name}")
    
    # Load dataset - try MTEB format first, fallback to sentence-transformers
    try:
        dataset = load_dataset(dataset_name)
        print("✅ Using MTEB format (consistent with pruning/finetuning)")
    except:
        print("⚠️ MTEB dataset not available, falling back to sentence-transformers format")
        dataset = load_dataset("sentence-transformers/stsb")
    
    test_data = dataset['test']
    
    # Extract sentences and scores
    sentences1 = test_data['sentence1']
    sentences2 = test_data['sentence2'] 
    scores = np.array(test_data['score'])
    
    print(f"Evaluating on {len(sentences1)} sentence pairs")
    
    # Encode sentences
    print("Encoding sentence 1...")
    embeddings1 = encode_texts(sentences1, model, tokenizer, embedding_head)
    
    print("Encoding sentence 2...")
    embeddings2 = encode_texts(sentences2, model, tokenizer, embedding_head)
    
    # Compute similarities
    similarities = F.cosine_similarity(embeddings1, embeddings2, dim=1)
    
    # Scale to [0, 5] range to match STS scores
    predicted_scores = (similarities + 1) * 2.5
    predicted_scores = predicted_scores.numpy()
    
    # Compute correlations
    spearman_corr, _ = spearmanr(predicted_scores, scores)
    pearson_corr, _ = pearsonr(predicted_scores, scores)
    
    print(f"\n📊 STS Evaluation Results:")
    print(f"Spearman correlation: {spearman_corr:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f}")
    
    return {
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'predicted_scores': predicted_scores,
        'true_scores': scores
    }


def main():
    model_path = "experiments/production_hf"
    
    # Load model with embedding head
    model, tokenizer, embedding_head = load_pruned_model(model_path)
    
    # Evaluate on STS
    results = evaluate_sts(model, tokenizer, embedding_head)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Final Spearman correlation: {results['spearman']:.4f}")
    
    if embedding_head is None:
        print("⚠️ WARNING: No embedding head was used - results may be significantly lower than expected!")
        print("   For optimal performance, ensure embedding_heads.pt exists in the model directory.")


if __name__ == "__main__":
    main()
