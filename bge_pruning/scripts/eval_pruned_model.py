#!/usr/bin/env python3
"""
Evaluate pruned BGE-M3 model on STS dataset
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
import torch.nn.functional as F
from tqdm import tqdm


def load_pruned_model(model_path):
    """Load the pruned model and tokenizer"""
    print(f"Loading pruned model from {model_path}")
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Attention heads: {model.config.num_attention_heads}")
    print(f"Intermediate size: {model.config.intermediate_size}")
    
    return model, tokenizer


def encode_texts(texts, model, tokenizer, batch_size=32, max_length=512):
    """Encode texts using the model"""
    model.eval()
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
            
            # Get embeddings
            outputs = model(**inputs)
            
            # Mean pooling (BGE-M3 style)
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


def evaluate_sts(model, tokenizer, dataset_name="sentence-transformers/stsb"):
    """Evaluate on STS dataset"""
    print(f"Loading STS dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    test_data = dataset['test']
    
    # Extract sentences and scores
    sentences1 = test_data['sentence1']
    sentences2 = test_data['sentence2'] 
    scores = np.array(test_data['score'])
    
    print(f"Evaluating on {len(sentences1)} sentence pairs")
    
    # Encode sentences
    print("Encoding sentence 1...")
    embeddings1 = encode_texts(sentences1, model, tokenizer)
    
    print("Encoding sentence 2...")
    embeddings2 = encode_texts(sentences2, model, tokenizer)
    
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
    
    # Load model
    model, tokenizer = load_pruned_model(model_path)
    
    # Evaluate on STS
    results = evaluate_sts(model, tokenizer)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Final Spearman correlation: {results['spearman']:.4f}")


if __name__ == "__main__":
    main()
