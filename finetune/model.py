import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Any, Optional

class FinetuneBGEM3(nn.Module):
    """Simplified BGE-M3 model for finetuning pruned models on STS tasks"""
    
    def __init__(self, model_path: str, temperature: float = 0.02):
        super().__init__()
        
        # Load pruned model from HuggingFace format
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Model parameters
        self.temperature = temperature
        
        # Freeze backbone initially (can be unfrozen later)
        self._freeze_backbone()
        
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone for full model finetuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def unfreeze_last_layers(self, num_layers: int = 2):
        """Unfreeze last N layers of the backbone"""
        total_layers = len(self.backbone.encoder.layer)
        for i in range(max(0, total_layers - num_layers), total_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        
        # Get embeddings from backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token for sentence embeddings (following BGE-M3 approach)
        embeddings = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return {
            'embeddings': embeddings,
            'last_hidden_state': outputs.last_hidden_state
        }
    
    def compute_sts_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute STS loss for sentence pairs"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        similarity_scores = batch['similarity_scores']
        
        # Forward pass
        outputs = self.forward(input_ids, attention_mask)
        embeddings = outputs['embeddings']  # [batch_size * 2, hidden_size]
        
        # Extract sentence pairs (interleaved format)
        batch_size = embeddings.size(0) // 2
        sent1_emb = embeddings[0::2]  # Even indices: first sentences
        sent2_emb = embeddings[1::2]  # Odd indices: second sentences
        
        # Compute cosine similarity
        predicted_sim = F.cosine_similarity(sent1_emb, sent2_emb, dim=-1)
        
        # Scale to [0, 5] range to match STS scores
        predicted_sim = (predicted_sim + 1) * 2.5
        
        # MSE loss
        loss = F.mse_loss(predicted_sim, similarity_scores)
        
        return {
            'loss': loss,
            'predicted_scores': predicted_sim,
            'ground_truth_scores': similarity_scores
        }
    
    def compute_spearman_correlation(self, predicted_scores: torch.Tensor, 
                                   ground_truth_scores: torch.Tensor) -> float:
        """Compute Spearman correlation for evaluation"""
        try:
            from scipy.stats import spearmanr
            pred_np = predicted_scores.detach().cpu().numpy()
            gt_np = ground_truth_scores.detach().cpu().numpy()
            correlation, _ = spearmanr(pred_np, gt_np)
            return float(correlation) if not torch.isnan(torch.tensor(correlation)) else 0.0
        except ImportError:
            # Fallback to Pearson correlation
            pred_centered = predicted_scores - predicted_scores.mean()
            gt_centered = ground_truth_scores - ground_truth_scores.mean()
            correlation = (pred_centered * gt_centered).sum() / (
                torch.sqrt((pred_centered ** 2).sum() * (gt_centered ** 2).sum()) + 1e-8
            )
            return float(correlation)
    
    def save_model(self, save_path: str):
        """Save finetuned model in HuggingFace format"""
        self.backbone.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save additional finetuning info
        import json
        import os
        finetune_info = {
            'base_model': 'pruned_bge_m3',
            'finetuning_task': 'sts',
            'temperature': self.temperature
        }
        with open(os.path.join(save_path, 'finetune_info.json'), 'w') as f:
            json.dump(finetune_info, f, indent=2)
