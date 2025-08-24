import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models.base import ComposerModel
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from typing import Dict, Optional, Any, Tuple
from transformers import AutoModel, AutoConfig

from .l0_module_embedding import L0ModuleEmbedding
from .embedding_heads import BGEEmbeddingHeads

class ComposerBGEM3(ComposerModel):
    """BGE-M3 model with L0 pruning and Composer interface"""
    
    def __init__(self, cfg):
        super().__init__()
        
        # Load pretrained BGE-M3 model and config
        model_name = getattr(cfg, 'base_model', 'BAAI/bge-m3')
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        
        # Override config with custom settings if provided
        if hasattr(cfg, 'd_model'):
            self.config.hidden_size = cfg.d_model
        if hasattr(cfg, 'n_layers'):
            self.config.num_hidden_layers = cfg.n_layers
        if hasattr(cfg, 'n_heads'):
            self.config.num_attention_heads = cfg.n_heads
            
        # Initialize embedding heads
        self.embedding_heads = BGEEmbeddingHeads(self.config)
        
        # Fix device string conversion
        device_name = get_device(None).name
        if device_name == "gpu":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize L0 module for pruning with model info
        self.l0_module = L0ModuleEmbedding(cfg, device_name, self.backbone)

        # Loss configurations
        self.use_sts_loss = getattr(cfg, 'use_sts_loss', True)
        self.use_contrastive_loss = getattr(cfg, 'use_contrastive_loss', True)
        self.temperature = getattr(cfg, 'temperature', 0.02)
        
        # Metrics storage
        self.train_metrics = {}
        self.eval_metrics = {}
        self.ref_model = None
        
    def forward(self, batch):
        """Forward pass through the model"""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        
        # Get actual batch size from tensor
        actual_batch_size = input_ids.size(0)
        
        # Get L0 masks and set them for hooks
        l0_output = self.l0_module()
        
        # Forward through pretrained backbone (hooks will apply masks)
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get embeddings from heads
        embedding_outputs = self.embedding_heads(
            hidden_states=backbone_outputs.last_hidden_state,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        return {
            'embeddings': embedding_outputs,
            'backbone_outputs': backbone_outputs,
            'l0_output': l0_output,
            'batch_size': actual_batch_size,
        }
    
    def eval_forward(self, batch, outputs=None):
        """Evaluation forward pass"""
        return outputs if outputs is not None else self.forward(batch)
    
    def loss(self, outputs, batch):
        """Compute loss for training"""
        embeddings = outputs['embeddings']
        l0_output = outputs['l0_output']
        batch_size = outputs['batch_size']
        
        # Safety check for valid batch size
        if batch_size <= 0:
            return torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        
        total_loss = 0.0
        loss_dict = {}
        
        # STS regression loss
        if self.use_sts_loss and 'similarity_scores' in batch:
            sts_loss = self.compute_sts_loss(embeddings, batch, batch_size)
            total_loss += sts_loss
            loss_dict['sts_loss'] = sts_loss
        
        # Contrastive learning loss
        if self.use_contrastive_loss and 'positive_ids' in batch:
            contrastive_loss = self.compute_contrastive_loss(embeddings, batch, batch_size)
            total_loss += contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss
        
        # L0 sparsity loss and constraints
        if hasattr(self.l0_module, 'get_sparsity_loss'):
            sparsity_loss, expected_sparsity, expected_score = self.l0_module.get_sparsity_loss()
            
            # Lagrangian constraints for target architecture
            constraint_loss = self.compute_constraint_loss(expected_sparsity)
            
            total_loss += sparsity_loss + constraint_loss
            loss_dict.update({
                'sparsity_loss': sparsity_loss,
                'constraint_loss': constraint_loss,
            })
        
        loss_dict['total_loss'] = total_loss
        return total_loss
    
    def compute_sts_loss(self, embeddings: Dict[str, torch.Tensor], batch: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Compute STS regression loss"""
        dense_emb = embeddings['dense_embedding']  # Shape: [actual_batch_size, embedding_dim]
        
        # Handle case where embedding might be flattened
        if dense_emb.dim() == 1:
            # Tensor is flattened, try to infer batch structure
            embedding_dim = 1024  # BGE-M3 default
            if dense_emb.size(0) % embedding_dim == 0:
                actual_batch = dense_emb.size(0) // embedding_dim
                dense_emb = dense_emb.view(actual_batch, embedding_dim)
                batch_size = actual_batch
            else:
                # Fallback: create dummy loss
                return torch.tensor(0.0, device=dense_emb.device, requires_grad=True)
        
        # Check if we have paired data or need to handle differently
        if batch_size % 2 == 0 and 'similarity_scores' in batch:
            # Paired sentence format
            pair_batch_size = batch_size // 2
            similarity_scores = batch['similarity_scores']
            
            if similarity_scores.size(0) == pair_batch_size and batch_size >= 2:
                # Reshape embeddings for sentence pairs
                dense_emb_reshaped = dense_emb.view(pair_batch_size, 2, -1)
                sent1_emb = dense_emb_reshaped[:, 0, :]
                sent2_emb = dense_emb_reshaped[:, 1, :]
            elif batch_size >= 2:
                # Interleaved format - take alternate indices
                sent1_emb = dense_emb[0::2]  # Even indices
                sent2_emb = dense_emb[1::2]  # Odd indices
                similarity_scores = similarity_scores[:sent1_emb.size(0)]
            else:
                # Single sample fallback
                sent1_emb = dense_emb[:1]
                sent2_emb = dense_emb[:1]
                similarity_scores = torch.ones(1, device=dense_emb.device)
        else:
            # Fallback: use first half vs second half
            if batch_size >= 2:
                mid = batch_size // 2
                sent1_emb = dense_emb[:mid]
                sent2_emb = dense_emb[mid:]
                similarity_scores = batch.get('similarity_scores', torch.ones(mid, device=dense_emb.device))
            else:
                # Single sample case
                sent1_emb = dense_emb
                sent2_emb = dense_emb
                similarity_scores = torch.ones(batch_size, device=dense_emb.device)
        
        # Compute cosine similarity
        predicted_sim = F.cosine_similarity(sent1_emb, sent2_emb, dim=-1)
        
        # Scale to [0, 5] range to match STS scores
        predicted_sim = (predicted_sim + 1) * 2.5
        
        # MSE loss
        sts_loss = F.mse_loss(predicted_sim, similarity_scores.float())
        return sts_loss
    
    def compute_contrastive_loss(self, embeddings: Dict[str, torch.Tensor], batch: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Compute InfoNCE contrastive loss"""
        dense_emb = embeddings['dense_embedding']  # [actual_batch_size, embedding_dim]
        
        # Compute similarity matrix with proper tensor operations
        similarity_matrix = torch.matmul(dense_emb, dense_emb.t()) / self.temperature
        
        # Create labels - for contrastive learning, each sample is its own positive
        labels = torch.arange(dense_emb.size(0), device=dense_emb.device)
        
        # InfoNCE loss
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        return contrastive_loss
    
    def compute_constraint_loss(self, expected_sparsity: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Lagrangian constraint loss for target architecture"""
        constraint_loss = 0.0
        
        for mask_name, sparsity in expected_sparsity.items():
            if mask_name in self.l0_module.lambdas:
                lambda_1_name = f"lambda_1_{mask_name}"
                lambda_2_name = f"lambda_2_{mask_name}"
                
                if lambda_1_name in self.l0_module.lambdas:
                    constraint_loss += self.l0_module.lambdas[lambda_1_name] * sparsity.mean()
                
                if lambda_2_name in self.l0_module.lambdas:
                    target_mask = getattr(self.l0_module.masks[mask_name], 'target_mask_size', None)
                    if target_mask is not None:
                        current_size = (1 - sparsity.mean()) * self.l0_module.masks[mask_name].mask_size
                        size_diff = current_size - target_mask
                        constraint_loss += self.l0_module.lambdas[lambda_2_name] * size_diff
        
        return constraint_loss
    
    def get_metrics(self, is_train: bool = False) -> Dict[str, Any]:
        """Get metrics for logging"""
        if is_train:
            return self.train_metrics
        else:
            return self.eval_metrics
    
    def prune_params(self, zs: Optional[Dict[str, torch.Tensor]] = None):
        """Prune model parameters based on masks"""
        if zs is None:
            zs = self.l0_module()
        
        # Prune backbone
        self.backbone.prune_params(zs)
        
        # Prune embedding heads
        hidden_z = zs.get('hidden_z', None)
        if hidden_z is not None:
            self.embedding_heads.prune_params(hidden_z)
    
    def get_model_info(self):
        """Get model architecture information"""
        return {
            'base_model_info': self.l0_module.base_model_info,
            'target_model_info': self.l0_module.target_model_info,
            'pruning_modules': self.l0_module.pruning_modules,
        }
    
    def compute_spearman_correlation(self, predicted_scores: torch.Tensor, 
                                   ground_truth_scores: torch.Tensor) -> float:
        """Compute Spearman correlation for STS evaluation"""
        try:
            from scipy.stats import spearmanr
            pred_np = predicted_scores.detach().cpu().numpy()
            gt_np = ground_truth_scores.detach().cpu().numpy()
            correlation, _ = spearmanr(pred_np, gt_np)
            return float(correlation)
        except ImportError:
            # Fallback to Pearson correlation if scipy not available
            pred_centered = predicted_scores - predicted_scores.mean()
            gt_centered = ground_truth_scores - ground_truth_scores.mean()
            correlation = (pred_centered * gt_centered).sum() / (
                torch.sqrt((pred_centered ** 2).sum() * (gt_centered ** 2).sum()) + 1e-8
            )
            return float(correlation)
    
    def extract_pruned_model(self) -> 'ComposerBGEM3':
        """Extract a pruned model with parameters permanently removed"""
        # Get current masks
        zs = self.l0_module()
        
        # Create new config based on pruned dimensions
        pruned_config = self._create_pruned_config(zs)
        
        # Create new model with pruned config
        pruned_model = ComposerBGEM3(pruned_config)
        
        # Copy and prune weights
        self._copy_pruned_weights(pruned_model, zs)
        
        return pruned_model
    
    def _create_pruned_config(self, zs: Dict[str, torch.Tensor]) -> DictConfig:
        """Create configuration for pruned model"""
        # This would create a new config with reduced dimensions
        # Implementation depends on specific pruning strategy
        pass
    
    def _copy_pruned_weights(self, target_model: 'ComposerBGEM3', 
                           zs: Dict[str, torch.Tensor]):
        """Copy weights from current model to pruned model"""
        # This would copy only the non-pruned weights
        # Implementation depends on specific pruning strategy
        pass
