#train_clean.py#!/usr/bin/env python3
"""
Production BGE-M3 Pruning Training Script
Focused on head/layer/intermediate pruning with MTEB support
"""

import torch
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Fix datasets import conflict by importing before local modules
import datasets as hf_datasets

from composer import Trainer, DataSpec
from composer.utils import reproducibility
from omegaconf import OmegaConf

# Import BGE components
from models.composer_bge_m3 import ComposerBGEM3
from data_loader import create_dataloader
from callbacks.pruning_callback import PruningCallback

def get_num_samples_in_batch(batch):
    """Get actual batch size from interleaved tensor pairs"""
    return batch['input_ids'].size(0) // 2

def split_batch(batch, microbatch_size):
    """Custom batch splitting for interleaved sentence pairs"""
    if microbatch_size is None:
        return [batch]
    
    actual_batch_size = batch['input_ids'].size(0) // 2  # True batch size
    if actual_batch_size <= microbatch_size:
        return [batch]
    
    # Split into microbatches
    microbatches = []
    for start_idx in range(0, actual_batch_size, microbatch_size):
        end_idx = min(start_idx + microbatch_size, actual_batch_size)
        
        # For interleaved tensors, we need indices: [start*2, start*2+1, ..., end*2-1]
        interleaved_indices = []
        for i in range(start_idx, end_idx):
            interleaved_indices.extend([i * 2, i * 2 + 1])
        
        microbatch = {
            'input_ids': batch['input_ids'][interleaved_indices],
            'attention_mask': batch['attention_mask'][interleaved_indices],
        }
        
        # Handle similarity_scores (has original batch dimension)
        if 'similarity_scores' in batch:
            microbatch['similarity_scores'] = batch['similarity_scores'][start_idx:end_idx]
        
        microbatches.append(microbatch)
    
    return microbatches

def setup_optimizer(model, cfg):
    """Production optimizer with stable L0 learning rates"""
    if not hasattr(model, 'l0_module'):
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Stable parameter groups for L0 pruning
    model_params = [p for n, p in model.named_parameters() 
                   if not n.startswith('l0_module') and p.requires_grad]
    mask_params = [p for p in model.l0_module.masks.parameters() if p.requires_grad]
    lambda_params = [p for p in model.l0_module.lambdas.parameters() if p.requires_grad]
    
    # LLM-Shearing style aggressive L0 learning rates
    base_lr = cfg.lr  # 5e-5
    l0_lr = 1.0       # Fixed high LR for L0 masks (1000x base)
    lagrangian_lr = 1.0  # Fixed high LR for Lagrangian multipliers
    
    param_groups = [
        {'params': model_params, 'lr': base_lr, 'weight_decay': cfg.weight_decay},
        {'params': mask_params, 'lr': l0_lr, 'weight_decay': 0.0},  # LLM-Shearing style
        {'params': lambda_params, 'lr': lagrangian_lr, 'weight_decay': 0.0}  # LLM-Shearing style
    ]
    
    return torch.optim.AdamW(param_groups, betas=cfg.get('betas', [0.9, 0.999]), eps=cfg.get('eps', 1e-8))

def main():
    parser = argparse.ArgumentParser(description='BGE-M3 Pruning Training')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default='sts', 
                       help='Dataset to use (sts, msmarco, or mteb/dataset-name)')
    
    args = parser.parse_args()
    
    # Set seed
    reproducibility.seed_all(args.seed)
    
    # Load configuration
    cfg = OmegaConf.load(args.config)
    print(f"🔥 BGE-M3 Production Pruning Training")
    print(f"📁 Config: {args.config}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎯 Pruning: head/layer/intermediate only")
    
    # Setup model
    print("🚀 Loading BGE-M3 with L0 pruning...")
    model = ComposerBGEM3(cfg.model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total parameters: {total_params:,}")
    print(f"📈 Trainable parameters: {trainable_params:,}")
    
    # Setup data with MTEB support
    print(f"📊 Loading {args.dataset} dataset from HuggingFace...")
    train_dataloader = create_dataloader(
        dataset_name=args.dataset,
        split="train",
        batch_size=cfg.get('batch_size', 16),
        max_length=cfg.get('max_length', 512)
    )
    
    # Use test split for datasets that don't have validation
    eval_split = "validation" if args.dataset == "sts" else "test"
    eval_dataloader = create_dataloader(
        dataset_name=args.dataset,
        split=eval_split,
        batch_size=cfg.get('batch_size', 16),
        max_length=cfg.get('max_length', 512)
    )
    
    # Wrap dataloaders in DataSpec for Composer batch size detection and custom splitting
    train_dataspec = DataSpec(
        dataloader=train_dataloader, 
        get_num_samples_in_batch=get_num_samples_in_batch,
        split_batch=split_batch
    )
    eval_dataspec = DataSpec(
        dataloader=eval_dataloader, 
        get_num_samples_in_batch=get_num_samples_in_batch,
        split_batch=split_batch
    )
    
    # Setup optimizer
    optimizer = setup_optimizer(model, cfg.optimizer)
    
    # Setup callbacks
    callbacks = [PruningCallback(log_interval=500)]
    
    # Convert PyTorch device names to Composer device names
    device = cfg.get('device', 'gpu')  # Default to 'gpu' for Composer
    
    # If user specified 'cuda' or torch.cuda.is_available(), use 'gpu'
    if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()):
        device = 'gpu'
    elif device == 'cpu':
        device = 'cpu'
    else:
        # Default fallback
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # Setup trainer with DataSpec for proper batch size detection
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataspec,
        eval_dataloader=eval_dataspec,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        save_interval=cfg.get('save_interval', '500ba'),
        save_folder=cfg.get('save_folder', 'experiments/checkpoints'),
        device=device,
        callbacks=callbacks,
        optimizers=optimizer,
        precision=cfg.get('precision', 'amp_bf16'),
    )
    # Start training
    print("🎯 Starting BGE-M3 pruning training...")
    trainer.fit()
    
    # Print final results
    print(f"✅ Production training completed!")
    if hasattr(model, 'l0_module'):
        zs = model.l0_module()
        print(f"🎯 Pruning results:")
        for mask_name, mask_tensor in zs.items():
            sparsity = (mask_tensor == 0).float().mean().item()
            print(f"  {mask_name}: {sparsity:.1%} sparsity")
    
    # Save pruned model in HuggingFace format
    hf_save_path = cfg.get('save_folder', 'experiments/production') + '_hf'
    model.save_pruned_hf_model(hf_save_path)
    print(f"💾 Model ready for deployment")

if __name__ == "__main__":
    main()
# models/compose_bge_m3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models.base import ComposerModel
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from typing import Dict, Optional, Any, Tuple
from transformers import AutoModel, AutoConfig, AutoTokenizer

from .l0_module_embedding import L0ModuleEmbedding
from .embedding_heads import BGEEmbeddingHeads
from .bge_m3_backbone import MaskedBGEM3Backbone

class ComposerBGEM3(ComposerModel):
    """BGE-M3 model with L0 pruning and Composer interface"""
    
    def __init__(self, cfg):
        super().__init__()
        
        # Load pretrained BGE-M3 model and config
        model_name = getattr(cfg, 'base_model', 'BAAI/bge-m3')
        self.base_model_name = model_name  # Store for HF export
        base_model = AutoModel.from_pretrained(model_name)
        self.config = base_model.config
        
        # Create masked backbone with original config
        self.backbone = MaskedBGEM3Backbone(self.config)
        self.backbone.load_state_dict(base_model.state_dict(), strict=False)
        
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
        
        # Validate configuration consistency
        self._validate_config()
        
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
        
        # Get L0 masks
        l0_output = self.l0_module()
        
        # Forward through masked backbone with L0 masks
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_z=l0_output.get('layer_z'),
            head_z=l0_output.get('head_z'),
            intermediate_z=l0_output.get('intermediate_z'),
        )
        
        # Get embeddings from heads (dense only for training efficiency)
        embedding_outputs = self.embedding_heads(
            hidden_states=backbone_outputs["last_hidden_state"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_sparse=False,
            return_multi_vector=False,
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
        """Production loss computation with proper tensor handling"""
        embeddings = outputs['embeddings']
        l0_output = outputs['l0_output']
        # Calculate batch size from interleaved tensor dimensions
        batch_size = batch['input_ids'].size(0) // 2
        
        total_loss = 0.0
        
        # Task-specific loss computation (infer task from batch contents)
        if 'similarity_scores' in batch:
            # STS task
            sts_loss = self.compute_sts_loss(embeddings, batch, batch_size)
            total_loss += sts_loss
        else:
            # Retrieval task
            contrastive_loss = self.compute_contrastive_loss(embeddings, batch_size)
            total_loss += contrastive_loss
        
        # L0 sparsity loss for pruning (scaled to balance with task loss)
        if hasattr(self.l0_module, 'get_sparsity_loss'):
            sparsity_loss, expected_sparsity, expected_score = self.l0_module.get_sparsity_loss()
            constraint_loss = self.compute_constraint_loss(expected_sparsity)
            # Scale pruning losses to match task loss magnitude (typically 1-5)
            pruning_weight = 20.0  # Amplify small sparsity losses to be significant
            total_loss += pruning_weight * (sparsity_loss + constraint_loss)
        
        return total_loss
    
    def compute_sts_loss(self, embeddings: Dict[str, torch.Tensor], batch: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Production STS loss with proper paired sentence handling"""
        dense_emb = embeddings['dense_embedding']  # [batch_size * 2, embedding_dim]
        similarity_scores = batch['similarity_scores']  # [batch_size]
        
        # Extract sentence pairs from interleaved format
        sent1_emb = dense_emb[0::2]  # Even indices: first sentences [batch_size, embedding_dim]
        sent2_emb = dense_emb[1::2]  # Odd indices: second sentences [batch_size, embedding_dim]
        
        # Compute cosine similarity
        predicted_sim = F.cosine_similarity(sent1_emb, sent2_emb, dim=-1)
        
        # Scale to [0, 5] range to match STS scores
        predicted_sim = (predicted_sim + 1) * 2.5
        
        # MSE loss
        sts_loss = F.mse_loss(predicted_sim, similarity_scores)
        return sts_loss
    
    def compute_contrastive_loss(self, embeddings: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """Production contrastive loss for query-passage pairs"""
        dense_emb = embeddings['dense_embedding']  # [batch_size * 2, embedding_dim]
        
        # Extract queries and passages from interleaved format
        query_emb = dense_emb[0::2]  # Even indices: queries [batch_size, embedding_dim]
        passage_emb = dense_emb[1::2]  # Odd indices: passages [batch_size, embedding_dim]
        
        # Compute similarity matrix: queries vs all passages
        similarity_matrix = torch.matmul(query_emb, passage_emb.t()) / self.temperature
        
        # Labels: each query should match its corresponding passage (diagonal)
        labels = torch.arange(batch_size, device=query_emb.device)
        
        # InfoNCE loss
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        return contrastive_loss
    
    def compute_constraint_loss(self, expected_sparsity: Dict[str, torch.Tensor]) -> torch.Tensor:
        """LLM-Shearing style constraint loss with warmup and quadratic penalty"""
        constraint_loss = 0.0
        
        for mask_name, sparsity in expected_sparsity.items():
            if mask_name in self.l0_module.masks:
                mask = self.l0_module.masks[mask_name]
                
                if hasattr(mask, 'target_sparsity') and mask.target_sparsity is not None:
                    # Get warmup target sparsity (gradually increases from 0 to final target)
                    current_target = self.l0_module.get_warmup_target_sparsity(mask.target_sparsity)
                    
                    # LLM-Shearing style: Linear + Quadratic penalty
                    sparsity_diff = sparsity.mean() - current_target
                    linear_penalty = torch.abs(sparsity_diff)
                    quadratic_penalty = sparsity_diff ** 2
                    
                    # Combine linear + quadratic (quadratic is stronger for large violations)
                    constraint_loss += linear_penalty + 5.0 * quadratic_penalty
        
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
        
        # Note: Hidden dimension pruning removed for production version
    
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
    
    def save_pruned_hf_model(self, save_path: str, tokenizer_name: str = None):
        """Save pruned model in HuggingFace format for production use"""
        import sys
        import os
        from pathlib import Path
        
        # Add project root to path for imports
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from utils.hf_export import export_pruned_backbone_clean
        import json
        
        # Use eval mode for deterministic masks
        was_training = self.training
        self.eval()
        
        # Get deterministic masks and apply pruning
        zs = self.l0_module()
        print("\n🎯 Applying pruning masks...")
        for mask_name, mask_tensor in zs.items():
            sparsity = (mask_tensor == 0).float().mean().item()
            print(f"  {mask_name}: {sparsity:.1%} sparsity")
        
        # Actually remove pruned parameters
        self.prune_params(zs)
        self._validate_pruned_model()
        
        # Save backbone using clean export (no padding)
        print(f"\n💾 Saving pruned model to {save_path}")
        base_model_name = tokenizer_name or getattr(self, 'base_model_name', 'BAAI/bge-m3')
        export_pruned_backbone_clean(self.backbone, save_path, base_model_name)
        
        # Save pruning info
        pruning_info = {
            'pruning_results': {name: float((mask == 0).float().mean()) for name, mask in zs.items()},
            'base_model': base_model_name,
            'final_config': {
                'num_hidden_layers': len(self.backbone.encoder.layer),
                'num_attention_heads': self.backbone.encoder.layer[0].attention.num_attention_heads if len(self.backbone.encoder.layer) > 0 else 0,
                'intermediate_size': self.backbone.encoder.layer[0].intermediate.dense.out_features if len(self.backbone.encoder.layer) > 0 else 0,
                'hidden_size': self.config.hidden_size
            }
        }
        
        with open(os.path.join(save_path, 'pruning_info.json'), 'w') as f:
            json.dump(pruning_info, f, indent=2)
        
        print(f"✅ Clean pruned model saved in HuggingFace format!")
        print(f"📁 Location: {save_path}")
        print(f"🔧 Usage: model = AutoModel.from_pretrained('{save_path}')")
        
        # Restore training mode
        if was_training:
            self.train()
        
        return save_path
    
    def _validate_config(self):
        """Validate model configuration for mathematical consistency"""
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.config.hidden_size}) must be divisible by "
                f"number of attention heads ({self.config.num_attention_heads}). "
                f"Adjust configuration to use valid combinations."
            )
    
    def _validate_pruned_model(self):
        """Validate pruned model is in correct state for production use"""
        # Validate backbone configuration
        backbone_config = self.backbone.config
        
        # Check layer count consistency
        actual_layers = len(self.backbone.encoder.layer)
        config_layers = backbone_config.num_hidden_layers
        if actual_layers != config_layers:
            raise ValueError(f"Layer count mismatch: actual={actual_layers}, config={config_layers}")
        
        # Check attention head consistency
        if backbone_config.hidden_size % backbone_config.num_attention_heads != 0:
            raise ValueError(f"Invalid attention configuration after pruning: "
                           f"hidden_size={backbone_config.hidden_size}, "
                           f"num_attention_heads={backbone_config.num_attention_heads}")
        
        print(f"✅ Model validation passed: {actual_layers} layers, "
              f"{backbone_config.num_attention_heads} heads, "
              f"{backbone_config.intermediate_size} intermediate size")
#models//bge_m3_backbone.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel, XLMRobertaConfig
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from typing import Optional, Dict, Any

class MaskedXLMRobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)

        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        head_z: Optional[torch.Tensor] = None,
    ) -> tuple:

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if head_z is not None:
            # Reshape for broadcasting: [num_heads] → [1, num_heads, 1, 1]
            head_z = head_z.view(1, -1, 1, 1)
            query_layer = query_layer * head_z
            key_layer = key_layer * head_z
            value_layer = value_layer * head_z

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class MaskedXLMRobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor, 
                intermediate_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        
        if intermediate_z is not None:
            hidden_states = hidden_states * intermediate_z
            
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class MaskedXLMRobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MaskedXLMRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MaskedXLMRobertaAttention(config)
        self.intermediate = MaskedXLMRobertaIntermediate(config)
        self.output = MaskedXLMRobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        head_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
    ) -> tuple:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            head_z=head_z,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        layer_output = self.feed_forward_chunk(attention_output, intermediate_z)
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output, intermediate_z=None):
        intermediate_output = self.intermediate(attention_output, intermediate_z)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class MaskedXLMRobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MaskedXLMRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        layer_z: Optional[torch.Tensor] = None,
        head_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
    ) -> tuple:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            # Check if layer is pruned
            if layer_z is not None and layer_z[i].item() == 0:
                continue

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Get layer-specific masks
            layer_head_z = head_z[i] if head_z is not None else None
            layer_intermediate_z = intermediate_z[i] if intermediate_z is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                output_attentions,
                layer_head_z,
                layer_intermediate_z,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

class MaskedBGEM3Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = XLMRobertaModel(config).embeddings
        self.encoder = MaskedXLMRobertaEncoder(config)
        self.pooler = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        layer_z: Optional[torch.Tensor] = None,
        head_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
    ) -> tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            layer_z=layer_z,
            head_z=head_z,
            intermediate_z=intermediate_z,
        )
        sequence_output = encoder_outputs[0]

        return {
            "last_hidden_state": sequence_output,
            "hidden_states": encoder_outputs[1] if output_hidden_states else None,
            "attentions": encoder_outputs[2] if output_attentions else None,
        }

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: tuple) -> torch.Tensor:
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})")

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        return extended_attention_mask

    def get_head_mask(self, head_mask: Optional[torch.Tensor], num_hidden_layers: int) -> Optional[torch.Tensor]:
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def prune_params(self, zs: Dict[str, torch.Tensor]):
        """Prune parameters based on masks"""
        if "layer_z" in zs:
            # Remove pruned layers and make indices contiguous
            layer_mask = zs["layer_z"]
            remaining_layers = torch.where(layer_mask > 0)[0]
            print(f"Pruning layers: keeping {len(remaining_layers)}/{len(layer_mask)} layers {remaining_layers.tolist()}")
            
            # Create new ModuleList with contiguous indices
            self.encoder.layer = nn.ModuleList([self.encoder.layer[i] for i in remaining_layers])
            self.config.num_hidden_layers = len(remaining_layers)
            
            print(f"New layer count: {self.config.num_hidden_layers} (indices now 0-{self.config.num_hidden_layers-1})")

        if "head_z" in zs:
            # Prune attention heads
            head_mask = zs["head_z"]
            total_remaining_heads = 0
            for i, layer in enumerate(self.encoder.layer):
                if i < head_mask.shape[0]:
                    layer_head_mask = head_mask[i].squeeze()
                    pruned_heads = [j for j, mask in enumerate(layer_head_mask) if mask == 0]
                    layer.attention.prune_heads(pruned_heads)
                    total_remaining_heads += layer.attention.num_attention_heads
            
            # Update config ensuring mathematical consistency
            if len(self.encoder.layer) > 0:
                avg_heads = total_remaining_heads // len(self.encoder.layer)
                # Ensure hidden_size is divisible by num_attention_heads
                if self.config.hidden_size % avg_heads != 0:
                    # Adjust to nearest valid head count that divides hidden_size
                    for h in range(avg_heads, 0, -1):
                        if self.config.hidden_size % h == 0:
                            avg_heads = h
                            break
                self.config.num_attention_heads = avg_heads



        if "intermediate_z" in zs:
            # Prune intermediate dimensions in MLP layers
            intermediate_mask = zs["intermediate_z"]
            total_remaining_intermediate = 0
            for i, layer in enumerate(self.encoder.layer):
                if i < intermediate_mask.shape[0]:
                    layer_int_mask = intermediate_mask[i]
                    remaining_idx = torch.where(layer_int_mask > 0)[0]
                    total_remaining_intermediate += len(remaining_idx)
                    
                    # Prune intermediate layers
                    if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense'):
                        old_weight = layer.intermediate.dense.weight.data
                        old_bias = layer.intermediate.dense.bias.data
                        layer.intermediate.dense = nn.Linear(old_weight.size(1), len(remaining_idx))
                        layer.intermediate.dense.weight.data = old_weight[remaining_idx]
                        layer.intermediate.dense.bias.data = old_bias[remaining_idx]
                    
                    if hasattr(layer, 'output') and hasattr(layer.output, 'dense'):
                        old_weight = layer.output.dense.weight.data
                        layer.output.dense = nn.Linear(len(remaining_idx), old_weight.size(0))
                        layer.output.dense.weight.data = old_weight[:, remaining_idx]
            
            # Update global config with average intermediate size per layer
            if len(self.encoder.layer) > 0:
                self.config.intermediate_size = total_remaining_intermediate // len(self.encoder.layer)
#models/embedding_heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class DenseEmbeddingHead(nn.Module):
    """Dense embedding head for BGE-M3"""
    
    def __init__(self, hidden_size: int, output_dim: int = None):
        super().__init__()
        self.output_dim = output_dim or hidden_size
        self.dense = nn.Linear(hidden_size, self.output_dim)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pool the hidden states (use CLS token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with attention mask
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Use CLS token
            pooled = hidden_states[:, 0]
        
        dense_embedding = self.dense(pooled)
        dense_embedding = self.activation(dense_embedding)
        return F.normalize(dense_embedding, p=2, dim=-1)

class SparseEmbeddingHead(nn.Module):
    """Sparse embedding head for BGE-M3"""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.vocab_size = vocab_size
        
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute token importance weights
        token_weights = self.linear(hidden_states).squeeze(-1)
        
        if attention_mask is not None:
            token_weights = token_weights * attention_mask
        
        # Apply ReLU and normalize
        token_weights = F.relu(token_weights)
        
        # Create sparse representation
        batch_size, seq_len = input_ids.shape
        sparse_embedding = torch.zeros(batch_size, self.vocab_size, device=input_ids.device)
        
        for i in range(batch_size):
            for j in range(seq_len):
                if attention_mask is None or attention_mask[i, j] > 0:
                    token_id = input_ids[i, j].item()
                    if 0 <= token_id < self.vocab_size:
                        sparse_embedding[i, token_id] += token_weights[i, j]
        
        return sparse_embedding

class MultiVectorEmbeddingHead(nn.Module):
    """Multi-vector embedding head for BGE-M3"""
    
    def __init__(self, hidden_size: int, output_dim: int = None):
        super().__init__()
        self.output_dim = output_dim or hidden_size
        self.dense = nn.Linear(hidden_size, self.output_dim)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Transform all token representations
        multi_vector = self.dense(hidden_states)
        
        if attention_mask is not None:
            # Zero out padded tokens
            multi_vector = multi_vector * attention_mask.unsqueeze(-1)
        
        # Normalize each vector
        multi_vector = F.normalize(multi_vector, p=2, dim=-1)
        return multi_vector

class BGEEmbeddingHeads(nn.Module):
    """Combined embedding heads for BGE-M3 model"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = getattr(config, 'vocab_size', 250002)
        
        # Initialize all three heads
        self.dense_head = DenseEmbeddingHead(self.hidden_size)
        self.sparse_head = SparseEmbeddingHead(self.hidden_size, self.vocab_size)
        self.multi_vector_head = MultiVectorEmbeddingHead(self.hidden_size)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dense: bool = True,
                return_sparse: bool = True,
                return_multi_vector: bool = True) -> Dict[str, torch.Tensor]:
        
        outputs = {}
        
        if return_dense:
            outputs['dense_embedding'] = self.dense_head(hidden_states, attention_mask)
        
        if return_sparse:
            outputs['sparse_embedding'] = self.sparse_head(hidden_states, input_ids, attention_mask)
        
        if return_multi_vector:
            outputs['multi_vector_embedding'] = self.multi_vector_head(hidden_states, attention_mask)
        
        return outputs
    
    def compute_similarity(self, 
                          query_embeddings: Dict[str, torch.Tensor],
                          doc_embeddings: Dict[str, torch.Tensor],
                          weights: Dict[str, float] = None) -> torch.Tensor:
        """Compute similarity scores between query and document embeddings"""
        
        if weights is None:
            weights = {'dense': 1.0, 'sparse': 0.3, 'multi_vector': 1.0}
        
        similarities = []
        
        # Dense similarity
        if 'dense_embedding' in query_embeddings and 'dense_embedding' in doc_embeddings:
            dense_sim = torch.mm(query_embeddings['dense_embedding'], 
                               doc_embeddings['dense_embedding'].t())
            similarities.append(weights.get('dense', 1.0) * dense_sim)
        
        # Sparse similarity
        if 'sparse_embedding' in query_embeddings and 'sparse_embedding' in doc_embeddings:
            sparse_sim = torch.mm(query_embeddings['sparse_embedding'],
                                doc_embeddings['sparse_embedding'].t())
            similarities.append(weights.get('sparse', 0.3) * sparse_sim)
        
        # Multi-vector similarity (max pooling)
        if 'multi_vector_embedding' in query_embeddings and 'multi_vector_embedding' in doc_embeddings:
            q_mv = query_embeddings['multi_vector_embedding']  # [batch_q, seq_len, dim]
            d_mv = doc_embeddings['multi_vector_embedding']    # [batch_d, seq_len, dim]
            
            # Compute all pairwise similarities and take max
            mv_sim = torch.einsum('qsd,dtd->qst', q_mv, d_mv.transpose(0, 1))
            mv_sim = mv_sim.max(dim=-1)[0].max(dim=-1)[0]  # Max over both sequence dimensions
            similarities.append(weights.get('multi_vector', 1.0) * mv_sim)
        
        # Combine similarities
        if similarities:
            total_similarity = torch.stack(similarities, dim=0).sum(dim=0)
        else:
            raise ValueError("No valid embedding pairs found for similarity computation")
        
        return total_similarity
    
    def prune_params(self, hidden_z: Optional[torch.Tensor] = None):
        """Prune embedding head parameters based on hidden dimension mask"""
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            
            # Prune dense head
            if hasattr(self.dense_head.dense, 'weight'):
                self.dense_head.dense.weight = nn.Parameter(
                    self.dense_head.dense.weight.data[:, remaining_index]
                )
            
            # Prune sparse head
            if hasattr(self.sparse_head.linear, 'weight'):
                self.sparse_head.linear.weight = nn.Parameter(
                    self.sparse_head.linear.weight.data[:, remaining_index]
                )
            
            # Prune multi-vector head
            if hasattr(self.multi_vector_head.dense, 'weight'):
                self.multi_vector_head.dense.weight = nn.Parameter(
                    self.multi_vector_head.dense.weight.data[:, remaining_index]
                )

#models/l0_module_embeddings.py
import math
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import Namespace as NS
from typing import Any, List
from composer.core.time import Time

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class Mask(nn.Module):
    def __init__(self, 
                 name: str,
                 mask_shape: List, 
                 num_params_per_mask: int, 
                 mask_output_shape: List, 
                 target_sparsity: float,
                 target_mask_size: int,
                 device: str,
                 eval_target_model: bool=True) -> None:
        super().__init__()
        self.name = name
        self.num_params_per_mask = num_params_per_mask
        self.mask_output_shape = mask_output_shape
        self.target_sparsity = target_sparsity
        self.droprate_init = 0.5
        self.temperature = 2./3.
        self.magical_number = 0.8
        self.beta = 0.83  # Hard Concrete temperature parameter
        self.device = device
        
        self.target_mask_size = target_mask_size
        self.eval_target_model = eval_target_model
        
        self.z_loga = self.initialize_mask(mask_shape) 
        self.mask_size = self.z_loga.shape[-1]
        
    def param_init_fn(self, module):
        mean = 0
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean, 1e-2)
        else:
            for tensor in module.parameters():
                tensor.data.normal_(mean, 1e-2)
    
    def initialize_mask(self, mask_shape):
        z_loga = nn.Parameter(torch.zeros(mask_shape, device=self.device))
        
        # Target-aware initialization to bias toward target architecture
        if self.target_mask_size is not None and mask_shape[-1] > 0:
            target_keep_prob = self.target_mask_size / mask_shape[-1]
            target_keep_prob = max(0.01, min(0.99, target_keep_prob))  # Clamp to valid range
            alpha_init = math.log(target_keep_prob / (1 - target_keep_prob))
            z_loga.data.fill_(alpha_init)
        else:
            self.param_init_fn(z_loga)
        
        return z_loga
    
    def cdf_qz(self, x=None):
        """
        CDF of Hard Concrete distribution - probability that output is 0
        """
        if x is None:
            x = self.z_loga
        # Threshold where stretched sigmoid becomes 0
        threshold = -limit_a / (limit_b - limit_a)  # 0.1 / 1.2 ≈ 0.083
        # Inverse sigmoid to find x value that gives this threshold
        threshold_logit = math.log(threshold / (1 - threshold))
        # CDF: P(output = 0) = P(x < threshold_logit * beta)
        return torch.sigmoid((threshold_logit * self.beta - x) / self.beta)
    
    def sample_z(self):
        # Hard Concrete distribution sampling
        eps = torch.rand_like(self.z_loga)
        eps = torch.clamp(eps, epsilon, 1 - epsilon)  # Avoid log(0)
        
        # Step 1: Sample from logistic distribution
        s = torch.sigmoid((torch.log(eps / (1 - eps)) + self.z_loga) / self.beta)
        
        # Step 2: Stretch to [γ, ζ] range (limit_a, limit_b)
        s_stretched = s * (limit_b - limit_a) + limit_a
        
        # Step 3: Hard clamp to [0, 1] - this creates the "hard" concrete distribution
        z = torch.clamp(s_stretched, 0, 1)
        
        return z
    
    def _deterministic_z(self, z_loga):
        # CDF-based expected sparsity calculation
        if self.target_mask_size is not None and self.eval_target_model:
            # Use exact target for final pruning
            expected_num_zeros = z_loga.shape[-1] - self.target_mask_size
        else:
            # Use CDF to compute expected sparsity from learned distribution
            expected_score = 1 - self.cdf_qz(z_loga)  # Probability of keeping
            expected_num_nonzeros = expected_score.sum()
            expected_num_zeros = z_loga.nelement() - expected_num_nonzeros.item()
        
        try:
            num_zeros = round(expected_num_zeros)
        except:
            print("num of zeros is nan....")
            num_zeros = 0
            
        num_zeros = max(0, min(num_zeros, z_loga.shape[-1]))
        
        soft_mask = torch.sigmoid(z_loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(z_loga.device)
            else:
                _, indices = torch.topk(z_loga, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask
    
    def deterministic_z(self):
        if self.z_loga.ndim == 1:
            z = self._deterministic_z(self.z_loga).reshape(*self.mask_output_shape)
        else:
            z_loga = self.z_loga.reshape(-1, self.z_loga.shape[-1])
            z = []
            for i in range(z_loga.shape[0]):
                z_ = self._deterministic_z(z_loga[i])
                z.append(z_)
            z = torch.stack(z).reshape(*self.mask_output_shape)
        return z
    
    def forward(self):
        func = self.sample_z if self.training else self.deterministic_z
        z = func().reshape(self.mask_output_shape)
        return z
    
    def constrain_parameters(self):
        self.z_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def calculate_expected_score_sparsity(self):
        # CDF-based expected score calculation
        # Use same method as _deterministic_z for consistency
        expected_score = 1 - self.cdf_qz(self.z_loga)  # Probability of keeping
        soft_mask = expected_score
        sparsity = 1 - soft_mask.mean(dim=-1)
        return soft_mask, sparsity

class L0ModuleEmbedding(nn.Module):
    def __init__(self, cfg, device, pretrained_model=None):
        super(L0ModuleEmbedding, self).__init__()
        
        # Extract model info from pretrained model if available
        if pretrained_model:
            self.base_model_info = self.extract_model_info(pretrained_model)
        else:
            self.base_model_info = self.set_model_info(cfg)
            
        l0_module_cfg = cfg.l0_module
        self.target_model_info = None
        target_model_cfg = getattr(l0_module_cfg, "target_model", None)
        if target_model_cfg is not None:
            # Set target model info if it has any pruning-related fields
            if (hasattr(target_model_cfg, 'n_layers') or hasattr(target_model_cfg, 'n_heads') or 
                hasattr(target_model_cfg, 'intermediate_size') or hasattr(target_model_cfg, 'd_model') or 
                hasattr(target_model_cfg, 'hidden_size')):
                self.target_model_info = self.set_model_info(target_model_cfg)
        
        # Focus on head/layer/intermediate pruning only
        self.pruning_modules = [m for m in l0_module_cfg.pruning_modules if m in ['head', 'layer', 'intermediate']]
        self.start_sparsity = l0_module_cfg.start_sparsity
        self.lagrangian_warmup_steps = Time.from_timestring(l0_module_cfg.lagrangian_warmup_steps).value
        self.device = device
        self.eval_target_model = l0_module_cfg.get("eval_target_model", True)
        
        # LLM-Shearing style sparsity warmup (20% of total training for warmup)
        self.sparsity_warmup_steps = int(0.2 * Time.from_timestring("12000ba").value)  # 20% of 12000 steps
        self.current_step = 0
        
        self.masks = nn.ModuleDict()
        self.lambdas = nn.ParameterDict()
        
        for module_name in self.pruning_modules:
            self.initialize_one_module(module_name)
    
    def update_training_step(self, step: int):
        """Update current training step for sparsity warmup"""
        self.current_step = step
    
    def get_warmup_target_sparsity(self, final_target_sparsity: float) -> float:
        """LLM-Shearing style sparsity warmup: 0 → target over warmup period"""
        if self.current_step >= self.sparsity_warmup_steps:
            return final_target_sparsity
        
        # Linear warmup from 0 to final_target_sparsity
        progress = self.current_step / self.sparsity_warmup_steps
        return progress * final_target_sparsity
    
    def extract_model_info(self, model):
        """Extract model configuration from pretrained model"""
        info = NS()
        config = model.config
        info.hidden_size = config.hidden_size
        info.intermediate_size = config.intermediate_size
        info.num_attention_heads = config.num_attention_heads
        info.num_layers = config.num_hidden_layers  # Map to num_layers
        info.head_dim = config.hidden_size // config.num_attention_heads
        info.vocab_size = config.vocab_size
        info.max_position_embeddings = config.max_position_embeddings
        
        # Calculate parameters for BGE-M3
        info.params_per_head = info.hidden_size * info.head_dim + info.head_dim * info.hidden_size
        info.params_per_intermediate_dim = info.hidden_size + info.hidden_size
        info.params_per_mlp_layer = info.hidden_size * info.intermediate_size + info.intermediate_size * info.hidden_size
        return info
    

    
    def set_model_info(self, cfg):
        info = NS()
        # Handle both old and new config field names
        info.hidden_size = getattr(cfg, 'd_model', getattr(cfg, 'hidden_size', 1024))
        info.intermediate_size = getattr(cfg, 'intermediate_size', 4096)
        info.num_attention_heads = getattr(cfg, 'n_heads', getattr(cfg, 'num_attention_heads', 16))
        info.num_layers = getattr(cfg, 'n_layers', getattr(cfg, 'num_hidden_layers', 24))
        info.vocab_size = getattr(cfg, 'vocab_size', 250002)
        
        # BGE-M3 specific parameters for XLM-RoBERTa
        info.head_dim = info.hidden_size // info.num_attention_heads
        info.params_per_head = info.hidden_size * info.head_dim + info.head_dim * info.hidden_size
        info.params_per_intermediate_dim = info.hidden_size + info.hidden_size
        info.params_per_mlp_layer = info.hidden_size * info.intermediate_size + info.intermediate_size * info.hidden_size
        
        return info
    
    def compute_num_params(self, model_info):
        # Embedding layer
        embedding_params = model_info.vocab_size * model_info.hidden_size
        
        # Per layer parameters
        per_layer_params = (
            # Self-attention: Q, K, V projections + output projection
            4 * model_info.hidden_size * model_info.hidden_size +
            # MLP: intermediate + output projections  
            2 * model_info.hidden_size * model_info.intermediate_size +
            # Layer norms (2 per layer)
            2 * model_info.hidden_size
        )
        
        total_params = embedding_params + model_info.num_layers * per_layer_params
        
        # Add output heads for BGE-M3 (dense, sparse, multi-vector)
        output_head_params = 3 * model_info.hidden_size * model_info.hidden_size
        total_params += output_head_params
        
        return total_params
    
    def compute_prunable_params(self):
        prunable_model_size = 0
        
        # Attention heads pruning
        if "head" in self.pruning_modules:
            prunable_head_size = (
                self.base_model_info.num_layers * 
                self.base_model_info.num_attention_heads * 
                self.base_model_info.params_per_head
            )
            prunable_model_size += prunable_head_size
        
        # MLP intermediate dimension pruning
        if "intermediate" in self.pruning_modules:
            prunable_mlp_size = (
                self.base_model_info.num_layers * 
                self.base_model_info.intermediate_size * 
                self.base_model_info.params_per_intermediate_dim
            )
            prunable_model_size += prunable_mlp_size
        
        # Layer pruning
        if "layer" in self.pruning_modules:
            layer_params = (
                4 * self.base_model_info.hidden_size * self.base_model_info.hidden_size +
                2 * self.base_model_info.hidden_size * self.base_model_info.intermediate_size +
                2 * self.base_model_info.hidden_size
            )
            prunable_model_size += self.base_model_info.num_layers * layer_params
        
        # Skip hidden dimension pruning (not included in production version)
        
        return prunable_model_size
    
    def initialize_one_module(self, module_name: str):
        func_name = f"initialize_{module_name}"
        method = getattr(self, func_name)
        method()
    
    # Hidden dimension pruning removed for production version
    # Focus on structured pruning: head, layer, intermediate only
    
    def initialize_head(self):
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads]
        num_params_per_mask = self.base_model_info.params_per_head
        # Simplified mask shape for proper broadcasting: [num_layers, num_heads]
        mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads]
        
        target_head_sparsity = None
        target_mask_size = None
        if self.target_model_info is not None and hasattr(self.target_model_info, 'num_attention_heads'):
            target_head_sparsity = 1 - self.target_model_info.num_attention_heads / self.base_model_info.num_attention_heads
            target_mask_size = self.target_model_info.num_attention_heads
            self.lambdas.update({
                "lambda_1_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                "lambda_2_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device))
            })
        
        head_mask = Mask(
            name="head",
            mask_shape=mask_shape,
            num_params_per_mask=num_params_per_mask,
            mask_output_shape=mask_output_shape,
            target_sparsity=target_head_sparsity,
            target_mask_size=target_mask_size,
            device=self.device,
            eval_target_model=self.eval_target_model
        )
        self.masks["head"] = head_mask
    
    def initialize_layer(self):
        mask_shape = [self.base_model_info.num_layers]
        num_params_per_mask = (
            4 * self.base_model_info.hidden_size * self.base_model_info.hidden_size +
            2 * self.base_model_info.hidden_size * self.base_model_info.intermediate_size +
            2 * self.base_model_info.hidden_size
        )
        mask_output_shape = [self.base_model_info.num_layers]
        
        target_layer_sparsity = None
        target_mask_size = None
        if self.target_model_info is not None and hasattr(self.target_model_info, 'num_layers'):
            target_layer_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            self.lambdas.update({
                "lambda_1_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                "lambda_2_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device))
            })
        
        layer_mask = Mask(
            name="layer",
            mask_shape=mask_shape,
            num_params_per_mask=num_params_per_mask,
            mask_output_shape=mask_output_shape,
            target_sparsity=target_layer_sparsity,
            target_mask_size=target_mask_size,
            device=self.device,
            eval_target_model=self.eval_target_model
        )
        self.masks["layer"] = layer_mask
    
    def initialize_intermediate(self):
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.intermediate_size]
        num_params_per_mask = self.base_model_info.params_per_intermediate_dim
        # Simplified mask shape: [num_layers, intermediate_size]
        mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.intermediate_size]
        
        target_int_sparsity = None
        target_mask_size = None
        if self.target_model_info is not None and hasattr(self.target_model_info, 'intermediate_size'):
            target_int_sparsity = 1 - self.target_model_info.intermediate_size / self.base_model_info.intermediate_size
            target_mask_size = self.target_model_info.intermediate_size
            self.lambdas.update({
                "lambda_1_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                "lambda_2_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device))
            })
        
        int_mask = Mask(
            name="intermediate",
            mask_shape=mask_shape,
            num_params_per_mask=num_params_per_mask,
            mask_output_shape=mask_output_shape,
            target_sparsity=target_int_sparsity,
            target_mask_size=target_mask_size,
            device=self.device,
            eval_target_model=self.eval_target_model
        )
        self.masks["intermediate"] = int_mask
    
    def get_sparsity_loss(self):
        sparsity_loss = 0
        expected_sparsity = {}
        expected_score = {}
        
        for mask_name, mask in self.masks.items():
            score, sparsity = mask.calculate_expected_score_sparsity()
            expected_sparsity[mask_name] = sparsity
            expected_score[mask_name] = score
            
            if mask_name in ["head", "layer", "intermediate"]:
                if mask.target_sparsity is not None:
                    sparsity_loss += torch.abs(sparsity.mean() - mask.target_sparsity)
        
        return sparsity_loss, expected_sparsity, expected_score
    
    def forward(self):
        zs = {}
        for mask_name, mask in self.masks.items():
            zs[f"{mask_name}_z"] = mask()
        return zs
    
    def constrain_parameters(self):
        for mask in self.masks.values():
            mask.constrain_parameters()
        
        # Constrain lambda parameters to be non-negative
        for param in self.lambdas.values():
            param.data.clamp_(min=0.0)
    

#utils/hf_export.py
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
            
            # Attention output projection: map all_head_size back to hidden_size
            hf_layer.attention.output.dense.weight.data.zero_()
            # Create identity mapping: [hidden_size, all_head_size] where first all_head_size rows are identity
            hf_layer.attention.output.dense.weight.data[:all_head_size, :all_head_size] = torch.eye(all_head_size)
            hf_layer.attention.output.dense.bias.data.zero_()
            
        else:
            # No head pruning: direct copy
            for name in ['query', 'key', 'value']:
                backbone_param = getattr(backbone_layer.attention, name)
                hf_param = getattr(hf_layer.attention.self, name)
                hf_param.weight.data.copy_(backbone_param.weight.data)
                hf_param.bias.data.copy_(backbone_param.bias.data)
            
            # Identity output projection
            hf_layer.attention.output.dense.weight.data = torch.eye(hidden_size)
            hf_layer.attention.output.dense.bias.data.zero_()
        
        # 4. MLP layers - direct copy
        hf_layer.intermediate.dense.weight.data.copy_(backbone_layer.intermediate.dense.weight.data)
        hf_layer.intermediate.dense.bias.data.copy_(backbone_layer.intermediate.dense.bias.data)
        hf_layer.output.dense.weight.data.copy_(backbone_layer.output.dense.weight.data)
        hf_layer.output.dense.bias.data.copy_(backbone_layer.output.dense.bias.data)
        
        # 5. Layer norms - copy from backbone structure
        # Backbone has LayerNorm in output module, HF has it in attention.output and output
        hf_layer.attention.output.LayerNorm.weight.data.copy_(backbone_layer.output.LayerNorm.weight.data)
        hf_layer.attention.output.LayerNorm.bias.data.copy_(backbone_layer.output.LayerNorm.bias.data)
        hf_layer.output.LayerNorm.weight.data.copy_(backbone_layer.output.LayerNorm.weight.data)
        hf_layer.output.LayerNorm.bias.data.copy_(backbone_layer.output.LayerNorm.bias.data)


def export_pruned_backbone_clean(backbone, save_path, base_model_name="BAAI/bge-m3"):
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

#callbacks/pruning_callback.py
import torch
from composer.core import Callback, State
from composer.loggers import Logger
from typing import Dict, Any

class PruningCallback(Callback):
    """Minimal callback for pruning monitoring"""
    
    def __init__(self, log_interval: int = 500):
        self.log_interval = log_interval
        self.step_count = 0
    
    def batch_end(self, state: State, logger: Logger) -> None:
        """Called at the end of each training batch"""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self._log_pruning_metrics(state, logger)
            self._update_lagrangian_multipliers(state)
    
    def _log_pruning_metrics(self, state: State, logger: Logger) -> None:
        """Log pruning-specific metrics"""
        model = state.model
        
        if not hasattr(model, 'l0_module'):
            return
            
        l0_module = model.l0_module
        
        # Update current step for sparsity warmup
        current_step = state.timestamp.batch.value if hasattr(state.timestamp, 'batch') else 0
        l0_module.update_training_step(current_step)
        
        # Constrain parameters to valid ranges
        l0_module.constrain_parameters()
        
        metrics = {}
        
        # Log actual sparsity using deterministic masks (same as final results)
        zs = l0_module()
        for mask_name, mask_tensor in zs.items():
            # Remove '_z' suffix and calculate actual sparsity
            base_name = mask_name.replace('_z', '')
            sparsity = (mask_tensor == 0).float().mean().item()
            metrics[f'sparsity/{base_name}'] = sparsity
        
        # Log to console
        sparsity_str = ", ".join([f"{k.split('/')[-1]}: {v:.3f}" for k, v in metrics.items()])
        print(f"Step {self.step_count}: Sparsity - {sparsity_str}")
        
        # Log to logger if available
        if logger and hasattr(logger, 'log_metrics'):
            logger.log_metrics(metrics)
        
        if hasattr(model, 'l0_module'):
            l0_module = model.l0_module
            
            # Log constraint violations
            constraint_loss = self._compute_constraint_violations(l0_module)
            logger.log_metrics({"train_constraint_loss": constraint_loss})
            
            # Log effective model size
            effective_params = self._compute_effective_params(l0_module)
            logger.log_metrics({"train_effective_params": effective_params})
            
            # Log mask statistics
            mask_stats = self._compute_mask_statistics(l0_module)
            for key, value in mask_stats.items():
                logger.log_metrics({f"train_mask_{key}": value})
    
    def _update_lagrangian_multipliers(self, state: State) -> None:
        """Update Lagrangian multipliers based on constraint violations"""
        model = state.model
        
        if hasattr(model, 'l0_module') and hasattr(model.l0_module, 'lambdas'):
            l0_module = model.l0_module
            current_step = state.timestamp.batch.value
            
            # Only start updating after warmup
            if current_step > l0_module.lagrangian_warmup_steps:
                self._adjust_lagrangian_multipliers(l0_module)
    
    def _adjust_lagrangian_multipliers(self, l0_module) -> None:
        """Adjust Lagrangian multipliers based on constraint satisfaction"""
        learning_rate = 0.001  # Reduced learning rate for stability
        max_lambda = 10.0  # Maximum multiplier value to prevent explosion
        
        for param_name, param in l0_module.lambdas.items():
            if param_name.startswith("lambda_1_"):
                # Adjust based on sparsity constraint
                mask_name = param_name.replace("lambda_1_", "")
                if mask_name in l0_module.masks:
                    mask = l0_module.masks[mask_name]
                    _, sparsity = mask.calculate_expected_score_sparsity()
                    
                    if mask.target_sparsity is not None:
                        violation = sparsity.mean() - mask.target_sparsity
                        param.data += learning_rate * violation
                        param.data.clamp_(min=0.0, max=max_lambda)
    
    def _compute_constraint_violations(self, l0_module) -> float:
        """Compute total constraint violations"""
        total_violation = 0.0
        
        for mask_name, mask in l0_module.masks.items():
            if mask.target_sparsity is not None:
                _, sparsity = mask.calculate_expected_score_sparsity()
                violation = torch.abs(sparsity.mean() - mask.target_sparsity)
                total_violation += violation.item()
        
        return total_violation
    
    def _compute_effective_params(self, l0_module) -> int:
        """Compute effective number of parameters after pruning"""
        total_effective = 0
        
        for mask_name, mask in l0_module.masks.items():
            score, _ = mask.calculate_expected_score_sparsity()
            effective_size = score.sum().item()
            total_effective += int(effective_size * mask.num_params_per_mask)
        
        return total_effective
    
    def _compute_mask_statistics(self, l0_module) -> Dict[str, float]:
        """Compute statistics about mask distributions"""
        stats = {}
        
        for mask_name, mask in l0_module.masks.items():
            z_loga = mask.z_loga
            
            # Mean and std of log-alpha values
            stats[f"{mask_name}_mean_log_alpha"] = z_loga.mean().item()
            stats[f"{mask_name}_std_log_alpha"] = z_loga.std().item()
            
            # Expected sparsity
            _, sparsity = mask.calculate_expected_score_sparsity()
            stats[f"{mask_name}_expected_sparsity"] = sparsity.mean().item()
        
        return stats
    
    def _log_mask_visualizations(self, state: State, logger: Logger) -> None:
        """Create and log mask visualizations"""
        model = state.model
        
        if hasattr(model, 'l0_module'):
            l0_module = model.l0_module
            
            for mask_name, mask in l0_module.masks.items():
                # Create histogram of mask values
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    z_loga = mask.z_loga.detach().cpu().numpy()
                    if z_loga.ndim > 1:
                        z_loga = z_loga.flatten()
                    
                    ax.hist(z_loga, bins=50, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Log-alpha values')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of {mask_name} mask values')
                    ax.grid(True, alpha=0.3)
                    
                    # Log the figure
                    logger.log_images({f"mask_distribution_{mask_name}": fig})
                    plt.close(fig)
                except ImportError:
                    pass  # Skip visualization if matplotlib not available
    
    def epoch_end(self, state: State, logger: Logger) -> None:
        """Called at the end of each epoch"""
        self._log_pruning_summary(state, logger)
    
    def _log_pruning_summary(self, state: State, logger: Logger) -> None:
        """Log summary of pruning progress"""
        model = state.model
        
        if hasattr(model, 'l0_module'):
            l0_module = model.l0_module
            
            # Log target vs current architecture
            if hasattr(l0_module, 'target_model_info') and l0_module.target_model_info is not None:
                target_info = l0_module.target_model_info
                base_info = l0_module.base_model_info
                
                reduction_stats = {
                    'target_hidden_reduction': 1 - target_info.hidden_size / base_info.hidden_size,
                    'target_layer_reduction': 1 - target_info.num_layers / base_info.num_layers,
                    'target_head_reduction': 1 - target_info.num_attention_heads / base_info.num_attention_heads,
                    'target_intermediate_reduction': 1 - target_info.intermediate_size / base_info.intermediate_size
                }
                
                for key, value in reduction_stats.items():
                    logger.log_metrics({f"pruning_{key}": value})

#config.yaml
# Production BGE-M3 Pruning Configuration
# Focused on head/layer/intermediate pruning with MTEB support

model:
  base_model: "BAAI/bge-m3"
  use_sts_loss: true
  use_contrastive_loss: true
  temperature: 0.07
  
  l0_module:
    # Production pruning: head, layer, intermediate only
    pruning_modules: ["layer", "head", "intermediate"]
    start_sparsity: 0.0
    lagrangian_warmup_steps: "1000ba"
    eval_target_model: true
    
    # Target architecture for 50% reduction (production setting)
    target_model:
      n_layers: 12  # From 24
      n_heads: 8   # From 16
      intermediate_size: 2048  # From 4096

optimizer:
  # Stable learning rates for production
  lr: 3e-5
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.01

# Training settings
batch_size: 16
max_length: 512
max_duration: "10000ba"
eval_interval: "500ba"
save_interval: "1000ba"
save_folder: "experiments/production"
seed: 42
device: gpu
precision: "amp_bf16"
