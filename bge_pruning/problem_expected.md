I'll analyze the BGE M3 pruning codebase to identify shape mismatch issues and provide a minimal debug report. Let me start by examining the key files to understand the architecture and potential issues. Based on my analysis of the BGE M3 pruning codebase, I've identified several critical shape mismatch issues and architectural problems. Here's my minimal debug report:

## BGE M3 Pruning Shape Mismatch Debug Report

### **Critical Issues Identified**

#### 1. **Hook Registration Shape Mismatch**
**Location**: `l0_module_embedding.py:164-183`
**Problem**: Forward hooks expect attention output format from transformers but BGE M3 uses XLM-RoBERTa which may have different output structure.

```python
# Problematic hook expecting wrong output format
def hook(module, input, output):
    if hasattr(self, 'current_masks') and 'head_z' in self.current_masks:
        head_mask = self.current_masks['head_z'][layer_idx]
        if output[0] is not None:
            output = (output[0] * head_mask.view(1, -1, 1, 1), *output[1:])
    return output
```

#### 2. **Mask Output Shape Inconsistency**  
**Location**: `l0_module_embedding.py:302-317`
**Problem**: Head mask shape `[layers, 1, heads, 1]` doesn't match attention computation requirements.

```python
# Mismatch: mask_output_shape vs actual usage
mask_output_shape = [self.base_model_info.num_layers, 1, self.base_model_info.num_attention_heads, 1]
# But hooks expect: [num_layers, num_heads] -> view(1, -1, 1, 1)
```

#### 3. **Batch Dimension Confusion**
**Location**: `composer_bge_m3.py:132-138`
**Problem**: Loss computation assumes incorrect batch reshaping.

```python
# Assumes batch_size * 2 can be reshaped to [batch_size, 2, -1]
# But this fails when batch_size * 2 is not evenly divisible
dense_emb_reshaped = dense_emb.view(batch_size, 2, -1)
```

#### 4. **Hidden Dimension Pruning Logic Error**
**Location**: `bge_m3_backbone.py:87-91`
**Problem**: Index selection may create dimension mismatch in attention computation.

```python
if hidden_z is not None:
    remaining_index = torch.where(~hidden_z.eq(0))[0]
    compressed_hidden_states = torch.index_select(hidden_states, dim=-1, index=remaining_index)
# This changes embedding dimension but subsequent layers expect original size
```

#### 5. **L0 Mask Deterministic Function Shape Issue**
**Location**: `l0_module_embedding.py:79-88`
**Problem**: Multi-dimensional mask reshaping logic is fragile.

```python
# May fail when z_loga.ndim != 1 and reshaping assumptions break
z_loga = self.z_loga.reshape(-1, self.z_loga.shape[-1])
z = torch.stack(z).reshape(*self.mask_output_shape)
```

### **Root Cause Analysis**

1. **Architecture Mismatch**: L0 module designed for generic transformer but BGE M3 has specific XLM-RoBERTa structure
2. **Inconsistent Shape Conventions**: Mixing 2D/4D mask shapes without clear dimension mapping
3. **Hook Implementation**: Forward hooks don't account for actual model output structure
4. **Batch Processing**: Data loader creates concatenated batches but loss assumes paired structure




The core issue is that the L0 pruning module assumes generic transformer architecture but BGE M3 has specific shape requirements that aren't properly handled in the current implementation.