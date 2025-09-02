
### What this project does
- **Goal**: Train and prune the BGE-M3 embedding model (structured pruning: layers, heads, intermediate) and export it in Hugging Face format.
- **Core pieces**: `ComposerBGEM3` wraps a custom backbone `MaskedBGEM3Backbone` with L0 masks; after training, masks are made deterministic, parameters are physically pruned, then the model is saved for HF.

### train_clean.py pipeline (relevant part)
- After training, it calls the model’s HF export:
```191:194:bge_pruning/train_clean.py
    # Save pruned model in HuggingFace format
    hf_save_path = cfg.get('save_folder', 'experiments/production') + '_hf'
    model.save_pruned_hf_model(hf_save_path)
    print(f"💾 Model ready for deployment")
```

### HF export chain and where the issue occurs
- `ComposerBGEM3.save_pruned_hf_model` applies masks, prunes parameters, validates, then calls the exporter:
```273:295:bge_pruning/models/composer_bge_m3.py
        from utils.hf_export import save_backbone_as_hf_model
        ...
        # Actually remove pruned parameters
        self.prune_params(zs)
        self._validate_pruned_model()
        
        # Save backbone as proper HuggingFace model
        print(f"\n💾 Saving pruned model to {save_path}")
        base_model_name = tokenizer_name or getattr(self, 'base_model_name', 'BAAI/bge-m3')
        save_backbone_as_hf_model(self.backbone, save_path, base_model_name)
```

- The exporter just delegates to the backbone’s own `save_pretrained` (no key remapping):
```96:129:bge_pruning/utils/hf_export.py
def save_backbone_as_hf_model(backbone, save_path, base_model_name="BAAI/bge-m3"):
    ...
    # Fix backbone config to match actual pruned dimensions
    ...
    # Save backbone with corrected config
    backbone.save_pretrained(save_path)
    ...
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(save_path)
```

- The backbone’s `save_pretrained` writes the raw state dict as-is and a minimal config:
```333:366:bge_pruning/models/bge_m3_backbone.py
    def save_pretrained(self, save_path: str):
        ...
        model_path = os.path.join(save_path, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        ...
        config_dict = {
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "intermediate_size": self.config.intermediate_size,
            "vocab_size": actual_vocab_size,
            "max_position_embeddings": actual_max_pos,
            "type_vocab_size": actual_type_vocab,
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaModel"],
        }
        ...
```

### Why HF loads with “randomly initialized missing components”
- The saved `pytorch_model.bin` uses this project’s custom module names that don’t match HF’s XLM-R keys. For example, keys look like:
  - `encoder.layer.N.attention.query.weight`, `...key.weight`, `...value.weight`
  - But HF `XLMRobertaModel` expects: `encoder.layer.N.attention.self.query.weight`, etc.
- There is also no `attention.output.dense.*` in this backbone (the code has no `attention.output` submodule), so those weights will always be missing when HF instantiates `XLMRobertaModel`.

- A conversion helper exists but is not used during saving:
```74:93:bge_pruning/utils/hf_export.py
def convert_backbone_to_hf_state_dict(backbone_state_dict):
    ...
    if ".attention.query." in key:
        new_key = key.replace(".attention.query.", ".attention.self.query.")
    elif ".attention.key." in key:
        new_key = key.replace(".attention.key.", ".attention.self.key.")
    elif ".attention.value." in key:
        new_key = key.replace(".attention.value.", ".attention.self.value.")
    elif ".attention.out_proj." in key:
        new_key = key.replace(".attention.out_proj.", ".attention.output.dense.")
```
- And a more complete config builder exists but is also unused:
```23:71:bge_pruning/utils/hf_export.py
def create_hf_config_from_backbone(backbone):
    ...
    return {
        "architectures": ["XLMRobertaModel"],
        "model_type": "xlm-roberta",
        "num_attention_heads": actual_heads,
        "num_hidden_layers": actual_layers,
        "intermediate_size": actual_intermediate,
        ...
    }
```

- Because the exporter saves the raw state dict without key remapping and the config declares `architectures: ["XLMRobertaModel"]`, `AutoModel.from_pretrained` builds an `XLMRobertaModel` and will:
  - Fail to find `attention.self.*` and `attention.output.*` weights in the bin.
  - Randomly initialize those missing parameters, exactly as you observed.

### Secondary risks worth noting
- After head pruning, per-layer `attention_head_size` is not recomputed; the code relies on keeping head_dim constant and ensuring hidden_size % new_heads == 0. While this can work internally, it’s another sign the saved structure is not a drop-in XLM-R.
- The saved config is minimal; defaults may differ from the original model (not the primary cause of random init, but can affect behavior).

### Verdict (FIXED)
- **Original Issue**: The HF export saved raw state dict with non-HF key names and missing `attention.output.dense` layers, causing random initialization on load.

- **Solution Applied**:
  1. **Fixed `MaskedBGEM3Backbone.save_pretrained`** to use proper conversion functions
  2. **Enhanced `convert_backbone_to_hf_state_dict`** to:
     - Map `.attention.query/key/value.` → `.attention.self.query/key/value.`
     - Create identity matrices for missing `attention.output.dense` layers (since backbone directly concatenates heads)
  3. **Used `create_hf_config_from_backbone`** for proper config generation
  4. **Simplified `save_backbone_as_hf_model`** to avoid duplication

- **Key Fix**: The backbone lacks learned output projections in attention layers, so we insert identity matrices as `attention.output.dense` weights, preserving the original behavior while satisfying HF's XLM-RoBERTa structure.
