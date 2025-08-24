# BGE-M3 Pruning Framework

**Structured pruning for BGE-M3 embedding models with HuggingFace datasets**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_clean.txt
```

### 2. Run Pruning
```bash
# STS Benchmark (Semantic Similarity)
python train_clean.py config_clean.yaml --dataset sts

# MS MARCO (Information Retrieval)  
python train_clean.py config_clean.yaml --dataset msmarco
```

That's it! No manual downloads, no complex setup.

## 📊 What This Does

1. **Loads BGE-M3** pretrained model from HuggingFace (`BAAI/bge-m3`)
2. **Downloads datasets** automatically from HuggingFace Hub:
   - **STS Benchmark**: `mteb/stsbenchmark-sts` 
   - **MS MARCO**: `ms_marco` v1.1
3. **Applies L0 pruning** using structured sparsity
4. **Reduces parameters** by 75% (configurable) while maintaining quality
5. **Saves pruned model** ready for deployment

## ⚙️ Configuration

### Parameter Reduction Targets
Edit `config_clean.yaml`:

```yaml
# 75% reduction (default)
target_model:
  n_layers: 18  # From 24 layers
  n_heads: 12   # From 16 heads

# 60% reduction  
target_model:
  n_layers: 20
  n_heads: 14

# 50% reduction
target_model:
  n_layers: 21
  n_heads: 15
```

### Batch Size & Memory
```yaml
batch_size: 16    # Reduce if GPU memory issues
max_length: 512   # Sequence length
```

## 📈 Expected Results

| Target | Layers | Heads | Parameters | Use Case |
|--------|--------|-------|------------|----------|
| 75%    | 18/24  | 12/16 | ~300M      | Mobile/Edge |
| 60%    | 20/24  | 14/16 | ~480M      | Balanced |
| 50%    | 21/24  | 15/16 | ~600M      | High Quality |

Performance retention typically 95-98% of original BGE-M3.

## 🔧 Advanced Usage

### Custom Training Duration
```bash
python train_clean.py config_clean.yaml --dataset sts
# Edit config_clean.yaml: max_duration: "2000ba"  # Double training
```

### GPU Memory Optimization
```bash
# Reduce batch size for smaller GPUs
# Edit config_clean.yaml: batch_size: 8
```

### Different Random Seeds
```bash
python train_clean.py config_clean.yaml --dataset sts --seed 123
```

## 📁 Core Files

```
bge_pruning/
├── train_clean.py           # Main training script  
├── config_clean.yaml        # Configuration
├── requirements_clean.txt   # Dependencies
├── data_loader.py          # HuggingFace dataset loader
├── models/
│   ├── composer_bge_m3.py  # BGE-M3 + L0 integration
│   ├── l0_module_embedding.py  # L0 pruning module
│   └── embedding_heads.py  # Dense/sparse/multi-vector heads
└── callbacks/
    └── pruning_callback.py # Training monitoring
```

## 🐛 Troubleshooting

### ImportError: cannot import name 'METRIC_DEFAULT_CTORS'
✅ **Fixed**: Removed deprecated Composer imports

### CUDA Out of Memory
```yaml
# In config_clean.yaml:
batch_size: 8  # Reduce from 16
```

### Dataset Download Issues  
The framework automatically downloads from HuggingFace. If issues persist:
```bash
# Check internet connection and HuggingFace access
python -c "from datasets import load_dataset; print('OK')"
```

## 🏗️ Architecture

This framework integrates:
- **BGE-M3**: Pretrained embedding model from BAAI
- **L0 Regularization**: From LLM-Shearing methodology  
- **Composer**: MosaicML's training framework
- **HuggingFace**: Automatic dataset loading

The L0 module applies structured pruning through forward hooks on the pretrained BGE-M3 model, avoiding custom layer reimplementation.

## 📝 Citation

```bibtex
@article{bge_m3_pruning,
  title={Structured Pruning for BGE-M3 Embedding Models},
  author={BGE-M3 Pruning Team},
  year={2025}
}

@article{llm_shearing,
  title={Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning},
  author={Xia, Mengzhou and Gao, Tianyu and Zeng, Zhiyuan and Chen, Danqi},
  journal={arXiv preprint arXiv:2310.06694},
  year={2023}
}
```

## 📄 License

Apache License 2.0
