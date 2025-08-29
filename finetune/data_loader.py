import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Optional

class STSDataset(Dataset):
    """STS benchmark dataset for finetuning using HuggingFace datasets"""
    
    def __init__(self, hf_dataset, tokenizer_path: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.data = self._process_hf_dataset(hf_dataset)
    
    def _process_hf_dataset(self, hf_dataset) -> List[Dict]:
        """Process HuggingFace dataset into our format"""
        data = []
        for item in hf_dataset:
            try:
                # STS-B format: label (0-5 scale), sentence1, sentence2
                score = float(item['label'])
                sent1 = item['sentence1'].strip()
                sent2 = item['sentence2'].strip()
                
                if sent1 and sent2:  # Only include non-empty sentences
                    data.append({
                        'sentence1': sent1,
                        'sentence2': sent2,
                        'score': score
                    })
            except (ValueError, KeyError, TypeError):
                continue
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize both sentences
        encoded1 = self.tokenizer(
            item['sentence1'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoded2 = self.tokenizer(
            item['sentence2'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Interleave format: [sent1, sent2] for batch processing
        input_ids = torch.cat([encoded1['input_ids'], encoded2['input_ids']], dim=0)
        attention_mask = torch.cat([encoded1['attention_mask'], encoded2['attention_mask']], dim=0)
        
        return {
            'input_ids': input_ids.squeeze(1),  # Remove extra dimension
            'attention_mask': attention_mask.squeeze(1),
            'similarity_scores': torch.tensor(item['score'], dtype=torch.float)
        }

def create_sts_dataloader(
    split: str,
    tokenizer_path: str,
    batch_size: int = 16, 
    max_length: int = 512, 
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """Create STS dataloader for finetuning using HuggingFace datasets"""
    
    # Load STS-B dataset from GLUE benchmark
    hf_dataset = load_dataset("glue", "stsb", split=split)
    
    if len(hf_dataset) == 0:
        raise ValueError(f"No data found for split: {split}")
    
    dataset = STSDataset(hf_dataset, tokenizer_path, max_length)
    
    if len(dataset) == 0:
        raise ValueError(f"No valid data found after processing split: {split}")
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def load_sts_dataset():
    """Load and return the complete STS-B dataset from HuggingFace"""
    try:
        dataset = load_dataset("glue", "stsb")
        print(f"✅ Loaded STS-B dataset from HuggingFace")
        print(f"   Train samples: {len(dataset['train'])}")
        print(f"   Validation samples: {len(dataset['validation'])}")
        print(f"   Test samples: {len(dataset['test'])}")
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load STS-B dataset: {e}")

def get_sts_info():
    """Get information about the STS-B dataset"""
    dataset = load_dataset("glue", "stsb")
    
    # Sample a few examples
    sample = dataset['train'][0]
    
    info = {
        "dataset_name": "STS-B (Semantic Textual Similarity Benchmark)",
        "task_type": "Regression (0-5 similarity score)",
        "splits": list(dataset.keys()),
        "train_size": len(dataset['train']),
        "validation_size": len(dataset['validation']),
        "test_size": len(dataset['test']),
        "features": list(sample.keys()),
        "example": {
            "sentence1": sample['sentence1'],
            "sentence2": sample['sentence2'], 
            "label": sample['label']
        }
    }
    
    return info
