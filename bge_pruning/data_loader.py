"""
Unified HuggingFace Dataset Loader for BGE-M3 Pruning
Clean, minimal implementation using datasets library
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
# Import HuggingFace datasets explicitly to avoid local module conflict
import datasets
from typing import Dict, Any, Optional

class BGEDataset(torch.utils.data.Dataset):
    """Unified dataset for BGE-M3 training with HuggingFace datasets"""
    
    def __init__(self, dataset_name: str, split: str = "train", 
                 tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Load dataset from HuggingFace
        if dataset_name == "sts":
            self.dataset = datasets.load_dataset("mteb/stsbenchmark-sts", split=split)
            self.task_type = "similarity"
        elif dataset_name == "msmarco":
            self.dataset = datasets.load_dataset("ms_marco", "v1.1", split=split)
            self.task_type = "retrieval"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        if self.task_type == "similarity":
            # STS format: sentence1, sentence2, score - process separately
            inputs1 = self.tokenizer(
                item['sentence1'], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            inputs2 = self.tokenizer(
                item['sentence2'], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            
            return {
                'input_ids_1': inputs1['input_ids'],  # Keep [1, seq_length] 
                'attention_mask_1': inputs1['attention_mask'],  # Keep [1, seq_length]
                'input_ids_2': inputs2['input_ids'],  # Keep [1, seq_length]
                'attention_mask_2': inputs2['attention_mask'],  # Keep [1, seq_length]
                'similarity_scores': torch.tensor(item['score'], dtype=torch.float),
                'task_type': 'sts'
            }
        
        else:  # retrieval
            # MS MARCO format: query, passage - process separately  
            query_inputs = self.tokenizer(
                item['query'], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            passage_inputs = self.tokenizer(
                item['passage'], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            
            return {
                'input_ids_1': query_inputs['input_ids'],  # Keep [1, seq_length]
                'attention_mask_1': query_inputs['attention_mask'],  # Keep [1, seq_length]
                'input_ids_2': passage_inputs['input_ids'],  # Keep [1, seq_length]
                'attention_mask_2': passage_inputs['attention_mask'],  # Keep [1, seq_length]
                'labels': torch.tensor(1, dtype=torch.long),  # Positive pair
                'task_type': 'retrieval'
            }

def create_dataloader(dataset_name: str, split: str = "train", batch_size: int = 16, 
                     tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512,
                     num_workers: int = 2) -> DataLoader:
    """Create DataLoader for BGE-M3 training"""
    dataset = BGEDataset(dataset_name, split, tokenizer_name, max_length)
    
    def collate_fn(batch):
        """Custom collate function - clean approach maintaining 2D tensors"""
        # Each item has tensors of shape [1, seq_length], concatenate them
        input_ids_1 = torch.cat([item['input_ids_1'] for item in batch], dim=0)  # [batch_size, seq_length]
        attention_mask_1 = torch.cat([item['attention_mask_1'] for item in batch], dim=0)  # [batch_size, seq_length]
        input_ids_2 = torch.cat([item['input_ids_2'] for item in batch], dim=0)  # [batch_size, seq_length]
        attention_mask_2 = torch.cat([item['attention_mask_2'] for item in batch], dim=0)  # [batch_size, seq_length]
        
        # Concatenate to create [batch_size * 2, seq_length] for transformer
        input_ids = torch.cat([input_ids_1, input_ids_2], dim=0)
        attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=0)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        # Add task-specific fields - expand to match input_ids dimension
        if batch[0]['task_type'] == 'sts':
            similarity_scores = torch.stack([item['similarity_scores'] for item in batch])
            # Expand to [batch_size * 2] to match input_ids first dimension
            similarity_scores_expanded = similarity_scores.repeat_interleave(2)
            result['similarity_scores'] = similarity_scores_expanded
        else:
            labels = torch.stack([item['labels'] for item in batch])
            # Expand to [batch_size * 2] to match input_ids first dimension  
            labels_expanded = labels.repeat_interleave(2)
            result['labels'] = labels_expanded
        
        return result
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
