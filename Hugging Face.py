```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
import numpy as np

class HuggingFaceNLIClient:
    """
    Love-OS NLI Backend for Zero-Time $\infty/\infty$ Detection.
    Recommended model: 'cross-encoder/nli-deberta-v3-small'
    """
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Label mapping (adjust based on specific HF model)
        # Typically: 0=Contradiction, 1=Entailment, 2=Neutral
        self.contradiction_idx = 0 

    def batch_predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
            
        inputs = self.tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
        # Extract contradiction probabilities
        contradiction_probs = probs[:, self.contradiction_idx].cpu().numpy().tolist()
        return contradiction_probs
