from typing import List, Dict, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class LLM:
    def __init__(self):
        self.model_name = "google/flan-t5-xl"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name
        )
        # Move the model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
            
    def generate(self, 
                 prompt: str,
                 max_length: int = 512,
                 temperature: float = 0.7,
                 verbose: bool = True) -> str:

        if verbose:
            print("\n=== PROMPT ===\n")
            print(prompt)
                
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        if verbose:
            print("\n=== RESPONSE ===\n")
            print(response)
                
        return response

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLM()
    return _llm_instance

_llm_instance = None
llm = get_llm()

__all__ = ['LLM', 'get_llm', 'llm']
