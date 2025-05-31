"""
Evaluation utilities for Hindi tokenizer.
"""

import json
import time
from collections import defaultdict
from tqdm import tqdm


class TokenizerEvaluator:
    """
    Evaluator for Hindi tokenizer performance and quality.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the evaluator.
        
        Args:
            tokenizer: The tokenizer to evaluate.
        """
        self.tokenizer = tokenizer
        self.results = {}
    
    def evaluate_speed(self, texts, batch_size=1):
        """
        Evaluate tokenization and detokenization speed.
        
        Args:
            texts (list): List of Hindi texts to evaluate.
            batch_size (int): Batch size for processing.
            
        Returns:
            dict: Speed metrics.
        """
        # Tokenization speed
        start_time = time.time()
        token_counts = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                tokens = self.tokenizer.encode(text)
                token_counts.append(len(tokens))
        
        tokenization_time = time.time() - start_time
        
        # Detokenization speed
        start_time = time.time()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                tokens = self.tokenizer.encode(text)
                _ = self.tokenizer.decode(tokens)
        
        detokenization_time = time.time() - start_time
        
        total_chars = sum(len(text) for text in texts)
        total_tokens = sum(token_counts)
        
        metrics = {
            "num_samples": len(texts),
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "chars_per_token": total_chars / total_tokens if total_tokens > 0 else 0,
            "tokenization_time": tokenization_time,
            "detokenization_time": detokenization_time,
            "tokens_per_second": total_tokens / tokenization_time if tokenization_time > 0 else 0,
            "chars_per_second": total_chars / tokenization_time if tokenization_time > 0 else 0
        }
        
        self.results["speed"] = metrics
        return metrics
    
    def evaluate_reconstruction(self, texts):
        """
        Evaluate text reconstruction quality.
        
        Args:
            texts (list): List of Hindi texts to evaluate.
            
        Returns:
            dict: Reconstruction metrics.
        """
        perfect_matches = 0
        char_errors = 0
        total_chars = 0
        
        for text in tqdm(texts, desc="Evaluating reconstruction"):
            total_chars += len(text)
            tokens = self.tokenizer.encode(text)
            reconstructed = self.tokenizer.decode(tokens)
            
            if text == reconstructed:
                perfect_matches += 1
            else:
                # Count character differences
                max_len = max(len(text), len(reconstructed))
                for i in range(max_len):
                    if i >= len(text) or i >= len(reconstructed) or text[i] != reconstructed[i]:
                        char_errors += 1
        
        metrics = {
            "num_samples": len(texts),
            "perfect_match_ratio": perfect_matches / len(texts) if len(texts) > 0 else 0,
            "character_error_rate": char_errors / total_chars if total_chars > 0 else 0
        }
        
        self.results["reconstruction"] = metrics
        return metrics
    
    def save_results(self, output_file):
        """
        Save evaluation results to a JSON file.
        
        Args:
            output_file (str): Path to the output file.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
