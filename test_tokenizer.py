#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for evaluating the Hindi tokenizer.
"""

from hindi_tokenizer.core import HindiTokenizer
from hindi_tokenizer.evaluator import TokenizerEvaluator
import json

def main():
    # Initialize the tokenizer with the trained model
    model_path = 'models/hindi_tokenizer_32k.model'
    tokenizer = HindiTokenizer(model_path=model_path)
    
    # Read sample Hindi text
    with open('data/sample_hindi.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # Initialize the evaluator
    evaluator = TokenizerEvaluator(tokenizer)
    
    # Evaluate reconstruction quality
    print("Evaluating reconstruction quality...")
    recon_results = evaluator.evaluate_reconstruction(texts)
    
    # Evaluate tokenization speed
    print("\nEvaluating tokenization speed...")
    speed_results = evaluator.evaluate_speed(texts)
    
    # Save results
    evaluator.save_results('test_results/evaluation_results.json')
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Perfect match ratio: {recon_results['perfect_match_ratio']*100:.2f}%")
    print(f"Character error rate: {recon_results['character_error_rate']*100:.4f}%")
    print(f"Tokenization speed: {speed_results['tokens_per_second']:.2f} tokens/second")
    print(f"Characters per token: {speed_results['chars_per_token']:.4f}")
    
    # Test tokenization of a sample sentence
    sample = "हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है।"
    tokens = tokenizer.encode(sample)
    pieces = tokenizer.encode_as_pieces(sample)
    
    print("\nSample Tokenization:")
    print(f"Original: {sample}")
    print(f"Token IDs: {tokens}")
    print(f"Token Pieces: {pieces}")
    print(f"Reconstructed: {tokenizer.decode(tokens)}")

if __name__ == "__main__":
    main()
