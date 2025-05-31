#!/usr/bin/env python3
"""
Script to train the Hindi tokenizer on a corpus.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hindi_tokenizer.trainer import TokenizerTrainer
from hindi_tokenizer.core import HindiTokenizer
from hindi_tokenizer.evaluator import TokenizerEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Hindi tokenizer")
    parser.add_argument("--input", required=True, help="Input corpus file")
    parser.add_argument("--model-dir", default="../models", help="Directory to save the model")
    parser.add_argument("--model-name", default="hindi_tokenizer", help="Name of the model")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--character-coverage", type=float, default=0.9995, help="Character coverage")
    parser.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram", help="Model type")
    parser.add_argument("--input-sentence-size", type=int, default=1000000, 
                        help="Number of sentences to use for training")
    parser.add_argument("--max-sentence-length", type=int, default=4192, help="Maximum sentence length")
    parser.add_argument("--test-file", help="Test file for evaluation")
    parser.add_argument("--test-sample-size", type=int, default=1000, 
                        help="Number of sentences to sample for testing")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create model directory
    model_dir = os.path.abspath(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Full path for the model
    model_prefix = os.path.join(model_dir, args.model_name)
    
    # Train the tokenizer
    logger.info(f"Training tokenizer with vocab size {args.vocab_size}")
    start_time = time.time()
    
    trainer = TokenizerTrainer()
    success = trainer.train(
        input_file=args.input,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        input_sentence_size=args.input_sentence_size,
        max_sentence_length=args.max_sentence_length
    )
    
    training_time = time.time() - start_time
    
    if not success:
        logger.error("Training failed")
        sys.exit(1)
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Model saved to {model_prefix}.model")
    
    # Evaluate the model if test file is provided
    if args.test_file and os.path.exists(args.test_file):
        logger.info(f"Evaluating model on {args.test_file}")
        
        # Load the trained model
        tokenizer = HindiTokenizer(model_path=f"{model_prefix}.model")
        evaluator = TokenizerEvaluator(tokenizer)
        
        # Load test data
        test_lines = []
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_lines.append(line)
        
        # Sample test data if needed
        if args.test_sample_size and args.test_sample_size < len(test_lines):
            import random
            random.seed(42)
            test_lines = random.sample(test_lines, args.test_sample_size)
        
        # Evaluate
        logger.info(f"Evaluating on {len(test_lines)} test sentences")
        speed_metrics = evaluator.evaluate_speed(test_lines)
        reconstruction_metrics = evaluator.evaluate_reconstruction(test_lines)
        
        # Save evaluation results
        eval_output = f"{model_prefix}_evaluation.json"
        evaluator.save_results(eval_output)
        logger.info(f"Evaluation results saved to {eval_output}")
        
        # Print summary
        logger.info("Evaluation summary:")
        logger.info(f"  Tokens per second: {speed_metrics['tokens_per_second']:.2f}")
        logger.info(f"  Characters per token: {speed_metrics['chars_per_token']:.2f}")
        logger.info(f"  Perfect reconstruction ratio: {reconstruction_metrics['perfect_match_ratio']:.4f}")
        logger.info(f"  Character error rate: {reconstruction_metrics['character_error_rate']:.6f}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
