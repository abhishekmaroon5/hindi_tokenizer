#!/usr/bin/env python3
"""
Script to prepare Hindi corpus data for tokenizer training.
"""

import os
import sys
import argparse
import logging
import random
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hindi_tokenizer.preprocessor import HindiPreprocessor
from hindi_tokenizer.utils import contains_hindi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_file(input_file, output_file, min_chars=10, max_chars=1000, 
                 min_hindi_ratio=0.5, remove_latin=False, remove_numbers=False):
    """
    Process a single file and write cleaned lines to the output file.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        min_chars (int): Minimum number of characters per line.
        max_chars (int): Maximum number of characters per line.
        min_hindi_ratio (float): Minimum ratio of Hindi characters.
        remove_latin (bool): Whether to remove Latin script characters.
        remove_numbers (bool): Whether to remove numbers.
        
    Returns:
        tuple: (processed_lines, total_lines)
    """
    preprocessor = HindiPreprocessor(
        remove_latin=remove_latin,
        normalize=True,
        remove_numbers=remove_numbers
    )
    
    processed_lines = 0
    total_lines = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            for line in tqdm(fin, desc=f"Processing {os.path.basename(input_file)}", 
                            unit="lines", leave=False):
                total_lines += 1
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip lines that are too short or too long
                if len(line) < min_chars or len(line) > max_chars:
                    continue
                
                # Check if the line contains enough Hindi characters
                hindi_chars = sum(1 for c in line if '\u0900' <= c <= '\u097F')
                if hindi_chars / len(line) < min_hindi_ratio:
                    continue
                
                # Preprocess the line
                processed_line = preprocessor.preprocess(line)
                if processed_line:
                    fout.write(processed_line + '\n')
                    processed_lines += 1
    
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")
        return 0, 0
    
    return processed_lines, total_lines


def process_directory(input_dir, output_file, file_pattern="*.txt", **kwargs):
    """
    Process all matching files in a directory and write to a single output file.
    
    Args:
        input_dir (str): Path to the input directory.
        output_file (str): Path to the output file.
        file_pattern (str): Pattern to match files.
        **kwargs: Additional arguments for process_file.
        
    Returns:
        tuple: (processed_lines, total_lines)
    """
    input_dir = Path(input_dir)
    input_files = list(input_dir.glob(file_pattern))
    
    if not input_files:
        logger.warning(f"No files matching {file_pattern} found in {input_dir}")
        return 0, 0
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Create a temporary directory for processed files
    temp_dir = Path(output_file).parent / "temp_processed"
    temp_dir.mkdir(exist_ok=True)
    
    processed_lines = 0
    total_lines = 0
    
    # Process files in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for input_file in input_files:
            temp_output = temp_dir / f"{input_file.stem}_processed.txt"
            futures.append(
                executor.submit(process_file, str(input_file), str(temp_output), **kwargs)
            )
        
        # Collect results
        for future in tqdm(futures, desc="Processing files", unit="file"):
            proc_lines, tot_lines = future.result()
            processed_lines += proc_lines
            total_lines += tot_lines
    
    # Combine all processed files into one
    with open(output_file, 'w', encoding='utf-8') as fout:
        for temp_file in temp_dir.glob("*_processed.txt"):
            with open(temp_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)
    
    # Clean up temporary files
    for temp_file in temp_dir.glob("*_processed.txt"):
        temp_file.unlink()
    temp_dir.rmdir()
    
    return processed_lines, total_lines


def sample_lines(input_file, output_file, num_lines, seed=42):
    """
    Sample a specific number of lines from a file.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        num_lines (int): Number of lines to sample.
        seed (int): Random seed.
        
    Returns:
        int: Number of lines sampled.
    """
    # Count total lines in the file
    total_lines = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    if total_lines <= num_lines:
        logger.warning(f"File contains fewer lines ({total_lines}) than requested ({num_lines})")
        return total_lines
    
    # Sample lines
    random.seed(seed)
    sample_indices = set(random.sample(range(total_lines), num_lines))
    
    sampled_lines = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(tqdm(fin, desc="Sampling lines", total=total_lines)):
            if i in sample_indices:
                fout.write(line)
                sampled_lines += 1
    
    return sampled_lines


def main():
    parser = argparse.ArgumentParser(description="Prepare Hindi corpus for tokenizer training")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output file")
    parser.add_argument("--is-dir", action="store_true", help="Input is a directory")
    parser.add_argument("--file-pattern", default="*.txt", help="File pattern when input is a directory")
    parser.add_argument("--min-chars", type=int, default=10, help="Minimum characters per line")
    parser.add_argument("--max-chars", type=int, default=1000, help="Maximum characters per line")
    parser.add_argument("--min-hindi-ratio", type=float, default=0.5, help="Minimum ratio of Hindi characters")
    parser.add_argument("--remove-latin", action="store_true", help="Remove Latin script characters")
    parser.add_argument("--remove-numbers", action="store_true", help="Remove numbers")
    parser.add_argument("--sample", type=int, help="Sample N lines from the output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Process input
    if args.is_dir:
        logger.info(f"Processing directory: {args.input}")
        processed_lines, total_lines = process_directory(
            args.input, args.output, 
            file_pattern=args.file_pattern,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            min_hindi_ratio=args.min_hindi_ratio,
            remove_latin=args.remove_latin,
            remove_numbers=args.remove_numbers
        )
    else:
        logger.info(f"Processing file: {args.input}")
        processed_lines, total_lines = process_file(
            args.input, args.output,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            min_hindi_ratio=args.min_hindi_ratio,
            remove_latin=args.remove_latin,
            remove_numbers=args.remove_numbers
        )
    
    logger.info(f"Processed {processed_lines} out of {total_lines} lines")
    
    # Sample if requested
    if args.sample and args.sample > 0:
        temp_output = args.output + ".full"
        os.rename(args.output, temp_output)
        
        sampled_lines = sample_lines(temp_output, args.output, args.sample, args.seed)
        logger.info(f"Sampled {sampled_lines} lines")
        
        # Clean up
        os.remove(temp_output)
    
    logger.info(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
