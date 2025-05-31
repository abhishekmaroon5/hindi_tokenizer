#!/usr/bin/env python3
"""
Script to download Hindi corpus data for training the tokenizer.
"""

import os
import sys
import argparse
import logging
import requests
import tarfile
import zipfile
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Available datasets
DATASETS = {
    "iitb": {
        "name": "IITB Hindi-English Parallel Corpus",
        "url": "http://www.cfilt.iitb.ac.in/~moses/iitb_en_hi_parallel/iitb_corpus_download/monolingual.hi.tgz",
        "description": "Monolingual Hindi corpus from IIT Bombay",
        "file_type": "tgz",
    },
    "oscar": {
        "name": "OSCAR Hindi Corpus",
        "url": "https://huggingface.co/datasets/oscar-corpus/OSCAR-2301",
        "description": "Hindi subset of the OSCAR corpus (requires Hugging Face datasets library)",
        "file_type": "hf",
    },
    "wikipedia": {
        "name": "Hindi Wikipedia",
        "url": "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2",
        "description": "Latest Hindi Wikipedia dump (requires wikiextractor)",
        "file_type": "bz2",
    },
    "cc100": {
        "name": "CC-100 Hindi",
        "url": "http://data.statmt.org/cc-100/hi.txt.xz",
        "description": "Hindi subset of the CC-100 corpus",
        "file_type": "xz",
    },
}


def download_file(url, output_path):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url (str): URL to download from.
        output_path (str): Path to save the downloaded file.
        
    Returns:
        bool: True if download was successful.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(output_path, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(output_path)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_archive(archive_path, output_dir, file_type):
    """
    Extract an archive file.
    
    Args:
        archive_path (str): Path to the archive file.
        output_dir (str): Directory to extract to.
        file_type (str): Type of archive ('tgz', 'zip', 'bz2', 'xz').
        
    Returns:
        bool: True if extraction was successful.
    """
    try:
        if file_type == 'tgz':
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=output_dir)
        elif file_type == 'zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif file_type in ['bz2', 'xz']:
            logger.info(f"For {file_type} files, use the appropriate tool to extract them.")
            return False
        
        logger.info(f"Extracted {archive_path} to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_hf_dataset(dataset_name, output_dir):
    """
    Download a dataset from Hugging Face.
    
    Args:
        dataset_name (str): Name of the dataset.
        output_dir (str): Directory to save the dataset.
        
    Returns:
        bool: True if download was successful.
    """
    try:
        from datasets import load_dataset
        
        logger.info(f"Downloading {dataset_name} from Hugging Face...")
        dataset = load_dataset(dataset_name, "hi", split="train")
        
        output_file = os.path.join(output_dir, f"{dataset_name.replace('/', '_')}_hi.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc=f"Writing {os.path.basename(output_file)}"):
                f.write(item['text'] + '\n')
        
        logger.info(f"Dataset saved to {output_file}")
        return True
    except ImportError:
        logger.error("Hugging Face datasets library not installed. Run 'pip install datasets'")
        return False
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Hindi corpus data")
    parser.add_argument(
        "--dataset", 
        choices=list(DATASETS.keys()), 
        default="iitb",
        help="Dataset to download"
    )
    parser.add_argument(
        "--output-dir", 
        default="../data",
        help="Directory to save the downloaded data"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available datasets and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available datasets:")
        for key, dataset in DATASETS.items():
            print(f"  {key}: {dataset['name']}")
            print(f"      {dataset['description']}")
        return
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = DATASETS[args.dataset]
    logger.info(f"Downloading {dataset['name']}...")
    
    if dataset['file_type'] == 'hf':
        success = download_hf_dataset(dataset['url'], output_dir)
    else:
        # Download the file
        file_name = os.path.basename(dataset['url'])
        output_path = os.path.join(output_dir, file_name)
        
        success = download_file(dataset['url'], output_path)
        
        # Extract if it's an archive
        if success and dataset['file_type'] in ['tgz', 'zip', 'bz2', 'xz']:
            success = extract_archive(output_path, output_dir, dataset['file_type'])
    
    if success:
        logger.info(f"Successfully downloaded {dataset['name']}")
    else:
        logger.error(f"Failed to download {dataset['name']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
