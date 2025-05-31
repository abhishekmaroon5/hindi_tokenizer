"""
Training utilities for Hindi tokenizer.
"""

import os
import logging
from tqdm import tqdm
import sentencepiece as spm
from hindi_tokenizer.preprocessor import HindiPreprocessor


class TokenizerTrainer:
    """
    Trainer for Hindi SentencePiece tokenizer.
    """
    
    def __init__(self, preprocessor=None):
        """
        Initialize the trainer.
        
        Args:
            preprocessor (HindiPreprocessor, optional): Preprocessor for Hindi text.
                If None, a default preprocessor will be used.
        """
        self.preprocessor = preprocessor or HindiPreprocessor()
        self.logger = logging.getLogger(__name__)
    
    def prepare_training_data(self, input_files, output_file, sample_size=None):
        """
        Prepare training data from multiple input files.
        
        Args:
            input_files (list): List of input file paths.
            output_file (str): Path to the output file.
            sample_size (int, optional): Maximum number of lines to include.
                If None, all lines will be included.
                
        Returns:
            int: Number of lines in the output file.
        """
        line_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as fout:
            for input_file in input_files:
                self.logger.info(f"Processing {input_file}")
                with open(input_file, 'r', encoding='utf-8') as fin:
                    for line in tqdm(fin, desc=f"Processing {os.path.basename(input_file)}"):
                        line = line.strip()
                        if not line:
                            continue
                        
                        processed_line = self.preprocessor.preprocess(line)
                        if processed_line:
                            fout.write(processed_line + '\n')
                            line_count += 1
                            
                            if sample_size and line_count >= sample_size:
                                break
                
                if sample_size and line_count >= sample_size:
                    break
        
        self.logger.info(f"Created training data with {line_count} lines")
        return line_count
    
    def train(self, input_file, model_prefix, vocab_size=32000, character_coverage=0.9995,
              model_type="unigram", input_sentence_size=1000000, max_sentence_length=4192,
              pad_id=0, unk_id=1, bos_id=2, eos_id=3,
              pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>"):
        """
        Train a SentencePiece model.
        
        Args:
            input_file (str): Path to the input text file.
            model_prefix (str): Prefix for the model files.
            vocab_size (int): Size of the vocabulary.
            character_coverage (float): Character coverage.
            model_type (str): Model type, either 'unigram' or 'bpe'.
            input_sentence_size (int): Number of sentences to use for training.
            max_sentence_length (int): Maximum sentence length.
            pad_id (int): ID for padding token.
            unk_id (int): ID for unknown token.
            bos_id (int): ID for beginning of sentence token.
            eos_id (int): ID for end of sentence token.
            pad_piece (str): Piece for padding token.
            unk_piece (str): Piece for unknown token.
            bos_piece (str): Piece for beginning of sentence token.
            eos_piece (str): Piece for end of sentence token.
            
        Returns:
            bool: True if training was successful.
        """
        train_args = f"--input={input_file} "
        train_args += f"--model_prefix={model_prefix} "
        train_args += f"--vocab_size={vocab_size} "
        train_args += f"--character_coverage={character_coverage} "
        train_args += f"--model_type={model_type} "
        train_args += f"--input_sentence_size={input_sentence_size} "
        train_args += f"--max_sentence_length={max_sentence_length} "
        train_args += f"--pad_id={pad_id} --unk_id={unk_id} --bos_id={bos_id} --eos_id={eos_id} "
        train_args += f"--pad_piece={pad_piece} --unk_piece={unk_piece} "
        train_args += f"--bos_piece={bos_piece} --eos_piece={eos_piece} "
        train_args += "--normalization_rule_name=identity"  # Preserve Hindi text as is
        
        self.logger.info(f"Training SentencePiece model with vocab size {vocab_size}")
        spm.SentencePieceTrainer.train(train_args)
        
        model_path = f"{model_prefix}.model"
        vocab_path = f"{model_prefix}.vocab"
        
        if os.path.exists(model_path) and os.path.exists(vocab_path):
            self.logger.info(f"Training successful. Model saved to {model_path}")
            return True
        else:
            self.logger.error("Training failed")
            return False
