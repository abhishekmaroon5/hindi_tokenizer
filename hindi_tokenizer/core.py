"""
Core implementation of the Hindi SentencePiece Tokenizer.
"""

import os
import sentencepiece as spm


class HindiTokenizer:
    """
    A SentencePiece tokenizer optimized for Hindi language.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the Hindi tokenizer.
        
        Args:
            model_path (str, optional): Path to a pre-trained SentencePiece model.
                If None, the tokenizer will need to be trained before use.
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = spm.SentencePieceProcessor()
            self.model.load(model_path)
    
    def train(self, input_file, model_name, vocab_size=32000, character_coverage=0.9995,
              model_type="unigram", input_sentence_size=1000000, max_sentence_length=4192):
        """
        Train a new SentencePiece model on Hindi text.
        
        Args:
            input_file (str): Path to the input text file (UTF-8 encoded Hindi text).
            model_name (str): Prefix for the model files.
            vocab_size (int): Size of the vocabulary.
            character_coverage (float): Character coverage.
            model_type (str): Model type, either 'unigram' or 'bpe'.
            input_sentence_size (int): Number of sentences to use for training.
            max_sentence_length (int): Maximum sentence length.
            
        Returns:
            bool: True if training was successful.
        """
        train_args = f"--input={input_file} "
        train_args += f"--model_prefix={model_name} "
        train_args += f"--vocab_size={vocab_size} "
        train_args += f"--character_coverage={character_coverage} "
        train_args += f"--model_type={model_type} "
        train_args += f"--input_sentence_size={input_sentence_size} "
        train_args += f"--max_sentence_length={max_sentence_length} "
        train_args += "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        train_args += "--pad_piece=<pad> --unk_piece=<unk> --bos_piece=<s> --eos_piece=</s> "
        train_args += "--normalization_rule_name=identity"  # Preserve Hindi text as is
        
        spm.SentencePieceTrainer.train(train_args)
        
        # Load the trained model
        model_path = f"{model_name}.model"
        if os.path.exists(model_path):
            self.model = spm.SentencePieceProcessor()
            self.model.load(model_path)
            return True
        return False
    
    def encode(self, text):
        """
        Encode Hindi text to token IDs.
        
        Args:
            text (str): Input Hindi text.
            
        Returns:
            list: List of token IDs.
        """
        if self.model is None:
            raise ValueError("Tokenizer model not loaded. Train or load a model first.")
        return self.model.encode(text)
    
    def encode_as_pieces(self, text):
        """
        Encode Hindi text to token pieces (subwords).
        
        Args:
            text (str): Input Hindi text.
            
        Returns:
            list: List of token pieces.
        """
        if self.model is None:
            raise ValueError("Tokenizer model not loaded. Train or load a model first.")
        return self.model.encode_as_pieces(text)
    
    def decode(self, ids):
        """
        Decode token IDs back to Hindi text.
        
        Args:
            ids (list): List of token IDs.
            
        Returns:
            str: Decoded Hindi text.
        """
        if self.model is None:
            raise ValueError("Tokenizer model not loaded. Train or load a model first.")
        return self.model.decode(ids)
