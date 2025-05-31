"""
Preprocessing utilities for Hindi text.
"""

import re
from hindi_tokenizer.utils import normalize_hindi_text


class HindiPreprocessor:
    """
    Preprocessor for Hindi text to prepare it for tokenization.
    """
    
    def __init__(self, remove_latin=False, normalize=True, remove_numbers=False):
        """
        Initialize the Hindi preprocessor.
        
        Args:
            remove_latin (bool): Whether to remove Latin script characters.
            normalize (bool): Whether to normalize the text.
            remove_numbers (bool): Whether to remove numbers.
        """
        self.remove_latin = remove_latin
        self.normalize = normalize
        self.remove_numbers = remove_numbers
    
    def preprocess(self, text):
        """
        Preprocess Hindi text.
        
        Args:
            text (str): Input Hindi text.
            
        Returns:
            str: Preprocessed Hindi text.
        """
        if self.normalize:
            text = normalize_hindi_text(text)
        
        if self.remove_latin:
            # Remove Latin script characters (a-z, A-Z)
            text = re.sub(r'[a-zA-Z]+', '', text)
        
        if self.remove_numbers:
            # Remove digits (both Hindi and Latin)
            text = re.sub(r'[0-9०-९]+', '', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def clean_corpus(self, input_file, output_file):
        """
        Clean a corpus file and write the result to a new file.
        
        Args:
            input_file (str): Path to the input corpus file.
            output_file (str): Path to the output cleaned corpus file.
            
        Returns:
            int: Number of lines processed.
        """
        line_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                
                cleaned_line = self.preprocess(line)
                if cleaned_line:
                    fout.write(cleaned_line + '\n')
                    line_count += 1
        
        return line_count
