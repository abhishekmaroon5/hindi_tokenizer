# Hindi Tokenizer Documentation

This document provides detailed information about the Hindi Tokenizer project, including its architecture, components, usage examples, and technical details.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Training Process](#training-process)
5. [Tokenization Process](#tokenization-process)
6. [API Reference](#api-reference)
7. [Gradio UI](#gradio-ui)
8. [Performance Metrics](#performance-metrics)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

## Overview

The Hindi Tokenizer is a specialized tool designed for subword tokenization of Hindi text written in the Devanagari script. It uses the SentencePiece library with a Unigram model to create efficient tokenization for Hindi language processing tasks, particularly useful for machine learning applications like machine translation, text classification, and language modeling.

The tokenizer is trained on a large corpus of Hindi text and can efficiently break down Hindi sentences into subword units, which helps capture the morphological richness of the language while maintaining a manageable vocabulary size.

## Architecture

The project follows a modular architecture with the following main components:

```
hindi_tokenizer/
├── core.py         # Core tokenizer implementation
├── trainer.py      # Training utilities
├── evaluator.py    # Evaluation tools
├── preprocessor.py # Hindi text preprocessing
└── utils.py        # Helper utilities
```

External interfaces:
- FastAPI server for programmatic access
- Gradio UI for interactive exploration

## Core Components

### HindiTokenizer (core.py)

The central class that provides the main tokenization functionality:

```python
class HindiTokenizer:
    def __init__(self, model_path=None):
        """Initialize the tokenizer with an optional model path."""
        
    def train(self, input_file, model_prefix, vocab_size=32000, 
              character_coverage=0.9999, model_type="unigram"):
        """Train a new tokenizer model on the provided input file."""
        
    def load(self, model_path):
        """Load a pre-trained tokenizer model."""
        
    def encode(self, text):
        """Encode text to token IDs."""
        
    def encode_as_pieces(self, text):
        """Encode text to subword pieces."""
        
    def decode(self, ids):
        """Decode token IDs back to text."""
        
    def vocab_size(self):
        """Return the vocabulary size of the model."""
```

### HindiPreprocessor (preprocessor.py)

Handles text normalization and cleaning specific to Hindi:

```python
class HindiPreprocessor:
    def __init__(self, remove_latin=True, min_hindi_char_ratio=0.5):
        """Initialize the preprocessor with configuration options."""
        
    def normalize_text(self, text):
        """Normalize Hindi text by handling various character forms."""
        
    def remove_latin_characters(self, text):
        """Remove Latin script characters from the text."""
        
    def is_valid_hindi_line(self, text):
        """Check if a line contains sufficient Hindi characters."""
        
    def process_line(self, line):
        """Process a single line of text."""
        
    def process_file(self, input_file, output_file, sample_size=None):
        """Process an entire file, with optional sampling."""
```

### TokenizerTrainer (trainer.py)

Manages the training process for new tokenizer models:

```python
class TokenizerTrainer:
    def __init__(self, input_file, model_prefix, vocab_size=32000, 
                 character_coverage=0.9999, model_type="unigram"):
        """Initialize the trainer with training parameters."""
        
    def prepare_training_data(self, preprocessor=None, sample_size=None):
        """Prepare and clean the training data."""
        
    def train(self):
        """Train the SentencePiece model."""
```

### TokenizerEvaluator (evaluator.py)

Provides metrics and evaluation tools for tokenizer performance:

```python
class TokenizerEvaluator:
    def __init__(self, tokenizer):
        """Initialize with a tokenizer instance."""
        
    def evaluate_reconstruction(self, test_file, num_samples=1000):
        """Evaluate how well the tokenizer can reconstruct text after tokenization."""
        
    def evaluate_speed(self, test_file, num_samples=1000, iterations=5):
        """Measure tokenization and detokenization speed."""
        
    def generate_report(self, test_file, output_file=None):
        """Generate a comprehensive evaluation report."""
```

## Training Process

The tokenizer is trained using the following process:

1. **Data Collection**: A large corpus of Hindi text is gathered (e.g., IITB Hindi Corpus).

2. **Preprocessing**:
   - Text normalization (handling various forms of Hindi characters)
   - Removal of Latin characters (optional)
   - Filtering lines with insufficient Hindi content
   - Sampling to manage training data size

3. **SentencePiece Training**:
   - Algorithm: Unigram model
   - Vocabulary size: 32,000 tokens (configurable)
   - Character coverage: 99.99% (to ensure comprehensive Hindi script coverage)
   - Special tokens: `<unk>`, `<s>`, `</s>`

4. **Model Output**:
   - `.model` file: Contains the trained model
   - `.vocab` file: Contains the vocabulary with token-to-ID mappings

Example training command:
```bash
python -m hindi_tokenizer train --input data/processed_corpus.txt --vocab-size 32000 --model-prefix models/hindi_tokenizer_32k
```

## Tokenization Process

The tokenization process follows these steps:

1. **Input**: Raw Hindi text

2. **Normalization** (optional preprocessing):
   - Character normalization
   - Removal of unwanted characters

3. **Tokenization**:
   - The text is segmented into subword units using the trained model
   - Each subword is mapped to its corresponding token ID

4. **Output**:
   - Token IDs: Integer representation of each subword
   - Pieces: String representation of each subword

Example tokenization:
```python
# Input text
text = "हिंदी भारत की राष्ट्रभाषा है।"

# Tokenization
token_ids = tokenizer.encode(text)
# [457, 8691, 1098, 4102, 15, 9]

pieces = tokenizer.encode_as_pieces(text)
# ["▁हिंदी", "▁भारत", "▁की", "▁राष्ट्र", "भाषा", "▁है।"]

# Detokenization
reconstructed = tokenizer.decode(token_ids)
# "हिंदी भारत की राष्ट्रभाषा है।"
```

## API Reference

The FastAPI server provides the following endpoints:

### GET `/`
Returns basic information about the API service.

**Response**:
```json
{
  "service": "Hindi Tokenizer API",
  "version": "1.0.0",
  "status": "active",
  "model_loaded": true
}
```

### POST `/tokenize`
Tokenizes the provided Hindi text.

**Request**:
```json
{
  "text": "हिंदी भारत की राष्ट्रभाषा है।",
  "return_pieces": false
}
```

**Response**:
```json
{
  "tokens": [457, 8691, 1098, 4102, 15, 9],
  "pieces": ["▁हिंदी", "▁भारत", "▁की", "▁राष्ट्र", "भाषा", "▁है।"],
  "num_tokens": 6
}
```

### POST `/detokenize`
Converts token IDs back to text.

**Request**:
```json
{
  "tokens": [457, 8691, 1098, 4102, 15, 9]
}
```

**Response**:
```json
{
  "text": "हिंदी भारत की राष्ट्रभाषा है।"
}
```

### GET `/model-info`
Returns information about the loaded tokenizer model.

**Response**:
```json
{
  "model_path": "models/hindi_tokenizer_32k.model",
  "vocab_size": 32000,
  "model_type": "unigram"
}
```

## Gradio UI

The Gradio UI provides an interactive interface for exploring the tokenizer functionality:

### Features

- **Text Input**: Enter or paste Hindi text for tokenization
- **Output Format Selection**: Choose between token IDs or subword pieces
- **Sample Texts**: Pre-loaded examples for quick testing
- **Tokenization Results**: View the tokenized output
- **Reconstructed Text**: Verify the quality of tokenization by checking the reconstructed text
- **Metrics Panel**: View statistics about the tokenization process
  - Total tokens
  - Character count
  - Characters per token ratio
  - Processing time

### Usage

1. Start the Gradio UI:
   ```bash
   python gradio_app.py
   ```

2. Open your browser at http://127.0.0.1:7860

3. Enter Hindi text or select a sample

4. Click "Tokenize" to see the results

## Performance Metrics

The Hindi tokenizer has been evaluated on a variety of metrics:

### Reconstruction Accuracy

- **Perfect Reconstruction Rate**: 100%
- The tokenizer can perfectly reconstruct the original text from the tokenized representation

### Speed Performance

- **Tokenization Speed**: ~1.15 million tokens/second
- **Detokenization Speed**: ~1.25 million tokens/second

(Measured on a standard machine with Intel Core i7 processor)

### Memory Usage

- **Model Size**: ~5MB
- **Runtime Memory**: Minimal footprint suitable for production environments

## Advanced Usage

### Custom Vocabulary Size

You can train the tokenizer with different vocabulary sizes based on your needs:

```python
tokenizer.train(
    input_file="hindi_corpus.txt",
    model_prefix="hindi_tokenizer_custom",
    vocab_size=8000  # Smaller vocabulary
)
```

### Character Coverage

Adjust character coverage to control how comprehensive the tokenizer is with rare characters:

```python
tokenizer.train(
    input_file="hindi_corpus.txt",
    model_prefix="hindi_tokenizer_custom",
    character_coverage=0.9995  # Higher coverage for rare characters
)
```

### Preprocessing Options

Customize preprocessing for specific use cases:

```python
preprocessor = HindiPreprocessor(
    remove_latin=True,  # Remove Latin script
    min_hindi_char_ratio=0.7  # Stricter Hindi content requirement
)

preprocessor.process_file(
    input_file="raw_corpus.txt",
    output_file="processed_corpus.txt",
    sample_size=500000  # Use a smaller sample
)
```

### Integration with Machine Learning Frameworks

The tokenizer can be easily integrated with popular ML frameworks:

```python
# PyTorch example
class HindiDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])
        # Truncate or pad as needed
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens)
```

## Troubleshooting

### Model Loading Issues

If you encounter issues loading the model:

1. Ensure the model path is correct and the file exists
2. Check that you have read permissions for the model file
3. Verify that the model was trained with a compatible version of SentencePiece

### Tokenization Errors

If tokenization produces unexpected results:

1. Check if the input text contains characters outside the training data's distribution
2. Ensure proper text normalization before tokenization
3. For very long texts, consider processing in smaller chunks

### Performance Issues

If tokenization is slower than expected:

1. Batch process texts when possible
2. Use the Python multiprocessing module for parallel processing
3. Consider using the C++ implementation of SentencePiece for production environments

### API Connection Problems

If you cannot connect to the API server:

1. Verify the server is running (`python run_api_server.py`)
2. Check that the port (default: 8000) is not in use by another application
3. Ensure your firewall allows connections to the port

---

For additional support or to report issues, please open an issue on the GitHub repository.
