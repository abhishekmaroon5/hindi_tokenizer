# Hindi SentencePiece Tokenizer

A production-ready SentencePiece tokenizer specifically designed and optimized for the Hindi language using the Devanagari script.

## 🌟 Features

- 🇮🇳 **Hindi-optimized**: Specifically designed for Devanagari script and Hindi language patterns
- 🚀 **High Performance**: Fast encoding/decoding with caching support
- 📊 **Comprehensive Evaluation**: Built-in metrics and analysis tools
- 🔧 **Production Ready**: Docker support, API server, and monitoring
- 🧪 **Well Tested**: Comprehensive test suite with CI/CD
- 📚 **Easy Integration**: Compatible with popular ML frameworks

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/hindi-sentencepiece-tokenizer.git
cd hindi-sentencepiece-tokenizer
pip install -r requirements.txt
```

### Basic Usage

```python
from hindi_tokenizer import HindiTokenizer

# Initialize tokenizer
tokenizer = HindiTokenizer()

# Train on your Hindi corpus
tokenizer.train(
    input_file="hindi_corpus.txt",
    model_name="hindi_tokenizer",
    vocab_size=32000
)

# Use the tokenizer
text = "यह एक हिंदी वाक्य है।"
tokens = tokenizer.encode(text)
pieces = tokenizer.encode_as_pieces(text)
decoded = tokenizer.decode(tokens)

print(f"Original: {text}")
print(f"Tokens: {tokens}")
print(f"Pieces: {pieces}")
print(f"Decoded: {decoded}")
```

### Command Line Usage

```bash
# Train a new model
python -m hindi_tokenizer train --input data/hindi_corpus.txt --vocab-size 32000

# Encode text
python -m hindi_tokenizer encode --text "आपका हिंदी टेक्स्ट यहाँ"

# Evaluate model
python -m hindi_tokenizer evaluate --test-file data/test_hindi.txt
```

## 📁 Repository Structure

```
hindi-sentencepiece-tokenizer/
├── hindi_tokenizer/           # Main package
│   ├── __init__.py
│   ├── core.py               # Core tokenizer implementation
│   ├── trainer.py            # Training utilities
│   ├── evaluator.py          # Evaluation tools
│   ├── preprocessor.py       # Hindi text preprocessing
│   └── utils.py              # Helper utilities
├── api/                      # REST API
│   ├── __init__.py
│   ├── server.py            # FastAPI server
│   └── models.py            # API data models
├── scripts/                  # Utility scripts
│   ├── download_data.py     # Download Hindi datasets
│   ├── prepare_corpus.py    # Corpus preparation
│   └── benchmark.py         # Performance benchmarking
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_tokenizer.py    # Core tests
│   ├── test_trainer.py      # Training tests
│   └── test_api.py          # API tests
├── data/                    # Sample data
│   ├── sample_hindi.txt
│   └── test_sentences.txt
├── models/                  # Trained models
│   └── .gitkeep
├── docs/                    # Documentation
│   ├── usage.md
│   ├── training.md
│   └── api.md
├── docker/                  # Docker files
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/                 # GitHub workflows
│   └── workflows/
│       └── ci.yml
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
├── pyproject.toml          # Modern Python packaging
├── .gitignore              # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks
└── LICENSE                 # MIT License
```
