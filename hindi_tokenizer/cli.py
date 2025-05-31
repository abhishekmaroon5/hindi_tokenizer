"""
Command-line interface for the Hindi Tokenizer.
"""

import argparse
import sys
from hindi_tokenizer.core import HindiTokenizer


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Hindi SentencePiece Tokenizer")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new tokenizer model")
    train_parser.add_argument("--input", required=True, help="Input text file for training")
    train_parser.add_argument("--model-name", required=True, help="Prefix for the model files")
    train_parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    train_parser.add_argument("--character-coverage", type=float, default=0.9995, 
                             help="Character coverage")
    train_parser.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram", 
                             help="Model type")
    
    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode text to tokens")
    encode_parser.add_argument("--model", required=True, help="Path to the model file")
    encode_parser.add_argument("--text", required=True, help="Text to encode")
    encode_parser.add_argument("--output-pieces", action="store_true", 
                              help="Output token pieces instead of IDs")
    
    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode tokens to text")
    decode_parser.add_argument("--model", required=True, help="Path to the model file")
    decode_parser.add_argument("--tokens", required=True, nargs="+", type=int, 
                              help="Token IDs to decode")
    
    args = parser.parse_args()
    
    if args.command == "train":
        tokenizer = HindiTokenizer()
        success = tokenizer.train(
            input_file=args.input,
            model_name=args.model_name,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            model_type=args.model_type
        )
        if success:
            print(f"Model trained successfully: {args.model_name}.model")
        else:
            print("Failed to train model", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "encode":
        tokenizer = HindiTokenizer(model_path=args.model)
        if args.output_pieces:
            pieces = tokenizer.encode_as_pieces(args.text)
            print(f"Pieces: {pieces}")
        else:
            ids = tokenizer.encode(args.text)
            print(f"IDs: {ids}")
    
    elif args.command == "decode":
        tokenizer = HindiTokenizer(model_path=args.model)
        text = tokenizer.decode(args.tokens)
        print(f"Decoded text: {text}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
