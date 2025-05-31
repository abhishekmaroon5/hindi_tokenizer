#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the Hindi Tokenizer API server.
"""

import os
import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description='Run Hindi Tokenizer API server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--model-path', default='models/hindi_tokenizer_32k.model', 
                        help='Path to the tokenizer model')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    # Set the environment variable for the model path
    os.environ['TOKENIZER_MODEL_PATH'] = args.model_path
    
    # Run the server
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
