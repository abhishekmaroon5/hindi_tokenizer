#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradio web interface for Hindi Tokenizer.
"""

import os
import time
import gradio as gr
from hindi_tokenizer.core import HindiTokenizer

# Load the tokenizer model
MODEL_PATH = os.environ.get('TOKENIZER_MODEL_PATH', 'models/hindi_tokenizer_32k.model')
tokenizer = HindiTokenizer(model_path=MODEL_PATH)

# Sample Hindi texts for demonstration
SAMPLE_TEXTS = [
    "हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है।",
    "भारतीय राजनीति में पिछड़ेपन के विचार का सूत्रीकरण उसे खत्म करने के मकसद से किया गया था",
    "आज का मौसम बहुत सुहावना है। बारिश के बाद चारों ओर हरियाली छा गई है।",
    "मैं आज सुबह जल्दी उठा और पार्क में टहलने गया। वहां कई लोग योग और व्यायाम कर रहे थे।"
]

def tokenize_text(text, output_format="ids"):
    """
    Tokenize the input text and return the results.
    
    Args:
        text (str): Input Hindi text
        output_format (str): Format of output tokens ("ids" or "pieces")
        
    Returns:
        tuple: (tokens, reconstructed_text, metrics)
    """
    if not text.strip():
        return "[]", "", {"Total Tokens": 0, "Characters": 0, "Chars/Token": 0, "Processing Time": "0 ms"}
    
    start_time = time.time()
    
    if output_format == "ids":
        tokens = tokenizer.encode(text)
        token_display = str(tokens)
    else:
        tokens = tokenizer.encode_as_pieces(text)
        token_display = str(tokens)
    
    reconstructed = tokenizer.decode(tokenizer.encode(text))
    
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Calculate metrics
    total_tokens = len(tokens)
    total_chars = len(text)
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    
    metrics = {
        "Total Tokens": total_tokens,
        "Characters": total_chars,
        "Chars/Token": round(chars_per_token, 2),
        "Processing Time": f"{processing_time:.2f} ms"
    }
    
    return token_display, reconstructed, metrics

def load_sample(sample_idx):
    """Load a sample text"""
    if 0 <= sample_idx < len(SAMPLE_TEXTS):
        return SAMPLE_TEXTS[sample_idx]
    return ""

def create_ui():
    """Create the Gradio UI"""
    with gr.Blocks(title="Hindi Tokenizer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# हिंदी टोकनाइज़र (Hindi Tokenizer)")
        gr.Markdown("### A subword tokenization tool for Hindi language using SentencePiece")
        
        with gr.Row():
            with gr.Column():
                # Input section
                gr.Markdown("### Input Text")
                text_input = gr.Textbox(
                    placeholder="Enter Hindi text here...",
                    lines=5,
                    label="Hindi Text"
                )
                
                with gr.Row():
                    tokenize_btn = gr.Button("Tokenize", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                output_format = gr.Radio(
                    ["ids", "pieces"], 
                    label="Output Format", 
                    value="ids",
                    info="Choose between token IDs or subword pieces"
                )
                
                # Sample selector
                gr.Markdown("### Sample Texts")
                with gr.Row():
                    sample_btns = [gr.Button(f"Sample {i+1}") for i in range(len(SAMPLE_TEXTS))]
            
            with gr.Column():
                # Output section
                gr.Markdown("### Tokenization Results")
                tokens_output = gr.Textbox(label="Tokens", lines=3)
                reconstructed_output = gr.Textbox(label="Reconstructed Text", lines=3)
                
                # Metrics
                gr.Markdown("### Metrics")
                metrics_output = gr.JSON(label="Tokenization Metrics")
        
        # Set up event handlers
        tokenize_btn.click(
            tokenize_text, 
            inputs=[text_input, output_format], 
            outputs=[tokens_output, reconstructed_output, metrics_output]
        )
        
        clear_btn.click(
            lambda: ("", "", {"Total Tokens": 0, "Characters": 0, "Chars/Token": 0, "Processing Time": "0 ms"}),
            inputs=None,
            outputs=[text_input, tokens_output, reconstructed_output, metrics_output]
        )
        
        # Sample button handlers
        for i, btn in enumerate(sample_btns):
            btn.click(
                load_sample,
                inputs=[gr.Number(value=i, visible=False)],
                outputs=[text_input]
            )
        
        # Model information
        with gr.Accordion("Model Information", open=False):
            model_info = {
                "Model Path": MODEL_PATH,
                "Vocabulary Size": tokenizer.vocab_size() if hasattr(tokenizer, "vocab_size") else "Unknown",
                "Model Type": "SentencePiece Unigram"
            }
            gr.JSON(value=model_info)
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("Hindi Tokenizer - Built with SentencePiece and Gradio")
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)
