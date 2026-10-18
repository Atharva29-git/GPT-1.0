# Character-Level GPT Language Model

A lightweight, character-level Generative Pre-trained Transformer (GPT) implemented from scratch in PyTorch. This project trains a decoder-only transformer model on a custom text dataset to generate human-like text character by character.

This implementation is heavily inspired by modern autoregressive language models and serves as an excellent educational tool for understanding the inner workings of Transformers, Self-Attention, and neural text generation.

## Features

- **Decoder-Only Transformer Architecture:** Implements the core components of modern LLMs, including multi-head self-attention, feed-forward layers, and layer normalization.
- **Character-Level Tokenization:** Learns to predict the next character in a sequence, eliminating the need for complex sub-word tokenizers like BPE.
- **GPU Acceleration:** Automatically detects and utilizes CUDA if available, falling back to CPU otherwise.
- **Configurable Hyperparameters:** Easily tweak embedding dimensions, number of attention heads, layers, and context length.
- **Built-in Text Generation:** Includes a `.generate()` method to sample text from the trained model.

## Requirements

- **PyTorch:** Designed to run optimally on CUDA-enabled GPUs, falling back to CPU if unavailable.
- **Dataset:** The script expects a plain text file named `input.txt` in the root directory. The model dynamically builds its vocabulary (unique characters) and learns text patterns directly from this corpus.

## Architecture Details

The model is built using the following PyTorch modules:

| Module | Description |
| --- | --- |
| `Head` | A single head of self-attention. |
| `MultiHeadAttention` | Multiple attention heads running in parallel, concatenating their outputs. |
| `FeedForward` | A simple multi-layer perceptron (MLP) applied to each position independently. |
| `Block` | A standard Transformer block combining Multi-Head Attention and the FeedForward network with residual connections and Layer Normalization. |
| `GPTLanguageModel` | The overarching wrapper that includes token embeddings, positional embeddings, a sequence of Transformer Blocks, and the final linear classification head. |

## Configuration & Hyperparameters

Constants at the top of the script define the model's capacity and training behavior. The default configuration produces a model with ~10 million parameters:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `batch_size` | 64 | Number of independent sequences processed in parallel. |
| `block_size` | 256 | Maximum context length (sequence length) for predictions. |
| `max_iters` | 5000 | Total number of training iterations. |
| `learning_rate` | 3e-4 | Learning rate for the AdamW optimizer. |
| `n_embd` | 384 | Dimensionality of the token and positional embeddings. |
| `n_head` | 6 | Number of parallel attention heads. |
| `n_layer` | 6 | Number of consecutive Transformer blocks. |
| `dropout` | 0.2 | Dropout probability for regularization. |
