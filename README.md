# GPT-from-Scratch (Decoder-only Transformer)

This repository contains an implementation of a Generative Pretrained Transformer (GPT) model from scratch using a decoder-only transformer architecture. The model utilizes self-attention mechanisms to generate coherent text based on the training data provided. 

**Key Features**

* **Self-attention mechanism:** The core feature that allows the model to capture long-range dependencies and context within the text.
* **Decoder-only architecture:** The model is based on a decoder-only transformer designed for autoregressive generation (generating one token at a time).
* **End-to-end workflow:** The entire process—training the model and generating text—is contained in a single Python file (gpt.py), making it easy to use.
* **Text generation:** After training, the model can generate text step-by-step, predicting the next token based on previous ones.

**Requirements**

Before running the script, ensure you have the following dependencies installed:

* Python 3.7+
* PyTorch 1.9+
* NumPy

You can install the required libraries by running:

```bash
pip install torch numpy
