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

You can install the required libraries by running:

```bash
pip install torch
```

**How to Use**
* **1. Training the Model**
To train the model, simply run the following command:

```bash
python gpt.py
```
This will start the training process. By default, the script uses a simple dataset (you can modify the dataset path within the script to use your own text data). During training, the model learns to predict the next token in the sequence, and the loss is printed at intervals.

* **2. Text Generation**
Once training is complete, the model can generate text. The script includes functionality to generate text after training. By modifying the code slightly, you can change the initial seed or prompt, and the model will generate text step-by-step based on what it has learned.

**Model Architecture**
The model is based on a decoder-only transformer, which is a simplified version of the original Transformer architecture proposed in the "Attention Is All You Need" paper. Key components include:

* **Self-attention Mechanism**
The self-attention mechanism enables the model to compute relationships between all tokens in the input sequence. This allows the model to attend to relevant parts of the sequence, regardless of the token’s position.
* **Decoder-Only Model**
In this model, only the decoder block of the transformer is used for autoregressive text generation. Each output token is generated by predicting the next token in the sequence, conditioned on the previously generated tokens.

**gpt.py**
* This file contains the entire code for training the model and generating text. It includes:
* The definition of the transformer model with the self-attention mechanism.
* Training loop for learning from the dataset.
* Text generation functionality after training.

**Example Workflow**
* **Run the training loop:**
 By executing python gpt.py, the model begins training on your dataset. The training loss will be printed at regular intervals.
* **Model saves automatically:**
 Once the training is complete, the model's weights are saved, and it can be used to generate text based on the learned knowledge.
* **Text generation:**
 After training, the model can be used to generate text by modifying the seed input in the script. It will generate text token-by-token, continuing from the seed provided.

# Results:
* Results can be seen in "results.txt" file in repo. I had used two datasets for this model which are shakespheare dataset(input.txt) and sherlock homes dataset(input2.txt). The model was able to generate the text in the dataset fashion. 

# Future Improvements 
* **Custom datasets**: Extend the script to support various text datasets with customizable preprocessing steps (e.g., tokenization).
* **Hyperparameter tuning**: Implement argument parsing to configure hyperparameters like batch_size, learning_rate, etc., from the command line.
* **Model checkpoints**: Add functionality to save model checkpoints during training to avoid losing progress in case of interruptions.
* **Advanced text generation**: Implement temperature sampling, top-k sampling, or other techniques to improve the quality and creativity of generated text.
# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Notes:
* **Simplicity**: This project is designed to be simple and easy to understand, with all code for training and text generation contained in a single script.
* **Text Generation**: Since the model is autoregressive, it generates text one token at a time, predicting each token based on the previous ones.

