# RNN-LSTM-Image-Captioning

This repository contains a **from-scratch implementation of an image captioning system** using **Vanilla RNNs, LSTMs, and Attention-based LSTMs** in PyTorch. The model generates natural language captions for images by combining a **CNN image encoder** with **recurrent sequence models**.

The code is primarily designed for **educational and experimental purposes**, focusing on understanding how sequence models work internally rather than relying on high-level PyTorch abstractions.

---

## 📌 Overview

The image captioning pipeline consists of:

1. **Image Encoder**
   - A pretrained **RegNet-X 400MF** CNN extracts spatial image features.
2. **Word Embedding Layer**
   - Converts word indices into dense vector representations.
3. **Sequence Model**
   - One of:
     - Vanilla RNN
     - LSTM
     - Attention-based LSTM
4. **Output Projection**
   - Maps hidden states to vocabulary scores.
5. **Temporal Softmax Loss**
   - Computes sequence-level cross-entropy loss during training.

---

## 🧠 Supported Models

| Model Type | Description |
|-----------|-------------|
| `rnn` | Vanilla RNN with tanh activation |
| `lstm` | Standard LSTM with input, forget, output, and gate cells |
| `attn` | Attention-based LSTM with scaled dot-product attention over CNN features |

---


All components are implemented inside a **single file** for clarity and grading convenience :contentReference[oaicite:0]{index=0}.

---

## 🧩 Key Components

### 1. ImageEncoder
- Uses **RegNet-X 400MF** pretrained on ImageNet
- Outputs spatial feature maps (`C × H × W`)
- Normalizes inputs using ImageNet statistics

### 2. Recurrent Modules (From Scratch)
- `rnn_step_forward`, `rnn_step_backward`
- `rnn_forward`, `rnn_backward`
- Custom `RNN` class
- Custom `LSTM` class
- Custom `AttentionLSTM` class

All recurrent logic is implemented **manually**, without using `torch.nn.RNN` or `torch.nn.LSTM`.

---

### 3. Attention Mechanism
- Implements **scaled dot-product attention**
- Computes alignment between:
  - Previous hidden state
  - Spatial CNN features
- Produces attention weights over a `4 × 4` feature grid

---

### 4. CaptioningRNN
Main model class that:
- Encodes images
- Embeds words
- Runs sequence model
- Computes training loss
- Generates captions at test time

Supports:
```python
cell_type = "rnn" | "lstm" | "attn"

## 🗂 File Structure

