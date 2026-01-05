# RNN-LSTM-Image-Captioning

⚠️ **Coursework Assignment Notice**  
This repository contains code developed **strictly as part of a university coursework assignment** on image captioning.  
All implementations are intended **for educational purposes only** and are **not designed for production use**.

---

## 📌 Project Overview

This coursework implements an **end-to-end image captioning system** using **Vanilla RNNs, LSTMs, and Attention-based LSTMs** in PyTorch. The primary goal of the assignment is to develop a deep, hands-on understanding of:

- Recurrent neural networks and sequence modeling  
- Long Short-Term Memory (LSTM) architectures  
- Attention mechanisms over spatial CNN features  
- Training and inference pipelines for image captioning  

To meet the learning objectives, most components are implemented **from scratch**, avoiding high-level PyTorch abstractions where possible.

---

## 🎓 Coursework Context

This project corresponds to an **Image Captioning assignment** in a Deep Learning / Computer Vision course.  
The assignment follows a **notebook-driven workflow** and emphasizes:

- Manual implementation of RNN and LSTM forward passes  
- Understanding backpropagation through time (BPTT)  
- Integration of CNN encoders with sequence decoders  
- Scaled dot-product attention  
- Sequence-level loss computation with masking  

---

---

## 📄 File Descriptions

### `rnn_lstm_captioning.py`
This file contains **all core implementations** required by the assignment :contentReference[oaicite:0]{index=0}, including:

- **ImageEncoder**
  - CNN-based encoder using pretrained RegNet-X 400MF
  - Outputs spatial feature maps for attention models
- **WordEmbedding**
  - Custom word embedding layer
- **Recurrent Models (from scratch)**
  - Vanilla RNN
  - LSTM
  - Attention-based LSTM
- **Attention Mechanism**
  - Scaled dot-product attention over CNN feature grids
- **CaptioningRNN**
  - Unified image captioning model supporting:
    - `cell_type="rnn"`
    - `cell_type="lstm"`
    - `cell_type="attn"`
- **Temporal Softmax Loss**
  - Sequence-level cross-entropy loss with `<NULL>` masking
- **Inference / Sampling**
  - Greedy decoding for caption generation

This file focuses purely on **model logic and learning algorithms**, as required by the coursework.

---

### `rnn_lstm_captioning_main.py`
This file is a **notebook-style execution script** adapted from Google Colab :contentReference[oaicite:1]{index=1}. It is responsible for:

- Mounting Google Drive and setting up the runtime environment
- Loading the preprocessed **COCO Captions dataset**
- Running **sanity checks** for all implemented functions
- Training RNN, LSTM, and Attention-based captioning models
- Overfitting on small datasets for debugging
- Performing full training runs
- Sampling captions at test time
- Visualizing attention maps for Attention LSTM
- Saving final losses for submission

This file **calls and evaluates** the functions and classes implemented in `rnn_lstm_captioning.py`.

---

## 🧠 Supported Models

| Model Type | Description |
|-----------|-------------|
| `rnn` | Vanilla RNN with tanh activation |
| `lstm` | Standard LSTM with gating mechanisms |
| `attn` | Attention-based LSTM with spatial attention |

---

## 🚀 Training Pipeline

During training (executed from `rnn_lstm_captioning_main.py`):

1. Images are encoded using a CNN backbone  
2. CNN features initialize hidden states  
3. Input captions are embedded into word vectors  
4. Sequence models process word embeddings  
5. Vocabulary scores are produced at each timestep  
6. Temporal softmax loss is computed (ignoring `<NULL>` tokens)  

---

## 🎯 Inference & Sampling

At test time:
- Caption generation starts with the `<START>` token
- Words are generated sequentially using greedy decoding
- Generation runs for a fixed maximum length
- Attention models return attention maps for visualization

---

## 📦 Dependencies

- Python ≥ 3.8  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib (for visualization)  

---

## ⚠️ Important Notes

- This code prioritizes **clarity and correctness** over efficiency
- No beam search or advanced decoding strategies are implemented
- CNN backbone weights are fixed
- Designed to meet **coursework learning objectives**, not benchmark performance

---

## 📜 Academic Integrity Notice

This repository is provided **for learning and reference only**.  
Reusing or submitting this code for another academic assignment **may violate academic integrity policies**.

---

## 📄 License

This project is intended **solely for academic coursework and educational use**.


