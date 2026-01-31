# 🤖 Building LLMs From Scratch

A comprehensive, hands-on implementation guide for understanding and building Large Language Models from first principles. This repository takes you from raw text processing to a complete GPT-style architecture through progressive, well-documented Jupyter notebooks.

---

## 🎯 Overview

This project demystifies the inner workings of modern Large Language Models by implementing each core component from scratch. Rather than treating LLMs as black boxes, you'll build and understand every layer - from tokenization strategies to multi-head attention mechanisms.

**Perfect for:** ML practitioners, researchers, and anyone who wants to deeply understand transformer-based language models beyond surface-level tutorials.

---

## 🏗️ Project Structure

```
LLM-From-Scratch/
│
├── 📓 Notebooks (Learning Path)
│   ├── 01. Data_preparation_&_sampling.ipynb    # Text → Tensors pipeline
│   ├── 02. Vector_embedding.ipynb               # Semantic embeddings
│   ├── 03. Attention_mechanism.ipynb            # Self-attention & multi-head
│   ├── 04. LLM_architecture(GPT).ipynb          # Complete GPT model
│   ├── 05. LLM_Loss_function.ipynb              # Loss calculation & optimization
│   ├── 06. LLM_Pretraining.ipynb                # Model training pipeline
│   ├── 08. Understanding_GPT2_Weights.ipynb     # Exploring pretrained weights
│   ├── 09. Model_Weights_Loading.ipynb          # Loading OpenAI GPT-2 weights
│   └── 10. GPT2_architecture_only.ipynb         # Minimal GPT-2 architecture
│
├── 🐍 Python Scripts
│   ├── 07. GPT-2_weights_download.py            # Download GPT-2 checkpoints
│   └── 11. GPT-2_complete_model.py              # Complete model implementation
│
├── 📦 Model Weights (auto-downloaded)
│   └── gpt2/124M/                               # GPT-2 Small pretrained weights
│
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
└── README.md                                    # You are here
```

---

## ✨ What You'll Learn

- **Text Processing Pipeline**: Tokenization strategies, vocabulary construction, and BPE encoding
- **Embedding Techniques**: Token embeddings, positional encodings, and semantic vector spaces
- **Attention Mechanisms**: Self-attention, multi-head attention, and causal masking
- **Transformer Architecture**: Complete GPT-style model implementation with all components
- **Pretrained Weights**: Loading and using OpenAI's GPT-2 weights
- **Modern Best Practices**: Real-world techniques used in production LLMs

---

## 📚 Curriculum

### [01. Data Preparation & Sampling](01.%20Data_preparation_%26_sampling.ipynb)

**Foundation: From Text to Tensors**

Build a complete data processing pipeline for language models:

- Custom tokenization using regex patterns
- Vocabulary construction and token mapping
- SimpleTokenizer implementation (V1 & V2)
- Byte Pair Encoding (BPE) with GPT-2's `tiktoken`
- Sliding window data generation for next-token prediction
- PyTorch Dataset and DataLoader implementation
- Token and positional embeddings

**Key Outputs:** Production-ready data loaders and embedding layers

---

### [02. Vector Embeddings](02.%20Vector_embedding.ipynb)

**Understanding Semantic Spaces**

Explore pretrained embeddings and semantic relationships:

- Loading and using Google's Word2Vec (300D)
- Computing cosine similarity between words
- Vector arithmetic for analogies (_king - man + woman ≈ queen_)
- Distance-based semantic analysis
- Understanding embedding geometry

**Key Outputs:** Intuition for how meaning is encoded in vector spaces

---

### [03. Attention Mechanism](03.%20Attention_mechanism.ipynb)

**The Core of Modern LLMs**

Implement the attention mechanism that powers transformers:

- Self-attention from scratch
- Query, Key, Value projections
- Scaled dot-product attention
- Multi-head attention architecture
- Causal masking for autoregressive generation
- Attention weight visualization

**Key Outputs:** Complete multi-head attention implementation

---

### [04. LLM Architecture (GPT)](04.%20LLM_architecture%28GPT%29.ipynb)

**Building a Complete Language Model**

Assemble all components into a working GPT-style model:

- Transformer blocks with attention and feedforward layers
- Layer normalization and residual connections
- Complete GPT architecture
- Model initialization and configuration
- Forward pass implementation
- Understanding model capacity and scaling

**Key Outputs:** Fully functional GPT model ready for training

---

### [05. LLM Loss Function](05.%20LLM_Loss_function.ipynb)

**Training Objectives & Optimization**

Implement loss functions and understand model training:

- Cross-entropy loss for language modeling
- Perplexity metrics
- Loss calculation across batches
- Text generation with trained models
- Understanding training dynamics

**Key Outputs:** Loss computation and generation pipeline

---

### [06. LLM Pretraining](06.%20LLM_Pretraining.ipynb)

**Complete Training Pipeline**

Train a GPT model from scratch:

- Data loading and preprocessing
- Training loop implementation
- Validation and model evaluation
- Learning rate scheduling
- GPU/CPU device management
- Model checkpointing and saving
- Text generation and inference

**Key Outputs:** Fully trained language model capable of text generation

---

### [07. GPT-2 Weights Download](07.%20GPT-2_weights_download.py)

**Downloading Pretrained Weights**

Python script to download OpenAI's GPT-2 model weights:

- Download GPT-2 checkpoints from OpenAI
- Support for different model sizes (124M, 355M, 774M, 1558M)
- Progress tracking during download
- Automatic file organization

**Key Outputs:** Local copy of GPT-2 pretrained weights

---

### [08. Understanding GPT-2 Weights](08.%20Understanding_GPT2_Weights.ipynb)

**Exploring Pretrained Model Structure**

Deep dive into GPT-2's weight structure:

- Loading TensorFlow checkpoints
- Understanding weight naming conventions
- Exploring layer-by-layer parameters
- Comparing architecture configurations

**Key Outputs:** Understanding of how pretrained weights are organized

---

### [09. Model Weights Loading](09.%20Model_Weights_Loading.ipynb)

**Loading OpenAI GPT-2 Weights**

Complete pipeline for using pretrained weights:

- Converting TensorFlow weights to PyTorch
- Mapping OpenAI weights to our architecture
- Loading weights into custom GPT model
- Text generation with pretrained model
- Saving and loading PyTorch checkpoints

**Key Outputs:** Working GPT-2 model with pretrained weights

---

### [10. GPT-2 Architecture Only](10.%20GPT2_architecture_only.ipynb)

**Minimal GPT-2 Implementation**

Clean, minimal GPT-2 Small architecture:

- Complete model in ~100 lines of code
- Well-commented implementation
- Model verification and testing
- Parameter counting (124M)

**Key Outputs:** Reference implementation for GPT-2 architecture

---

### [11. GPT-2 Complete Model](11.%20GPT-2_complete_model.py)

**Production-Ready Implementation**

Complete GPT-2 model as a standalone Python module:

- All architecture components
- Ready for import and use
- Clean, modular code structure

**Key Outputs:** Importable GPT-2 model module

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)
- 4GB+ RAM (8GB+ recommended for training)
- GPU optional (CUDA-compatible for faster training)

### Installation

#### Option 1: Using pip with requirements.txt (Recommended)

```bash
# Clone the repository
git clone https://github.com/sugam24/LLM-From-Scratch.git
cd LLM-From-Scratch

# Create a virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

#### Option 2: Manual Installation

```bash
# Install PyTorch (visit pytorch.org for your specific system)
pip install torch torchvision torchaudio

# Install other dependencies
pip install tiktoken matplotlib gensim kagglehub jupyter tensorflow tqdm requests
```

### Usage

Navigate through the notebooks sequentially (01 → 06). Each notebook is self-contained but builds conceptually on previous ones:

```bash
# Launch Jupyter Notebook
jupyter notebook

# Or use VS Code with Jupyter extension (recommended)
# Open any .ipynb file directly in VS Code
```

**Recommended Order:**

1. Start with `01. Data_preparation_&_sampling.ipynb`
2. Continue through notebooks 01-06 in numerical order
3. Explore pretrained weights in notebooks 07-10
4. Run all cells sequentially within each notebook

Run all cells in order to see the implementations and outputs. Code is heavily commented for clarity.

> **Note:** The `gpt2/` folder containing pretrained weights is not included in the repository (too large for GitHub). Run notebook 09 or script 07 to download the weights automatically.

---

## Learning Path

**Beginner Track (6-8 hours)**

- Focus on understanding concepts
- Run all cells and observe outputs
- Modify hyperparameters to see effects
- Complete notebooks 01-04

**Intermediate Track (12-16 hours)**

- Study the implementation details
- Experiment with architecture variations
- Train the model (notebooks 05-06)
- Load and use pretrained weights (notebooks 08-09)
- Implement additional features (dropout, different attention patterns)

**Advanced Track (20+ hours)**

- Train the model on custom datasets
- Implement advanced techniques (flash attention, sparse attention)
- Fine-tune pretrained GPT-2 on domain-specific data
- Optimize for production deployment

---

## 🔑 Key Concepts Covered

| Concept            | Notebook | Description                                |
| ------------------ | -------- | ------------------------------------------ |
| Tokenization       | 01       | Word-level, BPE, special tokens            |
| Embeddings         | 02       | Token, positional, pretrained (Word2Vec)   |
| Attention          | 03       | Self-attention, multi-head, causal masking |
| Architecture       | 04, 10   | Transformer blocks, GPT model, layer norm  |
| Loss & Metrics     | 05       | Cross-entropy, perplexity, generation      |
| Training           | 06       | Optimization loop, validation, checkpoints |
| Pretrained Weights | 08, 09   | Loading OpenAI GPT-2 weights               |
| Data Loading       | 01       | PyTorch Dataset/DataLoader, batching       |

---

## 🛠️ Tech Stack

- **PyTorch**: Deep learning framework for model implementation
- **TensorFlow**: Loading OpenAI's pretrained GPT-2 checkpoints
- **tiktoken**: OpenAI's fast BPE tokenizer for GPT-style encoding
- **gensim**: Word2Vec pretrained embeddings
- **matplotlib**: Visualization for loss curves and attention patterns
- **kagglehub**: Dataset downloading and management
- **NumPy**: Numerical operations
- **Jupyter**: Interactive notebook development

---

## 🤝 Contributing

Contributions are welcome! Whether it's fixing bugs, improving documentation, or adding new features:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add some improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📖 Additional Resources

### 📚 Primary Reference

This project is heavily inspired by and based on:

- **[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)** by **Sebastian Raschka** - The primary resource for this repository. Most of the code and concepts are adapted from this excellent book. Highly recommended for anyone wanting to deeply understand LLMs!
- [Sebastian Raschka's GitHub](https://github.com/rasbt) - Author's GitHub with additional resources

### 📄 Research Papers & Guides

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal GPT implementation

---

## 📜 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Sugam**  
GitHub: [@sugam24](https://github.com/sugam24)

---

## ⭐ Acknowledgments

- **Sebastian Raschka** - This project is based on his book _"Build a Large Language Model (From Scratch)"_. His clear explanations and well-structured code made understanding LLMs accessible. Most of the implementations in this repository are adapted from his work.
- Inspired by modern LLM research and educational content
- Built on the foundations of PyTorch and the open-source ML community
- Special thanks to all contributors and learners who provide feedback

---

<div align="center">

**If you found this helpful, please consider giving it a ⭐!**

_Built with ❤️ for the ML community_

</div>
