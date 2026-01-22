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
├── 01. Data_preparation_&_sampling.ipynb    # Text → Tensors pipeline
├── 02. Vector_embedding.ipynb                # Semantic embeddings
├── 03. Attention_mechanism.ipynb             # Self-attention & multi-head
├── 04. LLM_architecture(GPT).ipynb          # Complete GPT model
│
└── README.md                                 # You are here
```

---

## ✨ What You'll Learn

- **Text Processing Pipeline**: Tokenization strategies, vocabulary construction, and BPE encoding
- **Embedding Techniques**: Token embeddings, positional encodings, and semantic vector spaces
- **Attention Mechanisms**: Self-attention, multi-head attention, and causal masking
- **Transformer Architecture**: Complete GPT-style model implementation with all components
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

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch 1.12+
NumPy
Jupyter Notebook
tiktoken
gensim
matplotlib
```

### Installation

```bash
# Clone the repository
git clone https://github.com/sugam24/LLM-From-Scratch.git
cd LLM-From-Scratch

# Install dependencies
pip install torch numpy jupyter tiktoken gensim matplotlib

# Launch Jupyter
jupyter notebook
```

### Usage

Navigate through the notebooks sequentially (01 → 04). Each notebook is self-contained but builds conceptually on previous ones:

```bash
# Start with notebook 01
jupyter notebook "01. Data_preparation_&_sampling.ipynb"
```

Run all cells in order to see the implementations and outputs. Code is heavily commented for clarity.

---

## Learning Path

**Beginner Track (4-6 hours)**

- Focus on understanding concepts
- Run all cells and observe outputs
- Modify hyperparameters to see effects

**Intermediate Track (8-12 hours)**

- Study the implementation details
- Experiment with architecture variations
- Implement additional features (dropout, different attention patterns)

**Advanced Track (15+ hours)**

- Train the model on custom datasets
- Implement advanced techniques (flash attention, sparse attention)
- Optimize for production deployment

---

## 🔑 Key Concepts Covered

| Concept      | Notebook | Description                                |
| ------------ | -------- | ------------------------------------------ |
| Tokenization | 01       | Word-level, BPE, special tokens            |
| Embeddings   | 01, 02   | Token, positional, pretrained (Word2Vec)   |
| Attention    | 03       | Self-attention, multi-head, causal masking |
| Architecture | 04       | Transformer blocks, GPT model, layer norm  |
| Data Loading | 01       | PyTorch Dataset/DataLoader, batching       |

---

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **tiktoken**: OpenAI's fast BPE tokenizer
- **gensim**: Word2Vec pretrained embeddings
- **NumPy**: Numerical operations
- **Jupyter**: Interactive development

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

- Inspired by modern LLM research and educational content
- Built on the foundations of PyTorch and the open-source ML community
- Special thanks to all contributors and learners who provide feedback

---

<div align="center">

**If you found this helpful, please consider giving it a ⭐!**

_Built with ❤️ for the ML community_

</div>
