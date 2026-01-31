"""
GPT-2 Model Architecture Implementation
=======================================

This module provides a complete, scalable implementation of the GPT-2 architecture.
It supports all GPT-2 model variants: Small (124M), Medium (355M), Large (774M), and XL (1558M).

Based on the concepts and implementations from the LLM-From-Scratch notebooks:
- Notebook 03: Attention mechanism (MultiHeadAttention)
- Notebook 04: LLM architecture (GPT model components)
- Notebook 08: Understanding GPT-2 weights

Usage:
    from GPT2_architecture import GPTModel, get_config
    
    # Get configuration for desired model size
    config = get_config("gpt2-small")  # or "gpt2-medium", "gpt2-large", "gpt2-xl"
    
    # Create model
    model = GPTModel(config)
    
    # For loading pretrained weights, see notebook 08

Author: LLM-From-Scratch Project
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Literal


# =============================================================================
# GPT-2 Model Configurations
# =============================================================================

GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size (GPT-2 BPE tokenizer)
    "context_length": 1024,   # Maximum context length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of transformer blocks
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias (GPT-2 doesn't use bias)
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,          # Larger embedding dimension
    "n_heads": 16,            # More attention heads
    "n_layers": 24,           # More layers
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG_774M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,          # Even larger embedding
    "n_heads": 20,            # More heads
    "n_layers": 36,           # More layers
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG_1558M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1600,          # Largest embedding
    "n_heads": 25,            # Most heads
    "n_layers": 48,           # Most layers
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Model size mapping
MODEL_CONFIGS = {
    "gpt2-small": GPT_CONFIG_124M,
    "gpt2-medium": GPT_CONFIG_355M,
    "gpt2-large": GPT_CONFIG_774M,
    "gpt2-xl": GPT_CONFIG_1558M,
    # Aliases
    "124M": GPT_CONFIG_124M,
    "355M": GPT_CONFIG_355M,
    "774M": GPT_CONFIG_774M,
    "1558M": GPT_CONFIG_1558M,
}


def get_config(model_name: str = "gpt2-small") -> Dict:
    """
    Get the configuration dictionary for a specific GPT-2 model size.
    
    Args:
        model_name: One of "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"
                    or "124M", "355M", "774M", "1558M"
    
    Returns:
        Configuration dictionary with model hyperparameters
    
    Example:
        >>> config = get_config("gpt2-small")
        >>> print(config["n_layers"])
        12
    """
    if model_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return MODEL_CONFIGS[model_name].copy()


def print_model_comparison():
    """Print a comparison table of all GPT-2 model configurations."""
    print("\n" + "=" * 70)
    print("GPT-2 Model Sizes Comparison")
    print("=" * 70)
    print(f"{'Model':<15} {'Params':<15} {'Layers':<10} {'Heads':<10} {'Emb Dim':<10}")
    print("-" * 70)
    
    model_info = [
        ("GPT-2 Small", "124M", GPT_CONFIG_124M),
        ("GPT-2 Medium", "355M", GPT_CONFIG_355M),
        ("GPT-2 Large", "774M", GPT_CONFIG_774M),
        ("GPT-2 XL", "1558M", GPT_CONFIG_1558M),
    ]
    
    for name, params, cfg in model_info:
        print(f"{name:<15} {params:<15} {cfg['n_layers']:<10} {cfg['n_heads']:<10} {cfg['emb_dim']:<10}")
    print("=" * 70 + "\n")


# =============================================================================
# GELU Activation Function
# =============================================================================

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.
    
    This is the activation function used in GPT-2, which provides smoother
    gradients compared to ReLU and allows for better training dynamics.
    
    Uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# =============================================================================
# Layer Normalization
# =============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization as used in GPT-2.
    
    Normalizes across the embedding dimension with learnable scale and shift parameters.
    Uses biased variance estimation (divides by n, not n-1) for compatibility with
    TensorFlow's default behavior used in original GPT-2.
    
    Args:
        emb_dim: The embedding dimension to normalize over
        eps: Small constant for numerical stability (default: 1e-5)
    """
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))   # gamma (g in TF)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # beta (b in TF)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # Biased variance like TF
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# =============================================================================
# Feed Forward Network
# =============================================================================

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network used in each transformer block.
    
    Architecture: Linear → GELU → Linear
    The hidden dimension is 4× the embedding dimension (GPT-2 design choice).
    
    Args:
        cfg: Configuration dictionary with 'emb_dim' key
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expand to 4x
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # Project back
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# =============================================================================
# Multi-Head Attention
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention as used in GPT-2.
    
    Implements scaled dot-product attention with causal masking to prevent
    attending to future tokens. Uses combined QKV projection for efficiency.
    
    Args:
        d_in: Input dimension
        d_out: Output dimension (must be divisible by num_heads)
        context_length: Maximum sequence length for causal mask
        dropout: Dropout probability
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projections (False for GPT-2)
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # Separate projections for Q, K, V (can also be combined into one)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask: upper triangular matrix of ones
        # This prevents attending to future positions
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, d_in = x.shape
        
        # Project to Q, K, V
        queries = self.W_query(x)  # (batch, tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Reshape for multi-head attention
        # (batch, tokens, d_out) → (batch, tokens, heads, head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose to (batch, heads, tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)  # (batch, heads, tokens, tokens)
        
        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        context = (attn_weights @ values).transpose(1, 2)  # (batch, tokens, heads, head_dim)
        
        # Concatenate heads and project
        context = context.contiguous().view(batch_size, num_tokens, self.d_out)
        output = self.out_proj(context)
        
        return output


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block as used in GPT-2.
    
    Architecture (Pre-LayerNorm):
        x → LayerNorm → MultiHeadAttention → Dropout → + (residual)
        x → LayerNorm → FeedForward → Dropout → + (residual)
    
    Args:
        cfg: Configuration dictionary
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        
        # Multi-head attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        
        # Feed-forward network
        self.ff = FeedForward(cfg)
        
        # Layer normalization (Pre-LN architecture)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        # Dropout for residual connections
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        # Feed-forward block with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x


# =============================================================================
# GPT Model
# =============================================================================

class GPTModel(nn.Module):
    """
    Complete GPT-2 Model Implementation.
    
    Architecture:
        Input Token IDs
            ↓
        Token Embeddings + Position Embeddings
            ↓
        Dropout
            ↓
        N × Transformer Blocks
            ↓
        Final Layer Norm
            ↓
        Output Head (Linear projection to vocabulary)
            ↓
        Logits
    
    Args:
        cfg: Configuration dictionary containing:
            - vocab_size: Size of vocabulary
            - context_length: Maximum sequence length
            - emb_dim: Embedding dimension
            - n_heads: Number of attention heads
            - n_layers: Number of transformer blocks
            - drop_rate: Dropout probability
            - qkv_bias: Whether to use bias in attention
    
    Example:
        >>> config = get_config("gpt2-small")
        >>> model = GPTModel(config)
        >>> input_ids = torch.randint(0, 50257, (2, 128))  # batch=2, seq_len=128
        >>> logits = model(input_ids)
        >>> print(logits.shape)  # torch.Size([2, 128, 50257])
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        
        self.cfg = cfg
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Stack of transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # Final layer normalization
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        # Output projection to vocabulary
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT model.
        
        Args:
            in_idx: Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = in_idx.shape
        
        # Get token embeddings
        tok_embeds = self.tok_emb(in_idx)
        
        # Get position embeddings
        positions = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(positions)
        
        # Combine embeddings
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # Pass through transformer blocks
        x = self.trf_blocks(x)
        
        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get the total number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters
                          (for weight-tied model comparison)
        
        Returns:
            Total number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_emb.weight.numel()
            n_params -= self.pos_emb.weight.numel()
        return n_params
    
    def get_memory_footprint(self) -> str:
        """Get the memory footprint of the model in MB."""
        total_params = self.get_num_params()
        size_bytes = total_params * 4  # Assuming float32
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"


# =============================================================================
# Weight Loading Utilities
# =============================================================================

def load_weights_from_gpt2(model: GPTModel, params: Dict, verbose: bool = True) -> None:
    """
    Load weights from GPT-2 TensorFlow checkpoint into PyTorch model.
    
    This function handles the weight format conversion between TensorFlow and PyTorch:
    - TensorFlow Linear: weight shape = (input_dim, output_dim)
    - PyTorch Linear: weight shape = (output_dim, input_dim)
    - Therefore, we need to transpose weight matrices!
    
    Args:
        model: The GPTModel instance to load weights into
        params: Dictionary of weights loaded from TensorFlow checkpoint
                (use load_gpt2_params_from_tf_ckpt from notebook 08)
        verbose: Whether to print loading progress
    
    Example:
        >>> # First load params using the function from notebook 08
        >>> params = load_gpt2_params_from_tf_ckpt(ckpt_path, hparams)
        >>> # Then load into PyTorch model
        >>> model = GPTModel(get_config("gpt2-small"))
        >>> load_weights_from_gpt2(model, params)
    """
    def assign(layer, key_w, key_b=None, transpose=True):
        """Helper to assign weights to a layer."""
        if key_w in params or (isinstance(params.get(key_w), dict) and 'w' in params.get(key_w, {})):
            weight_dict = params[key_w] if isinstance(params[key_w], dict) else {'w': params[key_w]}
            weight = weight_dict.get('w', weight_dict)
            
            if hasattr(weight, 'shape') and transpose and len(weight.shape) == 2:
                weight = weight.T
            
            if hasattr(layer, 'weight'):
                layer.weight.data = torch.from_numpy(weight.copy()).float()
            
            if key_b and 'b' in weight_dict:
                layer.bias.data = torch.from_numpy(weight_dict['b'].copy()).float()
    
    if verbose:
        print("Loading GPT-2 weights into PyTorch model...")
    
    # Load token and position embeddings (no transpose needed)
    model.tok_emb.weight.data = torch.from_numpy(params['wte'].copy()).float()
    model.pos_emb.weight.data = torch.from_numpy(params['wpe'].copy()).float()
    
    # Load transformer blocks
    for i, block in enumerate(model.trf_blocks):
        block_params = params['blocks'][i]
        
        # Layer norms
        block.norm1.scale.data = torch.from_numpy(block_params['ln_1']['g'].copy()).float()
        block.norm1.shift.data = torch.from_numpy(block_params['ln_1']['b'].copy()).float()
        block.norm2.scale.data = torch.from_numpy(block_params['ln_2']['g'].copy()).float()
        block.norm2.shift.data = torch.from_numpy(block_params['ln_2']['b'].copy()).float()
        
        # Attention weights (need transpose)
        # GPT-2 uses combined c_attn for Q, K, V
        c_attn_w = block_params['attn']['c_attn']['w']  # Shape: (emb, 3*emb)
        c_attn_b = block_params['attn']['c_attn']['b']  # Shape: (3*emb,)
        
        # Split into Q, K, V
        emb_dim = c_attn_w.shape[0]
        q_w, k_w, v_w = c_attn_w[:, :emb_dim], c_attn_w[:, emb_dim:2*emb_dim], c_attn_w[:, 2*emb_dim:]
        q_b, k_b, v_b = c_attn_b[:emb_dim], c_attn_b[emb_dim:2*emb_dim], c_attn_b[2*emb_dim:]
        
        # Assign to model (transpose for PyTorch)
        block.att.W_query.weight.data = torch.from_numpy(q_w.T.copy()).float()
        block.att.W_key.weight.data = torch.from_numpy(k_w.T.copy()).float()
        block.att.W_value.weight.data = torch.from_numpy(v_w.T.copy()).float()
        
        # Biases (if present - GPT-2 doesn't use them but handle just in case)
        if model.cfg.get("qkv_bias", False):
            block.att.W_query.bias.data = torch.from_numpy(q_b.copy()).float()
            block.att.W_key.bias.data = torch.from_numpy(k_b.copy()).float()
            block.att.W_value.bias.data = torch.from_numpy(v_b.copy()).float()
        
        # Output projection
        c_proj_w = block_params['attn']['c_proj']['w']
        c_proj_b = block_params['attn']['c_proj']['b']
        block.att.out_proj.weight.data = torch.from_numpy(c_proj_w.T.copy()).float()
        block.att.out_proj.bias.data = torch.from_numpy(c_proj_b.copy()).float()
        
        # MLP weights
        c_fc_w = block_params['mlp']['c_fc']['w']
        c_fc_b = block_params['mlp']['c_fc']['b']
        block.ff.layers[0].weight.data = torch.from_numpy(c_fc_w.T.copy()).float()
        block.ff.layers[0].bias.data = torch.from_numpy(c_fc_b.copy()).float()
        
        mlp_proj_w = block_params['mlp']['c_proj']['w']
        mlp_proj_b = block_params['mlp']['c_proj']['b']
        block.ff.layers[2].weight.data = torch.from_numpy(mlp_proj_w.T.copy()).float()
        block.ff.layers[2].bias.data = torch.from_numpy(mlp_proj_b.copy()).float()
        
        if verbose:
            print(f"  Loaded block {i}")
    
    # Final layer norm
    model.final_norm.scale.data = torch.from_numpy(params['ln_f']['g'].copy()).float()
    model.final_norm.shift.data = torch.from_numpy(params['ln_f']['b'].copy()).float()
    
    # Output head uses same weights as token embedding (weight tying in original GPT-2)
    # For our implementation, we keep them separate but can optionally tie them
    model.out_head.weight.data = torch.from_numpy(params['wte'].copy()).float()
    
    if verbose:
        print("✅ All weights loaded successfully!")


# =============================================================================
# Text Generation Utilities
# =============================================================================

def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None
) -> torch.Tensor:
    """
    Generate text tokens using the model.
    
    Args:
        model: The GPT model
        idx: Initial token indices of shape (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate
        context_size: Maximum context length the model supports
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top-k most likely tokens
        eos_id: End-of-sequence token ID to stop generation
    
    Returns:
        Generated token indices of shape (batch_size, seq_len + generated)
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        # Crop context to maximum length
        idx_cond = idx[:, -context_size:]
        
        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus on last token
        logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sample next token
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Check for EOS
        if eos_id is not None and (idx_next == eos_id).all():
            break
    
    return idx


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-2 Architecture Implementation")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-small",
        choices=["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl", "124M", "355M", "774M", "1558M"],
        help="Model size to create"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print model comparison table"
    )
    
    args = parser.parse_args()
    
    if args.info:
        print_model_comparison()
    else:
        # Create and display model info
        print(f"\n🔧 Creating GPT-2 model: {args.model}")
        print("-" * 50)
        
        config = get_config(args.model)
        model = GPTModel(config)
        
        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        total_params = model.get_num_params()
        non_emb_params = model.get_num_params(non_embedding=True)
        
        print(f"\n📊 Parameter Count:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-embedding params: {non_emb_params:,}")
        print(f"  Memory footprint: {model.get_memory_footprint()}")
        
        # Test forward pass
        print(f"\n🧪 Testing forward pass...")
        test_input = torch.randint(0, config["vocab_size"], (2, 64))
        with torch.no_grad():
            output = model(test_input)
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"\n✅ Model created successfully!")
