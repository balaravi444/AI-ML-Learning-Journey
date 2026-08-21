"""
Day 81 — Transformers: How They Work
Topic: Self-Attention from Scratch
Date: 07 August 2026
Author: Bala Ravi

Building attention mechanism step by step.
This is what powers BERT, GPT, Claude!
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: np.ndarray = None) -> tuple:
    """
    Scaled Dot-Product Attention.

    The core of every Transformer!

    Attention(Q, K, V) = softmax(Q @ K.T / √d_k) @ V

    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        mask: Optional mask (for decoder)

    Returns:
        (output, attention_weights)
    """
    d_k = Q.shape[-1]

    # Step 1: Compute raw attention scores
    # scores[i][j] = how much token i attends to j
    scores = Q @ K.T / np.sqrt(d_k)

    # Step 2: Apply mask if provided (decoder)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 3: Softmax → attention weights
    # Each row sums to 1
    attention_weights = softmax(scores)

    # Step 4: Weighted sum of values
    output = attention_weights @ V

    return output, attention_weights


def demonstrate_self_attention() -> None:
    """Show self-attention step by step."""
    print("=== Self-Attention Step by Step ===\n")

    # Simulate: "The cat sat"
    # 3 tokens, d_model = 4
    np.random.seed(42)
    tokens = ["The", "cat", "sat"]
    d_model = 4
    d_k = 4

    # Token embeddings (normally learned)
    X = np.array([
        [0.1, 0.2, 0.3, 0.4],  # "The"
        [0.8, 0.7, 0.6, 0.5],  # "cat"
        [0.4, 0.5, 0.6, 0.7]   # "sat"
    ])

    print(f"Input tokens: {tokens}")
    print(f"Embeddings shape: {X.shape}\n")

    # Weight matrices (normally learned by training)
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1

    # Step 1: Compute Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    print(f"Q shape: {Q.shape} (queries)")
    print(f"K shape: {K.shape} (keys)")
    print(f"V shape: {V.shape} (values)\n")

    # Step 2: Attention
    output, weights = scaled_dot_product_attention(
        Q, K, V)

    print(f"Attention Weights (each row sums to 1):")
    print(f"{'':>8}", end='')
    for t in tokens:
        print(f"{t:>8}", end='')
    print()
    print("-" * 32)

    for i, token in enumerate(tokens):
        print(f"{token:>8}", end='')
        for j in range(len(tokens)):
            print(f"{weights[i][j]:>8.4f}", end='')
        print()

    print(f"\nOutput shape: {output.shape}")
    print(f"\n💡 Each row = how much that token")
    print(f"   attends to every other token!")
    print(f"   'cat' strongly attends to 'cat' (self)")
    print(f"   and also to 'sat' (what it did)! 🔥")


def multi_head_attention(
        X: np.ndarray,
        n_heads: int = 2,
        d_k: int = 4) -> np.ndarray:
    """
    Multi-Head Attention.

    Run attention h times in parallel!
    Each head learns different relationships.

    Args:
        X: Input (seq_len, d_model)
        n_heads: Number of attention heads
        d_k: Dimension per head

    Returns:
        Multi-head attention output
    """
    seq_len, d_model = X.shape
    head_outputs = []

    for h in range(n_heads):
        np.random.seed(h * 10)
        # Each head has its own weight matrices
        W_Q = np.random.randn(d_model, d_k) * 0.1
        W_K = np.random.randn(d_model, d_k) * 0.1
        W_V = np.random.randn(d_model, d_k) * 0.1

        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V

        head_out, weights = (
            scaled_dot_product_attention(Q, K, V))
        head_outputs.append(head_out)

    # Concatenate heads
    concat = np.concatenate(
        head_outputs, axis=-1)

    # Final linear projection
    np.random.seed(99)
    W_O = np.random.randn(
        n_heads * d_k, d_model) * 0.1
    output = concat @ W_O

    return output


def positional_encoding(
        seq_len: int,
        d_model: int) -> np.ndarray:
    """
    Sinusoidal Positional Encoding.

    Without this: Transformer is order-agnostic!
    "cat sat" = "sat cat" to the model!

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Sequence length
        d_model: Model dimension

    Returns:
        Positional encoding matrix (seq_len, d_model)
    """
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            denominator = 10000 ** (
                2 * i / d_model)
            PE[pos, i] = np.sin(
                pos / denominator)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(
                    pos / denominator)

    return PE


def demonstrate_positional_encoding() -> None:
    """Show why positional encoding matters."""
    print("\n=== Positional Encoding ===\n")

    seq_len = 5
    d_model = 8

    PE = positional_encoding(seq_len, d_model)

    print(f"Positional Encoding shape: {PE.shape}")
    print(f"(seq_len={seq_len}, d_model={d_model})\n")

    print("Each position has a unique pattern:")
    for pos in range(seq_len):
        pattern = ' '.join(
            [f"{v:>6.3f}" for v in PE[pos]])
        print(f"  Position {pos}: [{pattern}]")

    print(f"\n💡 Each position gets unique vector!")
    print(f"   Added to word embeddings.")
    print(f"   Model learns to use position info!")
    print(f"   'cat sat' ≠ 'sat cat' now! ✅")


def bert_vs_gpt_architecture() -> None:
    """Explain BERT vs GPT architectures."""
    print("\n=== BERT vs GPT Architecture ===\n")

    architectures = {
        'BERT (Encoder)': {
            'attention': 'Bidirectional (sees all tokens)',
            'training': 'Masked Language Model (MLM)',
            'example': '"The [MASK] sat on mat"',
            'best_for': [
                'Text classification',
                'Named Entity Recognition',
                'Question Answering',
                'Sentence embeddings'
            ],
            'models': ['BERT', 'RoBERTa',
                        'DistilBERT', 'ALBERT']
        },
        'GPT (Decoder)': {
            'attention': 'Causal (only sees left context)',
            'training': 'Next token prediction',
            'example': '"The cat sat on the..."',
            'best_for': [
                'Text generation',
                'Chatbots',
                'Code generation',
                'Creative writing'
            ],
            'models': ['GPT-2', 'GPT-3',
                        'GPT-4', 'LLaMA', 'Claude']
        }
    }

    for name, info in architectures.items():
        print(f"  {name}:")
        print(f"    Attention:  {info['attention']}")
        print(f"    Training:   {info['training']}")
        print(f"    Example:    {info['example']}")
        print(f"    Best for:")
        for use in info['best_for']:
            print(f"      → {use}")
        print(f"    Models: "
              f"{', '.join(info['models'])}")
        print()

    print("💡 For MemoryOS (Day 87):")
    print("   sentence-transformers (BERT-based)")
    print("   → encode documents to vectors")
    print("   Gemini API (GPT-like)")
    print("   → generate answers from context! 🔥")


if __name__ == "__main__":
    demonstrate_self_attention()

    print("\n=== Multi-Head Attention Demo ===\n")
    np.random.seed(42)
    X = np.random.randn(3, 8)
    output = multi_head_attention(
        X, n_heads=2, d_k=4)
    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✅ Multi-head attention preserves shape!")

    demonstrate_positional_encoding()
    bert_vs_gpt_architecture()
