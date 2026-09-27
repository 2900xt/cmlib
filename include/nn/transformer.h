#pragma once

#include "nn/nn.h"

// Multi-head scaled dot-product self attention over a (T, d_model) sequence
class MultiHeadAttention : public Layer
{
public:
    int d_model, n_heads, d_head;
    bool causal;
    Linear qkv, proj;
    MultiHeadAttention(int d_model, int n_heads, bool causal = true);
    Tensor forward(const Tensor &x) override;
    std::vector<Tensor> parameters() const override;
};

// Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))
class TransformerBlock : public Layer
{
public:
    LayerNorm ln1, ln2;
    MultiHeadAttention attn;
    Sequential mlp;
    TransformerBlock(int d_model, int n_heads, int d_ff, bool causal = true);
    Tensor forward(const Tensor &x) override;
    std::vector<Tensor> parameters() const override;
};

// Decoder-only (GPT style) language model over integer tokens
class TransformerLM : public Module
{
public:
    int vocab_size, max_len;
    Embedding token_emb, pos_emb;
    std::vector<std::shared_ptr<TransformerBlock> > blocks;
    LayerNorm ln_f;
    Linear head;

    TransformerLM(int vocab_size, int max_len, int d_model = 32, int n_heads = 4, int n_layers = 2, int d_ff = 128);

    // tokens (length T <= max_len) -> logits of shape (T, vocab_size)
    Tensor forward(const std::vector<int> &tokens);

    // Autoregressively extends `prompt` by n tokens. temperature <= 0 means greedy decoding.
    std::vector<int> generate(std::vector<int> prompt, int n, FP_DTYPE temperature = 0);

    std::vector<Tensor> parameters() const override;
};
