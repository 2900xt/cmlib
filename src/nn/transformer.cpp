#include "nn/transformer.h"
#include <cmath>

static void append(std::vector<Tensor> &out, const std::vector<Tensor> &params)
{
    out.insert(out.end(), params.begin(), params.end());
}

// MultiHeadAttention ---------------------------------------------------------

MultiHeadAttention::MultiHeadAttention(int d_model, int n_heads, bool causal)
    : d_model(d_model), n_heads(n_heads), d_head(d_model / n_heads), causal(causal),
      qkv(d_model, 3 * d_model), proj(d_model, d_model)
{
    if(d_model % n_heads != 0)
    {
        std::cerr << "d_model (" << d_model << ") must be divisible by n_heads (" << n_heads << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

Tensor MultiHeadAttention::forward(const Tensor &x)
{
    Tensor qkv_out = qkv(x);    // (T, 3 * d_model)
    FP_DTYPE scale = 1 / std::sqrt((FP_DTYPE)d_head);

    std::vector<Tensor> heads;
    for(int h = 0; h < n_heads; h++)
    {
        Tensor q = slice(qkv_out, h * d_head, (h + 1) * d_head);
        Tensor k = slice(qkv_out, d_model + h * d_head, d_model + (h + 1) * d_head);
        Tensor v = slice(qkv_out, 2 * d_model + h * d_head, 2 * d_model + (h + 1) * d_head);

        Tensor scores = matmul(q, transpose(k)) * scale;   // (T, T)
        if(causal) scores = causal_mask(scores);
        heads.push_back(matmul(softmax(scores), v));       // (T, d_head)
    }
    return proj(concat(heads));
}

std::vector<Tensor> MultiHeadAttention::parameters() const
{
    std::vector<Tensor> out = qkv.parameters();
    append(out, proj.parameters());
    return out;
}

// TransformerBlock -----------------------------------------------------------

TransformerBlock::TransformerBlock(int d_model, int n_heads, int d_ff, bool causal)
    : ln1(d_model), ln2(d_model), attn(d_model, n_heads, causal),
      mlp({layer<Linear>(d_model, d_ff), layer<GELU>(), layer<Linear>(d_ff, d_model)}) {}

Tensor TransformerBlock::forward(const Tensor &x)
{
    Tensor h = x + attn(ln1(x));
    return h + mlp(ln2(h));
}

std::vector<Tensor> TransformerBlock::parameters() const
{
    std::vector<Tensor> out = ln1.parameters();
    append(out, attn.parameters());
    append(out, ln2.parameters());
    append(out, mlp.parameters());
    return out;
}

// TransformerLM --------------------------------------------------------------

TransformerLM::TransformerLM(int vocab_size, int max_len, int d_model, int n_heads, int n_layers, int d_ff)
    : vocab_size(vocab_size), max_len(max_len), token_emb(vocab_size, d_model), pos_emb(max_len, d_model),
      ln_f(d_model), head(d_model, vocab_size)
{
    for(int i = 0; i < n_layers; i++)
    {
        blocks.push_back(std::make_shared<TransformerBlock>(d_model, n_heads, d_ff, true));
    }
}

Tensor TransformerLM::forward(const std::vector<int> &tokens)
{
    if(tokens.empty() || (int)tokens.size() > max_len)
    {
        std::cerr << "TransformerLM: sequence length " << tokens.size() << " must be in [1, " << max_len << "]" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<int> positions(tokens.size());
    for(size_t i = 0; i < tokens.size(); i++) positions[i] = i;

    Tensor x = token_emb(tokens) + pos_emb(positions);
    for(size_t i = 0; i < blocks.size(); i++) x = blocks[i]->forward(x);
    return head(ln_f(x));
}

std::vector<int> TransformerLM::generate(std::vector<int> prompt, int n, FP_DTYPE temperature)
{
    NoGradGuard no_grad;
    for(int step = 0; step < n; step++)
    {
        int start = std::max(0, (int)prompt.size() - max_len);
        std::vector<int> context(prompt.begin() + start, prompt.end());
        Tensor logits = forward(context);

        int T = logits.dim(0);
        Vector last(logits.data().begin() + (T - 1) * vocab_size, logits.data().begin() + T * vocab_size);

        int next = 0;
        if(temperature <= 0)
        {
            for(int j = 1; j < vocab_size; j++) if(last[j] > last[next]) next = j;
        }
        else
        {
            Tensor probs = softmax(Tensor({vocab_size}, last) / temperature);
            FP_DTYPE r = (FP_DTYPE)std::rand() / RAND_MAX, acc = 0;
            next = vocab_size - 1;
            for(int j = 0; j < vocab_size; j++)
            {
                acc += probs[j];
                if(r <= acc) { next = j; break; }
            }
        }
        prompt.push_back(next);
    }
    return prompt;
}

std::vector<Tensor> TransformerLM::parameters() const
{
    std::vector<Tensor> out = token_emb.parameters();
    append(out, pos_emb.parameters());
    for(size_t i = 0; i < blocks.size(); i++) append(out, blocks[i]->parameters());
    append(out, ln_f.parameters());
    append(out, head.parameters());
    return out;
}
