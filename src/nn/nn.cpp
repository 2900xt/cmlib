#include "nn/nn.h"
#include <cmath>

void Module::zero_grad() const
{
    std::vector<Tensor> params = parameters();
    for(size_t i = 0; i < params.size(); i++) params[i].zero_grad();
}

int Module::num_parameters() const
{
    std::vector<Tensor> params = parameters();
    int n = 0;
    for(size_t i = 0; i < params.size(); i++) n += params[i].size();
    return n;
}

static void append(std::vector<Tensor> &out, const std::vector<Tensor> &params)
{
    out.insert(out.end(), params.begin(), params.end());
}

// Linear ---------------------------------------------------------------------

Linear::Linear(int in_features, int out_features, bool use_bias)
{
    // Xavier/Glorot uniform initialization
    FP_DTYPE limit = std::sqrt(6.0L / (in_features + out_features));
    weight = Tensor::rand({in_features, out_features}, -limit, limit, true);
    if(use_bias) bias = Tensor::zeros({out_features}, true);
}

Tensor Linear::forward(const Tensor &x)
{
    Tensor out = matmul(x, weight);
    if(bias.defined()) out = out + bias;
    return out;
}

std::vector<Tensor> Linear::parameters() const
{
    if(bias.defined()) return {weight, bias};
    return {weight};
}

// Sequential -----------------------------------------------------------------

Sequential::Sequential(std::initializer_list<std::shared_ptr<Layer> > layers) : layers(layers) {}

Sequential &Sequential::add(std::shared_ptr<Layer> layer)
{
    layers.push_back(layer);
    return *this;
}

Tensor Sequential::forward(const Tensor &x)
{
    Tensor out = x;
    for(size_t i = 0; i < layers.size(); i++) out = layers[i]->forward(out);
    return out;
}

std::vector<Tensor> Sequential::parameters() const
{
    std::vector<Tensor> out;
    for(size_t i = 0; i < layers.size(); i++) append(out, layers[i]->parameters());
    return out;
}

// LayerNorm ------------------------------------------------------------------

LayerNorm::LayerNorm(int features, FP_DTYPE eps)
    : gamma(Tensor::ones({features}, true)), beta(Tensor::zeros({features}, true)), eps(eps) {}

Tensor LayerNorm::forward(const Tensor &x)
{
    Tensor centered = x - mean(x, -1);
    Tensor var = mean(pow(centered, 2), -1);
    return centered / sqrt(var + eps) * gamma + beta;
}

std::vector<Tensor> LayerNorm::parameters() const { return {gamma, beta}; }

// Embedding ------------------------------------------------------------------

Embedding::Embedding(int num_embeddings, int dim)
    : weight(Tensor::randn({num_embeddings, dim}, 0, 0.1L, true)) {}

Tensor Embedding::forward(const std::vector<int> &ids) const { return gather_rows(weight, ids); }

std::vector<Tensor> Embedding::parameters() const { return {weight}; }
