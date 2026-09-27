#pragma once

#include "tensor/tensor.h"
#include <memory>
#include <initializer_list>

// Anything that owns trainable parameters
class Module
{
public:
    virtual ~Module() {}
    virtual std::vector<Tensor> parameters() const { return {}; }
    void zero_grad() const;
    int num_parameters() const;
};

// A module that maps one tensor to another
class Layer : public Module
{
public:
    virtual Tensor forward(const Tensor &x) = 0;
    Tensor operator()(const Tensor &x) { return forward(x); }
};

// y = x W + b, with W of shape (in, out)
class Linear : public Layer
{
public:
    Tensor weight, bias;
    Linear(int in_features, int out_features, bool use_bias = true);
    Tensor forward(const Tensor &x) override;
    std::vector<Tensor> parameters() const override;
};

class ReLU : public Layer { public: Tensor forward(const Tensor &x) override { return relu(x); } };
class GELU : public Layer { public: Tensor forward(const Tensor &x) override { return gelu(x); } };
class Tanh : public Layer { public: Tensor forward(const Tensor &x) override { return tanh(x); } };
class Sigmoid : public Layer { public: Tensor forward(const Tensor &x) override { return sigmoid(x); } };

// Runs layers one after another
class Sequential : public Layer
{
public:
    std::vector<std::shared_ptr<Layer> > layers;
    Sequential() {}
    Sequential(std::initializer_list<std::shared_ptr<Layer> > layers);
    Sequential &add(std::shared_ptr<Layer> layer);
    Tensor forward(const Tensor &x) override;
    std::vector<Tensor> parameters() const override;
};

// Normalizes over the last axis, then scales and shifts
class LayerNorm : public Layer
{
public:
    Tensor gamma, beta;
    FP_DTYPE eps;
    LayerNorm(int features, FP_DTYPE eps = 1e-5);
    Tensor forward(const Tensor &x) override;
    std::vector<Tensor> parameters() const override;
};

// Lookup table: token ids -> (n, dim) rows
class Embedding : public Module
{
public:
    Tensor weight;
    Embedding(int num_embeddings, int dim);
    Tensor forward(const std::vector<int> &ids) const;
    Tensor operator()(const std::vector<int> &ids) const { return forward(ids); }
    std::vector<Tensor> parameters() const override;
};

// Shorthand for building layers inside Sequential{...}
template<typename T, typename... Args>
std::shared_ptr<Layer> layer(Args... args) { return std::make_shared<T>(args...); }
