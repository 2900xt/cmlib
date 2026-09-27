#pragma once

#include "tensor/tensor.h"

class Optimizer
{
public:
    std::vector<Tensor> params;
    FP_DTYPE lr;
    Optimizer(const std::vector<Tensor> &params, FP_DTYPE lr) : params(params), lr(lr) {}
    virtual ~Optimizer() {}
    virtual void step() = 0;
    void zero_grad();
};

// Stochastic gradient descent with optional momentum
class SGD : public Optimizer
{
public:
    FP_DTYPE momentum;
    std::vector<Vector> velocity;
    SGD(const std::vector<Tensor> &params, FP_DTYPE lr, FP_DTYPE momentum = 0);
    void step() override;
};

// Adam (with decoupled weight decay, i.e. AdamW, when weight_decay > 0)
class Adam : public Optimizer
{
public:
    FP_DTYPE beta1, beta2, eps, weight_decay;
    int t;
    std::vector<Vector> m, v;
    Adam(const std::vector<Tensor> &params, FP_DTYPE lr = 1e-3, FP_DTYPE beta1 = 0.9,
         FP_DTYPE beta2 = 0.999, FP_DTYPE eps = 1e-8, FP_DTYPE weight_decay = 0);
    void step() override;
};
