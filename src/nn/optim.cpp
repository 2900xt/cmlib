#include "nn/optim.h"
#include <cmath>

void Optimizer::zero_grad()
{
    for(size_t i = 0; i < params.size(); i++) params[i].zero_grad();
}

SGD::SGD(const std::vector<Tensor> &params, FP_DTYPE lr, FP_DTYPE momentum)
    : Optimizer(params, lr), momentum(momentum)
{
    for(size_t i = 0; i < params.size(); i++) velocity.push_back(Vector(params[i].size(), 0));
}

void SGD::step()
{
    for(size_t p = 0; p < params.size(); p++)
    {
        Vector &w = params[p].data(), &g = params[p].grad(), &vel = velocity[p];
        for(size_t i = 0; i < w.size(); i++)
        {
            vel[i] = momentum * vel[i] + g[i];
            w[i] -= lr * vel[i];
        }
    }
}

Adam::Adam(const std::vector<Tensor> &params, FP_DTYPE lr, FP_DTYPE beta1, FP_DTYPE beta2, FP_DTYPE eps, FP_DTYPE weight_decay)
    : Optimizer(params, lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay), t(0)
{
    for(size_t i = 0; i < params.size(); i++)
    {
        m.push_back(Vector(params[i].size(), 0));
        v.push_back(Vector(params[i].size(), 0));
    }
}

void Adam::step()
{
    t++;
    FP_DTYPE c1 = 1 - std::pow(beta1, (FP_DTYPE)t), c2 = 1 - std::pow(beta2, (FP_DTYPE)t);
    for(size_t p = 0; p < params.size(); p++)
    {
        Vector &w = params[p].data(), &g = params[p].grad();
        for(size_t i = 0; i < w.size(); i++)
        {
            m[p][i] = beta1 * m[p][i] + (1 - beta1) * g[i];
            v[p][i] = beta2 * v[p][i] + (1 - beta2) * g[i] * g[i];
            FP_DTYPE m_hat = m[p][i] / c1, v_hat = v[p][i] / c2;
            w[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * w[i]);
        }
    }
}
