#pragma once

#include "config.h"
#include "math/matrix.h"
#include <memory>
#include <functional>

typedef std::vector<int> Shape;

// Storage + autograd bookkeeping shared by every handle pointing at the same tensor
struct TensorNode
{
    Vector data;                // row-major values
    Vector grad;                // d(loss)/d(data), allocated lazily
    Shape shape;
    bool requires_grad = false;
    std::vector<std::shared_ptr<TensorNode>> parents;
    std::function<void()> backward_fn;  // pushes this node's grad into its parents
};

// An N-dimensional tensor with reverse-mode automatic differentiation.
// Tensors are cheap handles: copying a Tensor shares the underlying data.
class Tensor
{
public:
    std::shared_ptr<TensorNode> node;

    Tensor();
    Tensor(const Shape &shape, FP_DTYPE value = 0, bool requires_grad = false);
    Tensor(const Shape &shape, const Vector &data, bool requires_grad = false);

    // Creation helpers
    static Tensor zeros(const Shape &shape, bool requires_grad = false);
    static Tensor ones(const Shape &shape, bool requires_grad = false);
    static Tensor rand(const Shape &shape, FP_DTYPE mn = -1, FP_DTYPE mx = 1, bool requires_grad = false);
    static Tensor randn(const Shape &shape, FP_DTYPE mean = 0, FP_DTYPE stddev = 1, bool requires_grad = false);
    static Tensor from_matrix(const Matrix &m, bool requires_grad = false);
    Matrix to_matrix() const;

    // Shape information
    const Shape &shape() const;
    int ndim() const;
    int dim(int i) const;   // negative indices count from the back
    int size() const;
    bool defined() const;

    // Element access
    Vector &data() const;
    Vector &grad() const;
    FP_DTYPE &operator[](int i) const;
    FP_DTYPE &at(int i, int j) const;
    FP_DTYPE item() const;

    // Autograd
    bool requires_grad() const;
    Tensor &set_requires_grad(bool requires_grad);
    void backward() const;          // only valid on single element tensors
    void zero_grad() const;
    Tensor detach() const;          // copy of the data, cut from the graph
};

// Disables graph construction while in scope (useful for inference)
struct NoGradGuard
{
    bool previous;
    NoGradGuard();
    ~NoGradGuard();
};
bool grad_enabled();

// Element-wise arithmetic with numpy style broadcasting
Tensor operator+(const Tensor &a, const Tensor &b);
Tensor operator-(const Tensor &a, const Tensor &b);
Tensor operator*(const Tensor &a, const Tensor &b);
Tensor operator/(const Tensor &a, const Tensor &b);
Tensor operator+(const Tensor &a, FP_DTYPE b);
Tensor operator-(const Tensor &a, FP_DTYPE b);
Tensor operator*(const Tensor &a, FP_DTYPE b);
Tensor operator/(const Tensor &a, FP_DTYPE b);
Tensor operator+(FP_DTYPE a, const Tensor &b);
Tensor operator-(FP_DTYPE a, const Tensor &b);
Tensor operator*(FP_DTYPE a, const Tensor &b);
Tensor operator/(FP_DTYPE a, const Tensor &b);
Tensor operator-(const Tensor &a);

// Element-wise functions
Tensor exp(const Tensor &a);
Tensor log(const Tensor &a);
Tensor sqrt(const Tensor &a);
Tensor pow(const Tensor &a, FP_DTYPE p);
Tensor tanh(const Tensor &a);
Tensor sigmoid(const Tensor &a);
Tensor relu(const Tensor &a);
Tensor gelu(const Tensor &a);   // tanh approximation

// Reductions
Tensor sum(const Tensor &a);                                  // -> shape {1}
Tensor mean(const Tensor &a);                                 // -> shape {1}
Tensor sum(const Tensor &a, int axis, bool keepdim = true);
Tensor mean(const Tensor &a, int axis, bool keepdim = true);

// Linear algebra and shape manipulation
Tensor matmul(const Tensor &a, const Tensor &b);   // (..., k) x (k, m) -> (..., m)
Tensor transpose(const Tensor &a);                 // 2D only
Tensor reshape(const Tensor &a, const Shape &shape);
Tensor slice(const Tensor &a, int start, int end); // along the last axis, [start, end)
Tensor concat(const std::vector<Tensor> &tensors); // along the last axis
Tensor gather_rows(const Tensor &a, const std::vector<int> &rows);  // 2D: picks rows of a

// Neural network helpers
Tensor softmax(const Tensor &a);                   // along the last axis
Tensor causal_mask(const Tensor &a);               // 2D square: masks out j > i
Tensor mse_loss(const Tensor &pred, const Tensor &target);
Tensor bce_with_logits(const Tensor &logits, const Tensor &target);
Tensor cross_entropy(const Tensor &logits, const std::vector<int> &targets);  // (N, C) logits

std::ostream &operator<<(std::ostream &os, const Tensor &t);
