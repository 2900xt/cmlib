# cmlib

A small machine learning library written from scratch in C++11: no math libraries, only the standard library.

## Goals

- [x] create tensors (N-dimensional, with automatic differentiation)
- [x] add linear regression (and logistic regression)
- [x] add neural networks (layers, activations, losses, SGD / Adam)
- [x] add transformers (multi-head attention, GPT-style language model)
- [x] no math libraries

## Building

```sh
make                      # builds lib/libcmlib.a and every program in tests/ into bin/
make test                 # runs the unit tests (gradient checks + small training runs)
make run-<name>           # builds and runs tests/<name>.cpp, e.g. make run-transformerLM
make clean
```

Plots are drawn with [gnuplot](http://www.gnuplot.info/) when it is installed, and skipped otherwise.

| Program | What it does |
| --- | --- |
| `unitTests` | Checks every tensor op's gradient against finite differences, plus XOR / regression / transformer training |
| `linearReg` | Linear regression with the `Matrix` API and finite-difference gradient descent |
| `logisticReg` | Logistic regression with the `Matrix` API |
| `xorMLP` | A two-layer neural network learning XOR |
| `transformerLM` | A tiny character-level GPT trained on a few lines of Hamlet, then generating text |
| `plotTest` | gnuplot smoke test |

## Layout

```
include/config.h            FP_DTYPE (long double), Vector, Matrix typedefs
include/math/               vector.h / matrix.h: plain std::vector math (vadd, mdot, ...)
include/alg/grad_descent.h  finite-difference gradient descent for Matrix based Models
include/models/model.h      Model base class for the Matrix API
include/tensor/tensor.h     Tensor with reverse-mode autograd
include/nn/nn.h             Module, Layer, Linear, activations, Sequential, LayerNorm, Embedding
include/nn/optim.h          SGD (with momentum), Adam
include/nn/transformer.h    MultiHeadAttention, TransformerBlock, TransformerLM
include/plot/plot.h         gnuplot wrapper
```

## Tensors and autograd

`Tensor` is a handle to row-major data with a shape. Operations on tensors that require grad record a graph, and `backward()` on a scalar fills in `.grad()` for every tensor that contributed.

```cpp
#include "tensor/tensor.h"

Tensor w = Tensor::randn({3, 1}, 0, 1, true);   // requires_grad = true
Tensor x({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
Tensor y({4, 1}, {1, 0, 1, 0});

Tensor loss = bce_with_logits(matmul(x, w), y);
loss.backward();
std::cout << w.grad() << '\n';
```

Supported ops:

- `+ - * /` with numpy-style broadcasting (tensor-tensor and tensor-scalar)
- `exp log sqrt pow tanh sigmoid relu gelu`
- `sum mean` (whole tensor or along an axis), `matmul transpose reshape slice concat gather_rows`
- `softmax causal_mask mse_loss bce_with_logits cross_entropy`

Use `NoGradGuard` to turn off graph building for inference, and `Tensor::from_matrix` / `to_matrix` to convert to and from the `Matrix` API.

## Neural networks

```cpp
#include "nn/nn.h"
#include "nn/optim.h"

Sequential net({layer<Linear>(2, 8), layer<Tanh>(), layer<Linear>(8, 1)});
Adam optimizer(net.parameters(), 0.05);

for(int epoch = 0; epoch < 300; epoch++)
{
    optimizer.zero_grad();
    Tensor loss = bce_with_logits(net(x), y);
    loss.backward();
    optimizer.step();
}
```

Custom layers derive from `Layer`, implement `forward`, and return their tensors from `parameters()`.

## Transformers

`TransformerLM` is a decoder-only (GPT-style) model: token and learned positional embeddings, `n_layers` pre-norm blocks (causal multi-head self-attention + GELU MLP), a final LayerNorm, and a linear head.

```cpp
#include "nn/transformer.h"

TransformerLM model(vocab_size, /*max_len*/ 16, /*d_model*/ 32, /*n_heads*/ 4, /*n_layers*/ 2, /*d_ff*/ 64);
Tensor logits = model.forward(tokens);                 // (T, vocab_size)
Tensor loss = cross_entropy(logits, next_tokens);
std::vector<int> text = model.generate(prompt, 100);   // greedy, or pass a temperature > 0
```

Sequences are processed one at a time as `(T, d_model)` matrices; to batch, add up the losses of several sequences before calling `backward()` (see `tests/transformerLM.cpp`).
