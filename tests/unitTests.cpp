#include "config.h"
#include "math/matrix.h"
#include "math/vector.h"
#include "tensor/tensor.h"
#include "nn/nn.h"
#include "nn/optim.h"
#include "nn/transformer.h"
#include <cmath>
#include <functional>

static int failures = 0, checks = 0;

static void check(bool ok, const std::string &name)
{
    checks++;
    if(!ok)
    {
        failures++;
        std::cout << "FAIL: " << name << '\n';
    }
}

static bool close(FP_DTYPE a, FP_DTYPE b, FP_DTYPE tol = 1e-6)
{
    return std::fabs(a - b) <= tol * (1 + std::fabs(a) + std::fabs(b));
}

// Compares autograd gradients of sum(f(inputs) * r) against central finite differences
static void gradcheck(const std::string &name, std::vector<Tensor> inputs,
                      std::function<Tensor(const std::vector<Tensor>&)> f)
{
    for(size_t i = 0; i < inputs.size(); i++)
    {
        inputs[i].set_requires_grad(true);
        inputs[i].zero_grad();
    }

    Tensor probe = f(inputs);
    Tensor r = Tensor::rand(probe.shape(), -1, 1);
    std::function<Tensor()> loss = [&]() { return sum(f(inputs) * r); };

    Tensor l = loss();
    l.backward();

    FP_DTYPE eps = 1e-6, worst = 0;
    bool ok = true;
    for(size_t t = 0; t < inputs.size(); t++)
    {
        Vector &x = inputs[t].data();
        for(size_t i = 0; i < x.size(); i++)
        {
            FP_DTYPE orig = x[i];
            x[i] = orig + eps;
            FP_DTYPE fp = loss().item();
            x[i] = orig - eps;
            FP_DTYPE fm = loss().item();
            x[i] = orig;
            FP_DTYPE numeric = (fp - fm) / (2 * eps), analytic = inputs[t].grad()[i];
            worst = std::max(worst, std::fabs(numeric - analytic));
            if(!close(numeric, analytic, 1e-5)) ok = false;
        }
    }
    if(!ok) std::cout << "  " << name << ": max abs grad error " << worst << '\n';
    check(ok, "gradcheck " + name);
}

static void test_matrix()
{
    Matrix a = {{1, 2}, {3, 4}}, b = {{5, 6}, {7, 8}};
    Matrix c = mdot(a, b);
    check(c[0][0] == 19 && c[0][1] == 22 && c[1][0] == 43 && c[1][1] == 50, "mdot");
    Matrix t = mtranspose(a);
    check(t[0][1] == 3 && t[1][0] == 2, "mtranspose");
    check(msum(a)[0] == 10 && msum(a, 0)[1] == 6 && msum(a, 1)[1] == 7, "msum");
    check(vdot({1, 2, 3}, {4, 5, 6}) == 32, "vdot");
}

static void test_tensor_basics()
{
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3}, {10, 20, 30});
    Tensor c = a + b;
    check(c.shape() == Shape({2, 3}) && c.at(1, 2) == 36, "broadcast add");

    Tensor col({2, 1}, {1, 2});
    Tensor d = a * col;
    check(d.at(0, 2) == 3 && d.at(1, 0) == 8, "broadcast column mul");

    Tensor s = sum(a, 0, false);
    check(s.shape() == Shape({3}) && s[0] == 5 && s[2] == 9, "sum axis 0");
    Tensor m = mean(a, -1);
    check(m.shape() == Shape({2, 1}) && m[1] == 5, "mean last axis");

    Tensor mm = matmul(a, transpose(a));
    check(mm.at(0, 0) == 14 && mm.at(0, 1) == 32 && mm.at(1, 1) == 77, "matmul");

    Tensor sm = softmax(Tensor({1, 3}, {1, 1, 1}));
    check(close(sm[0], 1.0L / 3), "softmax");

    Tensor masked = softmax(causal_mask(Tensor({2, 2}, {0, 5, 0, 0})));
    check(close(masked.at(0, 0), 1) && close(masked.at(0, 1), 0), "causal mask");

    check(Tensor::from_matrix(a.to_matrix()).data() == a.data(), "matrix round trip");

    // grads accumulate when a tensor is used twice
    Tensor x({1}, {3}, true);
    (x * x + x).backward();
    check(close(x.grad()[0], 7), "reused tensor grad");

    // no graph is built under NoGradGuard
    {
        NoGradGuard guard;
        Tensor y = x * 2;
        check(!y.requires_grad(), "no grad guard");
    }
    check(grad_enabled(), "grad re-enabled");
}

static void test_gradients()
{
    typedef const std::vector<Tensor> &In;
    Tensor a = Tensor::rand({3, 4}), b = Tensor::rand({3, 4}), row = Tensor::rand({4});
    Tensor pos = Tensor::rand({3, 4}, 0.5, 2), sq = Tensor::rand({4, 4}), w = Tensor::rand({4, 5});

    gradcheck("add broadcast", {a, row}, [](In x) { return x[0] + x[1]; });
    gradcheck("sub", {a, b}, [](In x) { return x[0] - x[1]; });
    gradcheck("mul broadcast", {a, row}, [](In x) { return x[0] * x[1]; });
    gradcheck("div", {a, pos}, [](In x) { return x[0] / x[1]; });
    gradcheck("scalar ops", {a}, [](In x) { return (2 - x[0]) * 3 + 1 / (x[0] + 5); });
    gradcheck("exp/log", {pos}, [](In x) { return log(x[0]) + exp(x[0]); });
    gradcheck("sqrt/pow", {pos}, [](In x) { return sqrt(x[0]) + pow(x[0], 3); });
    gradcheck("tanh/sigmoid", {a}, [](In x) { return tanh(x[0]) + sigmoid(x[0]); });
    gradcheck("relu", {a}, [](In x) { return relu(x[0]); });
    gradcheck("gelu", {a}, [](In x) { return gelu(x[0]); });
    gradcheck("sum/mean", {a}, [](In x) { return sum(x[0]) + mean(x[0]); });
    gradcheck("sum axis", {a}, [](In x) { return sum(x[0], 0, false) * 2 + mean(x[0], 0, false); });
    gradcheck("matmul", {a, w}, [](In x) { return matmul(x[0], x[1]); });
    gradcheck("transpose", {a}, [](In x) { return transpose(x[0]); });
    gradcheck("reshape", {a}, [](In x) { return reshape(x[0], {2, 6}); });
    gradcheck("slice/concat", {a, b}, [](In x) { return concat({slice(x[0], 1, 3), x[1]}); });
    gradcheck("gather_rows", {a}, [](In x) { return gather_rows(x[0], {2, 0, 2}); });
    gradcheck("softmax", {a}, [](In x) { return softmax(x[0]); });
    gradcheck("causal_mask+softmax", {sq}, [](In x) { return softmax(causal_mask(x[0])); });
    gradcheck("mse_loss", {a, b}, [](In x) { return mse_loss(x[0], x[1]); });
    Tensor labels = Tensor::rand({3, 4}, 0, 1);
    gradcheck("bce_with_logits", {a}, [labels](In x) { return bce_with_logits(x[0], labels); });
    gradcheck("cross_entropy", {a}, [](In x) { return cross_entropy(x[0], {1, 3, 0}); });

    LayerNorm ln(4);
    gradcheck("layer_norm", {a}, [&ln](In x) { return ln(x[0]); });

    MultiHeadAttention attn(4, 2);
    gradcheck("attention input", {a}, [&attn](In x) { return attn(x[0]); });

    // gradient w.r.t. model parameters of a whole transformer
    TransformerLM lm(5, 6, 8, 2, 1, 16);
    std::vector<int> tokens = {0, 3, 1, 4}, targets = {3, 1, 4, 2};
    gradcheck("transformer params", lm.parameters(),
              [&lm, &tokens, &targets](In) { return cross_entropy(lm.forward(tokens), targets); });
}

static void test_training()
{
    // A tiny MLP must fit XOR
    srand(1);
    Tensor x({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
    Tensor y({4, 1}, {0, 1, 1, 0});
    Sequential net({layer<Linear>(2, 8), layer<Tanh>(), layer<Linear>(8, 1)});
    Adam opt(net.parameters(), 0.05);
    FP_DTYPE loss = 0;
    for(int i = 0; i < 500; i++)
    {
        opt.zero_grad();
        Tensor l = bce_with_logits(net(x), y);
        l.backward();
        opt.step();
        loss = l.item();
    }
    check(loss < 0.01, "XOR training (loss " + std::to_string((double)loss) + ")");

    // SGD on linear regression recovers the true weights
    Tensor X = Tensor::rand({64, 1}, 0, 1);
    Tensor Y = X * 3 + 2;
    Linear lin(1, 1);
    SGD sgd(lin.parameters(), 0.5, 0.9);
    for(int i = 0; i < 300; i++)
    {
        sgd.zero_grad();
        mse_loss(lin(X), Y).backward();
        sgd.step();
    }
    check(close(lin.weight[0], 3, 1e-3) && close(lin.bias[0], 2, 1e-3), "SGD linear regression");

    // A transformer can memorize a short sequence
    std::vector<int> seq = {1, 2, 3, 4, 0, 1, 2, 3, 4, 0};
    std::vector<int> in(seq.begin(), seq.end() - 1), target(seq.begin() + 1, seq.end());
    TransformerLM lm(5, 9, 16, 2, 1, 32);
    Adam lm_opt(lm.parameters(), 0.01);
    for(int i = 0; i < 100; i++)
    {
        lm_opt.zero_grad();
        cross_entropy(lm.forward(in), target).backward();
        lm_opt.step();
    }
    std::vector<int> gen = lm.generate({1}, 8);
    check(gen == std::vector<int>(seq.begin(), seq.begin() + 9), "transformer memorization");
}

int main()
{
    srand(0);
    test_matrix();
    test_tensor_basics();
    test_gradients();
    test_training();

    std::cout << checks - failures << "/" << checks << " checks passed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
