#include "tensor/tensor.h"
#include "math/vector.h"
#include <cmath>
#include <unordered_set>

static bool g_grad_enabled = true;

NoGradGuard::NoGradGuard() : previous(g_grad_enabled) { g_grad_enabled = false; }
NoGradGuard::~NoGradGuard() { g_grad_enabled = previous; }
bool grad_enabled() { return g_grad_enabled; }

static void fail(const std::string &msg)
{
    std::cerr << msg << std::endl;
    exit(EXIT_FAILURE);
}

static int shape_size(const Shape &shape)
{
    int n = 1;
    for(size_t i = 0; i < shape.size(); i++) n *= shape[i];
    return n;
}

static std::string shape_str(const Shape &shape)
{
    std::string s = "(";
    for(size_t i = 0; i < shape.size(); i++)
    {
        if(i) s += ", ";
        s += std::to_string(shape[i]);
    }
    return s + ")";
}

static void ensure_grad(TensorNode *n)
{
    if(n->grad.size() != n->data.size()) n->grad.assign(n->data.size(), 0);
}

// ---------------------------------------------------------------------------
// Tensor handle
// ---------------------------------------------------------------------------

Tensor::Tensor() {}

Tensor::Tensor(const Shape &shape, FP_DTYPE value, bool requires_grad)
    : node(std::make_shared<TensorNode>())
{
    node->shape = shape;
    node->data.assign(shape_size(shape), value);
    node->requires_grad = requires_grad;
}

Tensor::Tensor(const Shape &shape, const Vector &data, bool requires_grad)
    : node(std::make_shared<TensorNode>())
{
    if((int)data.size() != shape_size(shape))
    {
        fail("Tensor data of size " + std::to_string(data.size()) + " does not match shape " + shape_str(shape));
    }
    node->shape = shape;
    node->data = data;
    node->requires_grad = requires_grad;
}

Tensor Tensor::zeros(const Shape &shape, bool requires_grad) { return Tensor(shape, 0, requires_grad); }
Tensor Tensor::ones(const Shape &shape, bool requires_grad) { return Tensor(shape, 1, requires_grad); }

Tensor Tensor::rand(const Shape &shape, FP_DTYPE mn, FP_DTYPE mx, bool requires_grad)
{
    return Tensor(shape, vrand(shape_size(shape), mn, mx), requires_grad);
}

Tensor Tensor::randn(const Shape &shape, FP_DTYPE mean, FP_DTYPE stddev, bool requires_grad)
{
    // Box-Muller transform
    Tensor out(shape, 0, requires_grad);
    Vector &d = out.data();
    for(size_t i = 0; i < d.size(); i++)
    {
        FP_DTYPE u1 = ((FP_DTYPE)std::rand() + 1) / ((FP_DTYPE)RAND_MAX + 2);
        FP_DTYPE u2 = ((FP_DTYPE)std::rand() + 1) / ((FP_DTYPE)RAND_MAX + 2);
        d[i] = mean + stddev * std::sqrt(-2 * std::log(u1)) * std::cos(2 * M_PI * u2);
    }
    return out;
}

Tensor Tensor::from_matrix(const Matrix &m, bool requires_grad)
{
    int rows = m.size(), cols = rows ? m[0].size() : 0;
    Vector data;
    data.reserve(rows * cols);
    for(int i = 0; i < rows; i++) data.insert(data.end(), m[i].begin(), m[i].end());
    return Tensor({rows, cols}, data, requires_grad);
}

Matrix Tensor::to_matrix() const
{
    if(ndim() != 2) fail("to_matrix() requires a 2D tensor, got shape " + shape_str(shape()));
    Matrix out = mmake(dim(0), dim(1));
    for(int i = 0; i < dim(0); i++)
        for(int j = 0; j < dim(1); j++)
            out[i][j] = at(i, j);
    return out;
}

const Shape &Tensor::shape() const { return node->shape; }
int Tensor::ndim() const { return node->shape.size(); }
int Tensor::size() const { return node->data.size(); }
bool Tensor::defined() const { return (bool)node; }

int Tensor::dim(int i) const
{
    if(i < 0) i += ndim();
    if(i < 0 || i >= ndim()) fail("Dimension index out of range for shape " + shape_str(shape()));
    return node->shape[i];
}

Vector &Tensor::data() const { return node->data; }

Vector &Tensor::grad() const
{
    ensure_grad(node.get());
    return node->grad;
}

FP_DTYPE &Tensor::operator[](int i) const { return node->data[i]; }
FP_DTYPE &Tensor::at(int i, int j) const { return node->data[i * dim(-1) + j]; }

FP_DTYPE Tensor::item() const
{
    if(size() != 1) fail("item() requires a single element tensor, got shape " + shape_str(shape()));
    return node->data[0];
}

bool Tensor::requires_grad() const { return node->requires_grad; }

Tensor &Tensor::set_requires_grad(bool requires_grad)
{
    node->requires_grad = requires_grad;
    return *this;
}

void Tensor::zero_grad() const
{
    node->grad.assign(node->data.size(), 0);
}

Tensor Tensor::detach() const
{
    return Tensor(shape(), data());
}

void Tensor::backward() const
{
    if(size() != 1) fail("backward() can only be called on a single element tensor, got shape " + shape_str(shape()));
    if(!requires_grad()) fail("backward() called on a tensor that does not require grad");

    // Iterative post-order DFS -> topological order of the graph
    std::vector<TensorNode*> order;
    std::unordered_set<TensorNode*> visited;
    std::vector<std::pair<TensorNode*, size_t> > stack;
    stack.push_back(std::make_pair(node.get(), (size_t)0));
    visited.insert(node.get());
    while(!stack.empty())
    {
        TensorNode *cur = stack.back().first;
        size_t &next = stack.back().second;
        if(next < cur->parents.size())
        {
            TensorNode *parent = cur->parents[next++].get();
            if(parent->requires_grad && !visited.count(parent))
            {
                visited.insert(parent);
                stack.push_back(std::make_pair(parent, (size_t)0));
            }
        }
        else
        {
            order.push_back(cur);
            stack.pop_back();
        }
    }

    ensure_grad(node.get());
    node->grad[0] += 1;
    for(int i = (int)order.size() - 1; i >= 0; i--)
    {
        if(order[i]->backward_fn)
        {
            ensure_grad(order[i]);
            order[i]->backward_fn();
        }
    }
}

// ---------------------------------------------------------------------------
// Op helpers
// ---------------------------------------------------------------------------

// Creates the output of an op and links it into the graph if any input needs grad
static Tensor make_result(const Shape &shape, const Vector &data, const std::vector<Tensor> &inputs)
{
    Tensor out(shape, data);
    if(!g_grad_enabled) return out;
    for(size_t i = 0; i < inputs.size(); i++)
    {
        if(inputs[i].requires_grad()) out.node->requires_grad = true;
    }
    if(out.requires_grad())
    {
        for(size_t i = 0; i < inputs.size(); i++) out.node->parents.push_back(inputs[i].node);
    }
    return out;
}

static Shape broadcast_shape(const Shape &a, const Shape &b)
{
    size_t nd = std::max(a.size(), b.size());
    Shape out(nd);
    for(size_t i = 0; i < nd; i++)
    {
        int da = i < nd - a.size() ? 1 : a[i - (nd - a.size())];
        int db = i < nd - b.size() ? 1 : b[i - (nd - b.size())];
        if(da != db && da != 1 && db != 1)
        {
            fail("Cannot broadcast shapes " + shape_str(a) + " and " + shape_str(b));
        }
        out[i] = std::max(da, db);
    }
    return out;
}

// For every flat index of `out`, the flat index of the broadcast input `in`
static std::vector<int> broadcast_indices(const Shape &out, const Shape &in)
{
    int nd = out.size(), off = nd - in.size();
    std::vector<int> strides(nd, 0);
    int s = 1;
    for(int d = nd - 1; d >= off; d--)
    {
        if(in[d - off] != 1) strides[d] = s;
        s *= in[d - off];
    }

    int n = shape_size(out);
    std::vector<int> idx(n);
    std::vector<int> counter(nd, 0);
    int cur = 0;
    for(int i = 0; i < n; i++)
    {
        idx[i] = cur;
        for(int d = nd - 1; d >= 0; d--)
        {
            counter[d]++;
            cur += strides[d];
            if(counter[d] < out[d]) break;
            cur -= strides[d] * out[d];
            counter[d] = 0;
        }
    }
    return idx;
}

// f(x, y) -> z, dfdx(x, y, z), dfdy(x, y, z)
template<typename F, typename DA, typename DB>
static Tensor binary_op(const Tensor &a, const Tensor &b, F f, DA dfda, DB dfdb)
{
    Shape shape = broadcast_shape(a.shape(), b.shape());
    std::vector<int> ia = broadcast_indices(shape, a.shape());
    std::vector<int> ib = broadcast_indices(shape, b.shape());

    const Vector &A = a.data(), &B = b.data();
    Vector data(ia.size());
    for(size_t i = 0; i < ia.size(); i++) data[i] = f(A[ia[i]], B[ib[i]]);

    Tensor out = make_result(shape, data, {a, b});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node, bn = b.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, bn, on, ia, ib, dfda, dfdb]()
        {
            if(an->requires_grad) ensure_grad(an.get());
            if(bn->requires_grad) ensure_grad(bn.get());
            for(size_t i = 0; i < ia.size(); i++)
            {
                FP_DTYPE x = an->data[ia[i]], y = bn->data[ib[i]], g = on->grad[i];
                if(an->requires_grad) an->grad[ia[i]] += g * dfda(x, y, on->data[i]);
                if(bn->requires_grad) bn->grad[ib[i]] += g * dfdb(x, y, on->data[i]);
            }
        };
    }
    return out;
}

// f(x) -> y, dfdx(x, y)
template<typename F, typename DF>
static Tensor unary_op(const Tensor &a, F f, DF dfdx)
{
    const Vector &A = a.data();
    Vector data(A.size());
    for(size_t i = 0; i < A.size(); i++) data[i] = f(A[i]);

    Tensor out = make_result(a.shape(), data, {a});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, on, dfdx]()
        {
            ensure_grad(an.get());
            for(size_t i = 0; i < an->data.size(); i++)
            {
                an->grad[i] += on->grad[i] * dfdx(an->data[i], on->data[i]);
            }
        };
    }
    return out;
}

// ---------------------------------------------------------------------------
// Element-wise arithmetic
// ---------------------------------------------------------------------------

Tensor operator+(const Tensor &a, const Tensor &b)
{
    return binary_op(a, b,
        [](FP_DTYPE x, FP_DTYPE y) { return x + y; },
        [](FP_DTYPE, FP_DTYPE, FP_DTYPE) { return (FP_DTYPE)1; },
        [](FP_DTYPE, FP_DTYPE, FP_DTYPE) { return (FP_DTYPE)1; });
}

Tensor operator-(const Tensor &a, const Tensor &b)
{
    return binary_op(a, b,
        [](FP_DTYPE x, FP_DTYPE y) { return x - y; },
        [](FP_DTYPE, FP_DTYPE, FP_DTYPE) { return (FP_DTYPE)1; },
        [](FP_DTYPE, FP_DTYPE, FP_DTYPE) { return (FP_DTYPE)-1; });
}

Tensor operator*(const Tensor &a, const Tensor &b)
{
    return binary_op(a, b,
        [](FP_DTYPE x, FP_DTYPE y) { return x * y; },
        [](FP_DTYPE, FP_DTYPE y, FP_DTYPE) { return y; },
        [](FP_DTYPE x, FP_DTYPE, FP_DTYPE) { return x; });
}

Tensor operator/(const Tensor &a, const Tensor &b)
{
    return binary_op(a, b,
        [](FP_DTYPE x, FP_DTYPE y) { return x / y; },
        [](FP_DTYPE, FP_DTYPE y, FP_DTYPE) { return 1 / y; },
        [](FP_DTYPE x, FP_DTYPE y, FP_DTYPE) { return -x / (y * y); });
}

Tensor operator+(const Tensor &a, FP_DTYPE b)
{
    return unary_op(a,
        [b](FP_DTYPE x) { return x + b; },
        [](FP_DTYPE, FP_DTYPE) { return (FP_DTYPE)1; });
}

Tensor operator-(const Tensor &a, FP_DTYPE b) { return a + (-b); }

Tensor operator*(const Tensor &a, FP_DTYPE b)
{
    return unary_op(a,
        [b](FP_DTYPE x) { return x * b; },
        [b](FP_DTYPE, FP_DTYPE) { return b; });
}

Tensor operator/(const Tensor &a, FP_DTYPE b) { return a * (1 / b); }
Tensor operator+(FP_DTYPE a, const Tensor &b) { return b + a; }
Tensor operator-(FP_DTYPE a, const Tensor &b) { return (-b) + a; }
Tensor operator*(FP_DTYPE a, const Tensor &b) { return b * a; }

Tensor operator/(FP_DTYPE a, const Tensor &b)
{
    return unary_op(b,
        [a](FP_DTYPE x) { return a / x; },
        [a](FP_DTYPE x, FP_DTYPE) { return -a / (x * x); });
}

Tensor operator-(const Tensor &a) { return a * (FP_DTYPE)-1; }

// ---------------------------------------------------------------------------
// Element-wise functions
// ---------------------------------------------------------------------------

Tensor exp(const Tensor &a)
{
    return unary_op(a,
        [](FP_DTYPE x) { return std::exp(x); },
        [](FP_DTYPE, FP_DTYPE y) { return y; });
}

Tensor log(const Tensor &a)
{
    return unary_op(a,
        [](FP_DTYPE x) { return std::log(x); },
        [](FP_DTYPE x, FP_DTYPE) { return 1 / x; });
}

Tensor sqrt(const Tensor &a)
{
    return unary_op(a,
        [](FP_DTYPE x) { return std::sqrt(x); },
        [](FP_DTYPE, FP_DTYPE y) { return 1 / (2 * y); });
}

Tensor pow(const Tensor &a, FP_DTYPE p)
{
    return unary_op(a,
        [p](FP_DTYPE x) { return std::pow(x, p); },
        [p](FP_DTYPE x, FP_DTYPE) { return p * std::pow(x, p - 1); });
}

Tensor tanh(const Tensor &a)
{
    return unary_op(a,
        [](FP_DTYPE x) { return std::tanh(x); },
        [](FP_DTYPE, FP_DTYPE y) { return 1 - y * y; });
}

Tensor sigmoid(const Tensor &a)
{
    return unary_op(a,
        [](FP_DTYPE x) { return 1 / (1 + std::exp(-x)); },
        [](FP_DTYPE, FP_DTYPE y) { return y * (1 - y); });
}

Tensor relu(const Tensor &a)
{
    return unary_op(a,
        [](FP_DTYPE x) { return x > 0 ? x : 0; },
        [](FP_DTYPE x, FP_DTYPE) { return (FP_DTYPE)(x > 0 ? 1 : 0); });
}

Tensor gelu(const Tensor &a)
{
    const FP_DTYPE c = std::sqrt(2 / M_PI);
    return unary_op(a,
        [c](FP_DTYPE x) { return (FP_DTYPE)0.5 * x * (1 + std::tanh(c * (x + (FP_DTYPE)0.044715 * x * x * x))); },
        [c](FP_DTYPE x, FP_DTYPE)
        {
            FP_DTYPE t = std::tanh(c * (x + (FP_DTYPE)0.044715 * x * x * x));
            FP_DTYPE dt = (1 - t * t) * c * (1 + 3 * (FP_DTYPE)0.044715 * x * x);
            return (FP_DTYPE)0.5 * (1 + t) + (FP_DTYPE)0.5 * x * dt;
        });
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

Tensor sum(const Tensor &a)
{
    Tensor out = make_result({1}, {vsum(a.data())}, {a});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, on]()
        {
            ensure_grad(an.get());
            for(size_t i = 0; i < an->grad.size(); i++) an->grad[i] += on->grad[0];
        };
    }
    return out;
}

Tensor mean(const Tensor &a)
{
    return sum(a) / (FP_DTYPE)a.size();
}

Tensor sum(const Tensor &a, int axis, bool keepdim)
{
    if(axis < 0) axis += a.ndim();
    if(axis < 0 || axis >= a.ndim()) fail("sum(): axis out of range for shape " + shape_str(a.shape()));

    int outer = 1, n = a.dim(axis), inner = 1;
    for(int d = 0; d < axis; d++) outer *= a.dim(d);
    for(int d = axis + 1; d < a.ndim(); d++) inner *= a.dim(d);

    Shape shape = a.shape();
    if(keepdim) shape[axis] = 1;
    else shape.erase(shape.begin() + axis);
    if(shape.empty()) shape.push_back(1);

    const Vector &A = a.data();
    Vector data(outer * inner, 0);
    for(int o = 0; o < outer; o++)
        for(int k = 0; k < n; k++)
            for(int i = 0; i < inner; i++)
                data[o * inner + i] += A[(o * n + k) * inner + i];

    Tensor out = make_result(shape, data, {a});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, on, outer, n, inner]()
        {
            ensure_grad(an.get());
            for(int o = 0; o < outer; o++)
                for(int k = 0; k < n; k++)
                    for(int i = 0; i < inner; i++)
                        an->grad[(o * n + k) * inner + i] += on->grad[o * inner + i];
        };
    }
    return out;
}

Tensor mean(const Tensor &a, int axis, bool keepdim)
{
    return sum(a, axis, keepdim) / (FP_DTYPE)a.dim(axis);
}

// ---------------------------------------------------------------------------
// Linear algebra and shape manipulation
// ---------------------------------------------------------------------------

Tensor matmul(const Tensor &a, const Tensor &b)
{
    if(b.ndim() != 2 || a.dim(-1) != b.dim(0))
    {
        fail("Cannot matmul tensors with shapes " + shape_str(a.shape()) + " and " + shape_str(b.shape()));
    }
    int k = b.dim(0), m = b.dim(1), n = a.size() / k;

    const Vector &A = a.data(), &B = b.data();
    Vector data(n * m, 0);
    for(int i = 0; i < n; i++)
        for(int p = 0; p < k; p++)
        {
            FP_DTYPE x = A[i * k + p];
            if(x == 0) continue;
            for(int j = 0; j < m; j++) data[i * m + j] += x * B[p * m + j];
        }

    Shape shape = a.shape();
    shape.back() = m;
    Tensor out = make_result(shape, data, {a, b});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node, bn = b.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, bn, on, n, k, m]()
        {
            const Vector &G = on->grad;
            if(an->requires_grad)
            {
                // dA = dOut @ B^T
                ensure_grad(an.get());
                for(int i = 0; i < n; i++)
                    for(int p = 0; p < k; p++)
                    {
                        FP_DTYPE s = 0;
                        for(int j = 0; j < m; j++) s += G[i * m + j] * bn->data[p * m + j];
                        an->grad[i * k + p] += s;
                    }
            }
            if(bn->requires_grad)
            {
                // dB = A^T @ dOut
                ensure_grad(bn.get());
                for(int i = 0; i < n; i++)
                    for(int p = 0; p < k; p++)
                    {
                        FP_DTYPE x = an->data[i * k + p];
                        if(x == 0) continue;
                        for(int j = 0; j < m; j++) bn->grad[p * m + j] += x * G[i * m + j];
                    }
            }
        };
    }
    return out;
}

Tensor transpose(const Tensor &a)
{
    if(a.ndim() != 2) fail("transpose() requires a 2D tensor, got shape " + shape_str(a.shape()));
    int r = a.dim(0), c = a.dim(1);
    Vector data(r * c);
    for(int i = 0; i < r; i++)
        for(int j = 0; j < c; j++)
            data[j * r + i] = a[i * c + j];

    Tensor out = make_result({c, r}, data, {a});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, on, r, c]()
        {
            ensure_grad(an.get());
            for(int i = 0; i < r; i++)
                for(int j = 0; j < c; j++)
                    an->grad[i * c + j] += on->grad[j * r + i];
        };
    }
    return out;
}

Tensor reshape(const Tensor &a, const Shape &shape)
{
    if(shape_size(shape) != a.size())
    {
        fail("Cannot reshape " + shape_str(a.shape()) + " into " + shape_str(shape));
    }
    Tensor out = make_result(shape, a.data(), {a});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, on]()
        {
            ensure_grad(an.get());
            for(size_t i = 0; i < on->grad.size(); i++) an->grad[i] += on->grad[i];
        };
    }
    return out;
}

Tensor slice(const Tensor &a, int start, int end)
{
    int cols = a.dim(-1), rows = a.size() / cols, w = end - start;
    if(start < 0 || end > cols || w <= 0)
    {
        fail("Invalid slice [" + std::to_string(start) + ", " + std::to_string(end) + ") of shape " + shape_str(a.shape()));
    }
    Vector data(rows * w);
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < w; j++)
            data[i * w + j] = a[i * cols + start + j];

    Shape shape = a.shape();
    shape.back() = w;
    Tensor out = make_result(shape, data, {a});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, on, rows, cols, start, w]()
        {
            ensure_grad(an.get());
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < w; j++)
                    an->grad[i * cols + start + j] += on->grad[i * w + j];
        };
    }
    return out;
}

Tensor concat(const std::vector<Tensor> &tensors)
{
    if(tensors.empty()) fail("concat() needs at least one tensor");
    int rows = tensors[0].size() / tensors[0].dim(-1), cols = 0;
    std::vector<int> offsets;
    for(size_t t = 0; t < tensors.size(); t++)
    {
        Shape a = tensors[t].shape(), b = tensors[0].shape();
        a.pop_back(); b.pop_back();
        if(a != b) fail("concat(): shapes " + shape_str(tensors[0].shape()) + " and " + shape_str(tensors[t].shape()) + " differ");
        offsets.push_back(cols);
        cols += tensors[t].dim(-1);
    }

    Vector data(rows * cols);
    for(size_t t = 0; t < tensors.size(); t++)
    {
        int w = tensors[t].dim(-1);
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < w; j++)
                data[i * cols + offsets[t] + j] = tensors[t][i * w + j];
    }

    Shape shape = tensors[0].shape();
    shape.back() = cols;
    Tensor out = make_result(shape, data, tensors);
    if(out.requires_grad())
    {
        std::vector<std::shared_ptr<TensorNode> > nodes;
        for(size_t t = 0; t < tensors.size(); t++) nodes.push_back(tensors[t].node);
        TensorNode *on = out.node.get();
        out.node->backward_fn = [nodes, on, rows, cols, offsets]()
        {
            for(size_t t = 0; t < nodes.size(); t++)
            {
                if(!nodes[t]->requires_grad) continue;
                ensure_grad(nodes[t].get());
                int w = nodes[t]->shape.back();
                for(int i = 0; i < rows; i++)
                    for(int j = 0; j < w; j++)
                        nodes[t]->grad[i * w + j] += on->grad[i * cols + offsets[t] + j];
            }
        };
    }
    return out;
}

Tensor gather_rows(const Tensor &a, const std::vector<int> &rows)
{
    if(a.ndim() != 2) fail("gather_rows() requires a 2D tensor, got shape " + shape_str(a.shape()));
    int n = a.dim(0), cols = a.dim(1);
    Vector data(rows.size() * cols);
    for(size_t i = 0; i < rows.size(); i++)
    {
        if(rows[i] < 0 || rows[i] >= n) fail("gather_rows(): index " + std::to_string(rows[i]) + " out of range");
        for(int j = 0; j < cols; j++) data[i * cols + j] = a[rows[i] * cols + j];
    }

    Tensor out = make_result({(int)rows.size(), cols}, data, {a});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, on, rows, cols]()
        {
            ensure_grad(an.get());
            for(size_t i = 0; i < rows.size(); i++)
                for(int j = 0; j < cols; j++)
                    an->grad[rows[i] * cols + j] += on->grad[i * cols + j];
        };
    }
    return out;
}

// ---------------------------------------------------------------------------
// Neural network helpers
// ---------------------------------------------------------------------------

Tensor softmax(const Tensor &a)
{
    int cols = a.dim(-1), rows = a.size() / cols;
    const Vector &A = a.data();
    Vector data(A.size());
    for(int i = 0; i < rows; i++)
    {
        FP_DTYPE mx = A[i * cols];
        for(int j = 1; j < cols; j++) mx = std::max(mx, A[i * cols + j]);
        FP_DTYPE total = 0;
        for(int j = 0; j < cols; j++)
        {
            data[i * cols + j] = std::exp(A[i * cols + j] - mx);
            total += data[i * cols + j];
        }
        for(int j = 0; j < cols; j++) data[i * cols + j] /= total;
    }

    Tensor out = make_result(a.shape(), data, {a});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, on, rows, cols]()
        {
            // dx = y * (dy - sum(dy * y))
            ensure_grad(an.get());
            for(int i = 0; i < rows; i++)
            {
                FP_DTYPE dot = 0;
                for(int j = 0; j < cols; j++) dot += on->grad[i * cols + j] * on->data[i * cols + j];
                for(int j = 0; j < cols; j++)
                {
                    an->grad[i * cols + j] += on->data[i * cols + j] * (on->grad[i * cols + j] - dot);
                }
            }
        };
    }
    return out;
}

Tensor causal_mask(const Tensor &a)
{
    if(a.ndim() != 2 || a.dim(0) != a.dim(1))
    {
        fail("causal_mask() requires a square 2D tensor, got shape " + shape_str(a.shape()));
    }
    const FP_DTYPE NEG_INF = -1e30;
    int n = a.dim(0);
    Vector data = a.data();
    for(int i = 0; i < n; i++)
        for(int j = i + 1; j < n; j++)
            data[i * n + j] = NEG_INF;

    Tensor out = make_result(a.shape(), data, {a});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> an = a.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [an, on, n]()
        {
            ensure_grad(an.get());
            for(int i = 0; i < n; i++)
                for(int j = 0; j <= i; j++)
                    an->grad[i * n + j] += on->grad[i * n + j];
        };
    }
    return out;
}

Tensor mse_loss(const Tensor &pred, const Tensor &target)
{
    return mean(pow(pred - target, 2));
}

Tensor bce_with_logits(const Tensor &logits, const Tensor &target)
{
    if(logits.shape() != target.shape())
    {
        fail("bce_with_logits(): shapes " + shape_str(logits.shape()) + " and " + shape_str(target.shape()) + " differ");
    }
    // loss = max(z, 0) - z*y + log(1 + exp(-|z|)), numerically stable
    int n = logits.size();
    FP_DTYPE total = 0;
    for(int i = 0; i < n; i++)
    {
        FP_DTYPE z = logits[i], y = target[i];
        total += std::max(z, (FP_DTYPE)0) - z * y + std::log1p(std::exp(-std::fabs(z)));
    }

    Tensor out = make_result({1}, {total / n}, {logits});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> zn = logits.node, yn = target.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [zn, yn, on, n]()
        {
            ensure_grad(zn.get());
            for(int i = 0; i < n; i++)
            {
                FP_DTYPE s = 1 / (1 + std::exp(-zn->data[i]));
                zn->grad[i] += on->grad[0] * (s - yn->data[i]) / n;
            }
        };
    }
    return out;
}

Tensor cross_entropy(const Tensor &logits, const std::vector<int> &targets)
{
    if(logits.ndim() != 2 || logits.dim(0) != (int)targets.size())
    {
        fail("cross_entropy(): logits of shape " + shape_str(logits.shape()) + " do not match "
             + std::to_string(targets.size()) + " targets");
    }
    int n = logits.dim(0), c = logits.dim(1);
    Vector probs(n * c);
    FP_DTYPE total = 0;
    for(int i = 0; i < n; i++)
    {
        if(targets[i] < 0 || targets[i] >= c) fail("cross_entropy(): target " + std::to_string(targets[i]) + " out of range");
        FP_DTYPE mx = logits[i * c];
        for(int j = 1; j < c; j++) mx = std::max(mx, logits[i * c + j]);
        FP_DTYPE s = 0;
        for(int j = 0; j < c; j++) s += std::exp(logits[i * c + j] - mx);
        FP_DTYPE log_z = mx + std::log(s);
        for(int j = 0; j < c; j++) probs[i * c + j] = std::exp(logits[i * c + j] - log_z);
        total += log_z - logits[i * c + targets[i]];
    }

    Tensor out = make_result({1}, {total / n}, {logits});
    if(out.requires_grad())
    {
        std::shared_ptr<TensorNode> zn = logits.node;
        TensorNode *on = out.node.get();
        out.node->backward_fn = [zn, on, probs, targets, n, c]()
        {
            // d/dz = (softmax(z) - onehot(y)) / n
            ensure_grad(zn.get());
            for(int i = 0; i < n; i++)
                for(int j = 0; j < c; j++)
                {
                    FP_DTYPE g = probs[i * c + j] - (j == targets[i] ? 1 : 0);
                    zn->grad[i * c + j] += on->grad[0] * g / n;
                }
        };
    }
    return out;
}

std::ostream &operator<<(std::ostream &os, const Tensor &t)
{
    os << std::fixed << std::setprecision(4);
    os << "Tensor" << shape_str(t.shape()) << "\n[";
    int cols = t.ndim() ? t.dim(-1) : 1;
    for(int i = 0; i < t.size(); i++)
    {
        if(i && i % cols == 0) os << "\n ";
        os << t[i];
        if(i != t.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}
