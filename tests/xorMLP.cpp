#include "config.h"
#include "tensor/tensor.h"
#include "nn/nn.h"
#include "nn/optim.h"
#include "math/vector.h"
#include "plot/plot.h"

// Trains a small neural network on XOR, which a linear model cannot solve
int main()
{
    srand(42);
    Tensor x({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
    Tensor y({4, 1}, {0, 1, 1, 0});

    Sequential net({
        layer<Linear>(2, 8),
        layer<Tanh>(),
        layer<Linear>(8, 1)
    });
    Adam optimizer(net.parameters(), 0.05);

    int epochs = 300;
    Vector epochArray, lossArray;
    for(int epoch = 0; epoch < epochs; epoch++)
    {
        optimizer.zero_grad();
        Tensor loss = bce_with_logits(net(x), y);
        loss.backward();
        optimizer.step();

        epochArray.push_back(epoch);
        lossArray.push_back(loss.item());
        if(epoch % 50 == 0) std::cout << "Epoch: " << epoch << "; Loss: " << loss.item() << '\n';
    }

    Tensor pred = sigmoid(net(x));
    std::cout << std::setprecision(3);
    for(int i = 0; i < 4; i++)
    {
        std::cout << x.at(i, 0) << " XOR " << x.at(i, 1) << " -> " << pred[i] << " (expected " << y[i] << ")\n";
    }

    plot({{epochArray, lossArray, "with lines title 'BCE Loss'"}}, "XOR Training Loss", "Epoch", "Loss");
}
