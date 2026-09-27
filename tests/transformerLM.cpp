#include "config.h"
#include "tensor/tensor.h"
#include "nn/nn.h"
#include "nn/optim.h"
#include "nn/transformer.h"
#include "plot/plot.h"

// Trains a tiny character level GPT on a short text, then samples from it
int main()
{
    srand(0);
    std::string text =
        "to be, or not to be, that is the question. "
        "whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune, "
        "or to take arms against a sea of troubles. ";

    // Character vocabulary
    std::string vocab;
    for(size_t i = 0; i < text.size(); i++)
    {
        if(vocab.find(text[i]) == std::string::npos) vocab += text[i];
    }
    std::vector<int> data;
    for(size_t i = 0; i < text.size(); i++) data.push_back(vocab.find(text[i]));

    int context = 16, batch_size = 4, steps = 800;
    TransformerLM model(vocab.size(), context, 32, 4, 2, 64);
    Adam optimizer(model.parameters(), 5e-3);
    std::cout << "Vocab size: " << vocab.size() << "; Parameters: " << model.num_parameters() << '\n';

    Vector stepArray, lossArray;
    for(int step = 0; step < steps; step++)
    {
        optimizer.zero_grad();
        Tensor loss = Tensor::zeros({1});
        for(int b = 0; b < batch_size; b++)
        {
            int start = rand() % (data.size() - context - 1);
            std::vector<int> input(data.begin() + start, data.begin() + start + context);
            std::vector<int> target(data.begin() + start + 1, data.begin() + start + context + 1);
            loss = loss + cross_entropy(model.forward(input), target);
        }
        loss = loss / (FP_DTYPE)batch_size;
        loss.backward();
        optimizer.step();

        stepArray.push_back(step);
        lossArray.push_back(loss.item());
        if(step % 50 == 0) std::cout << "Step: " << step << "; Loss: " << loss.item() << '\n';
    }

    std::string prompt = "to be";
    std::vector<int> tokens;
    for(size_t i = 0; i < prompt.size(); i++) tokens.push_back(vocab.find(prompt[i]));
    std::vector<int> out = model.generate(tokens, 150);

    std::string generated;
    for(size_t i = 0; i < out.size(); i++) generated += vocab[out[i]];
    std::cout << "\nGenerated (greedy):\n" << generated << "\n";

    plot({{stepArray, lossArray, "with lines title 'Cross Entropy'"}}, "Transformer Training Loss", "Step", "Loss");
}
