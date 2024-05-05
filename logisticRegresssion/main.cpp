#include <torch/torch.h>
#include <iostream>
#include <matplotlibcpp.h>


namespace plt = matplotlibcpp;

class SimpleNN : public torch::nn::Module {
public:
    SimpleNN(int input_size, int hidden_size, int output_size) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = fc2(x);
        return x;
    }

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
    torch::Tensor inputs = torch::tensor({{2.3, 4.2}, {1.3, 3.2}, {3.3, 2.2}});
    torch::Tensor targets = torch::tensor({{1.0}, {0.0}, {1.0}});

    int input_size = 2;
    int hidden_size = 64;
    int output_size = 1;
    int num_epochs = 1000;
    double learning_rate = 0.01;

    SimpleNN model(input_size, hidden_size, output_size);

    torch::nn::BCEWithLogitsLoss criterion;
    torch::optim::SGD optimizer(model.parameters(), learning_rate);

    std::vector<double> losses;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Forward pass
        torch::Tensor outputs = model.forward(inputs);
        // Compute loss
        torch::Tensor loss = criterion(outputs, targets);
        // Zero gradients
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        losses.push_back(loss.item<float>());

        if ((epoch+1) % 100 == 0) {
            std::cout << "Epoch [" << (epoch+1) << "/" << num_epochs << "], Loss: " << loss.item<float>() << std::endl;
        }
    }

    // Plotting the loss
    plt::plot(losses);
    plt::grid(true);
    plt::xlabel("Epoch");
    plt::ylabel("Loss");
    plt::title("Training Loss");
    plt::show();

    torch::Tensor test_input = torch::tensor({{4.0, 2.5}});
    double prediction = torch::sigmoid(model.forward(test_input)).item<double>();
    std::cout << "Prediction for test input: " << prediction << std::endl;

    return 0;
}
