#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>


int main() {
    // Load the traced script module
    try {
        torch::jit::script::Module net = torch::jit::load("../models/net.pt");
    } catch (const torch::Error& e) {
        std::cerr << "Error " << e.what() << "\n";
    } catch (...) {
        std::cerr << "Unknown error" << "\n";
    }
    // Generate a random input tensor
    torch::Tensor x = torch::randn({1, 5});
    std::cout << x << std::endl;
    return 0;
}