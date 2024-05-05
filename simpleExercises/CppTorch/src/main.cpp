#include "network.h"  // Replace with the actual name of your header file

#include <iostream>
#include <torch/torch.h>

int main() {
    // Create an instance of the Net module
   Net net(50, 10);  // Assuming 100 and 50 are the dimensions for fc1 and fc2
   std::cout << net << "\n\n";

   torch::Tensor x, output;
   x = torch::randn({2, 50});
   output = net->forward(x);

   std::cout << output;
    return 0;
}