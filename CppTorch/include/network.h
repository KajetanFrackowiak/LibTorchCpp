#pragma once

#include <iostream>
#include <torch/torch.h>

struct NetImpl : torch::nn::Module {
    NetImpl(int fc1_dims, int fc2_dims) 
        : fc1(register_module("fc1", torch::nn::Linear(fc1_dims, fc1_dims))),
          fc2(register_module("fc2", torch::nn::Linear(fc1_dims, fc2_dims))),
          out(register_module("out", torch::nn::Linear(fc2_dims, 1))) {}

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = out(x);
        return x;
    }

    torch::nn::Linear fc1, fc2, out;
};

TORCH_MODULE(Net);
