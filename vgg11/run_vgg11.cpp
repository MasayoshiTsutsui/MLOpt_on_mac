#include <torch/script.h>
#include <iostream>
#include <time.h>

int main() {
  torch::jit::script::Module module; 
  try {
    module = torch::jit::load("../vgg11_h224_w224_c3.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // repeat forward 10 times and measure the average time
  double total_time = 0.0;
  for (int i = 0; i < 100; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto output = module.forward(inputs);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    total_time += diff.count();
    std::cout << output.toTensor().slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  }
  std::cout << "average time: " << total_time / 100 << "s\n";
  std::cout << "ok\n";
  return 0;
}