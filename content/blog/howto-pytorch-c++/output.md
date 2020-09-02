---
title: How To Run a pre-trained PyTorch model in C++
date: 2020-09-02
subject: ["ml", "howto", "pytorch", "deploy", "C++", "TorchScript"]
author: RSP
description: How To Run a pre-trained PyTorch model in C++
featimg: output.png
---

In this post we will go through the steps of running a pre-trained PyTorch model in C++ on MacOS (or other platform where you can compile C/C++). The steps are as follows

1. Convert PyTorch model (.pt file) to a TorchScript ScriptModule
2. Serialize the the Script Module to a file
3. Load the Script Module in C++
4. Build/Make the C++ application using CMake

This follows the official [PyTorch tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html) but is adapted to our Speech Commands Recognition model.

Why would we want to do something like this? There could be several reasons

1. Speed: C/C++ is known to be faster
2. Memory footprint: Python is not famous for memory footprint use
3. Targeting Edge ML (embedded systems) which don't have a lot of memory or CPU horsepower
4. Integrating into a native app (iOS or MacOS)
5. Production cloud service

## Convert PyTorch model (.pt file) to a TorchScript ScriptModule

### What is TorchScript?

An intermediate representation of a PyTorch model that can be run in C++. We can obtain TorchScript of a PyTorch model (subclass of nn.Module) by

1. Tracing an existing module
2. Use scripting to directly compile a module

Tracing is accomplished by creating some sample inputs and then calling the forward method and recording / tracing by a function called torch.jit.trace. The scripting method is useful when there is some control flow (data dependent execution) in the model. We show the tracing method below for our Speech Commands quantized model.

#### LOAD MODEL WEIGHTS, QUANTIZE WEIGHTS TO 8-BIT

```python
PATH = "./models/speech_commands_model.pt"
nnModel = models.SpeechCommandsModel().to(device)       # Instantiate our model and move model to GPU if available
nnModel.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
quantized_model = torch.quantization.quantize_dynamic(
    nnModel, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
quantized_model.eval()
nnModel.eval()
```

#### MODEL TRACING WITH INPUTS

```python
testFiles = utils.get_filenames('files',searchstr='SCRIC20*')
(X,y) = scd.get_file_features(testFiles[0], padLR=False)
traced_model = torch.jit.trace(quantized_model, (X))
print(traced_model)
```

    SpeechCommandsModel(
      original_name=SpeechCommandsModel
      (conv1): Conv2d(original_name=Conv2d)
      (bn1): BatchNorm2d(original_name=BatchNorm2d)
      (conv2): Conv2d(original_name=Conv2d)
      (bn2): BatchNorm2d(original_name=BatchNorm2d)
      (fc1): Linear(
        original_name=Linear
        (_packed_params): RecursiveScriptModule(original_name=LinearPackedParams)
      )
      (bn3): BatchNorm1d(original_name=BatchNorm1d)
      (fc2): Linear(
        original_name=Linear
        (_packed_params): RecursiveScriptModule(original_name=LinearPackedParams)
      )
    )

#### TORCHSCRIPT SCRIPTMODULE INTERMEDIATE REPRESENTATION

```python
print(traced_model.code)
```

    def forward(self,
        input: Tensor) -> Tensor:
      _0 = self.fc2
      _1 = self.bn3
      _2 = self.fc1
      _3 = self.bn2
      _4 = self.conv2
      _5 = (self.bn1).forward((self.conv1).forward(input, ), )
      input0 = torch.max_pool2d(_5, [2], annotate(List[int], []), [0, 0], [1, 1], False)
      input1 = torch.relu(input0)
      _6 = (_3).forward((_4).forward(input1, ), )
      input2 = torch.max_pool2d(_6, [2], annotate(List[int], []), [0, 0], [1, 1], False)
      x = torch.relu(input2)
      x0 = torch.view(x, [-1, 1536])
      x1 = torch.relu((_1).forward((_2).forward(x0, ), ))
      _7 = torch.log_softmax((_0).forward(x1, ), 1, None)
      return _7

#### VERIFY THAT OUTPUTS ARE MATCHING

```python
print(traced_model(X))             # TORCHSCRIPT version of QUANTIZED MODEL
print(quantized_model(X))          # QUANTIZED MODEL
print(nnModel(X))                  # ORIGINAL MODEL
```

    tensor([[-3.4839, -4.6787, -1.6807, -6.9479, -0.3579, -4.0282, -8.1275, -5.0048,
             -6.2447, -5.5977, -3.1421]])
    tensor([[-3.4839, -4.6787, -1.6807, -6.9479, -0.3579, -4.0282, -8.1275, -5.0048,
             -6.2447, -5.5977, -3.1421]])
    tensor([[-3.4179, -4.6061, -1.6919, -6.9600, -0.3590, -4.0849, -8.1481, -4.9647,
             -6.2618, -5.5800, -3.1247]], grad_fn=<LogSoftmaxBackward>)

### What is special about this TorchScript code?

According to the official tutorial, there are several advantages to having a intermediate representation of the model graph

1. TorchScript code can be invoked in its own interpreter and many requests can be
   processed on the same instance simultaneously due to absence of a global instance lock
2. This format allows to save the whole model to disk and load it
   into another environment
3. TorchScript gives a representation in which we can do compiler
   optimizations
4. TorchScript allows to interface with many backend/device runtimes

Let's take their word for it and keep these in mind for now and remind ourselves later when we see the usecase in action.

## Serialize the the Script Module to a file

```python
traced_model.save('models/traced_qsc.zip')
loaded = torch.jit.load('models/traced_qsc.zip')
print(loaded)
print(loaded.code)
```

    RecursiveScriptModule(
      original_name=SpeechCommandsModel
      (conv1): RecursiveScriptModule(original_name=Conv2d)
      (bn1): RecursiveScriptModule(original_name=BatchNorm2d)
      (conv2): RecursiveScriptModule(original_name=Conv2d)
      (bn2): RecursiveScriptModule(original_name=BatchNorm2d)
      (fc1): RecursiveScriptModule(
        original_name=Linear
        (_packed_params): RecursiveScriptModule(original_name=LinearPackedParams)
      )
      (bn3): RecursiveScriptModule(original_name=BatchNorm1d)
      (fc2): RecursiveScriptModule(
        original_name=Linear
        (_packed_params): RecursiveScriptModule(original_name=LinearPackedParams)
      )
    )
    def forward(self,
        input: Tensor) -> Tensor:
      _0 = self.fc2
      _1 = self.bn3
      _2 = self.fc1
      _3 = self.bn2
      _4 = self.conv2
      _5 = (self.bn1).forward((self.conv1).forward(input, ), )
      input0 = torch.max_pool2d(_5, [2], annotate(List[int], []), [0, 0], [1, 1], False)
      input1 = torch.relu(input0)
      _6 = (_3).forward((_4).forward(input1, ), )
      input2 = torch.max_pool2d(_6, [2], annotate(List[int], []), [0, 0], [1, 1], False)
      x = torch.relu(input2)
      x0 = torch.view(x, [-1, 1536])
      x1 = torch.relu((_1).forward((_2).forward(x0, ), ))
      _7 = torch.log_softmax((_0).forward(x1, ), 1, None)
      return _7

## Load the Script Module in C++

The PyTorch C++ API, also known as LibTorch, is used to load the serialized PyTorch model in C++. The LibTorch distribution consists of shared libraries, headers and build config files. CMake is the recommended build configuration tool.

We have a few (boring install) steps to do now

1. Download and install [LibTorch](https://pytorch.org/cppdocs/installing.html). Just a measly 2 GB unzipped.
2. Install [CMake](https://cmake.org/download/) if you don't have it already

Next let us try to compile a fairly simple C++ program which loads the serialized ScriptModule (.zip file that was created earlier) and then passes a random input tensor to the model for prediction.

```cpp
#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  std::cout << "Model "<< argv[1]<<" loaded fine\n";

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::randn({1, 1, 64, 101}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output << "\n";
  int y_hat = output.argmax(1).item().toInt();
  std::cout << "Predicted class: " << y_hat <<"\n";
}
```

### Build Procedure

We have the following directory structure

```Terminal
projects
  example-app
    example-app.cpp
    CMakeList.txt
```

The CMakeList.txt consists of the following commands

```CMAKE
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
```

We then issue the following terminal commands (change for your setup as needed!):

```Terminal
cd projects/example-app
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/ragh/Downloads/libtorch ..
cmake --build . --config Release
./example-app ../../../models/traced_qsc.zip
./example-app ../../../models/traced_qsc.zip
Model ../../../models/traced_qsc.zip loaded fine
Columns 1 to 10-7.8962 -8.8210 -4.3701 -7.6351 -4.3408 -6.9469 -5.3084 -4.0581 -4.6869 -7.8740

Columns 11 to 11-0.0613
[ CPUFloatType{1,11} ]
Predicted class: 10
```

## Conclusion

Using tracing, we created a serialized TorchScript ScriptModule of our speech commands model. We then loaded this model using the C++ API and performed a model prediction in C++. The output we got to a random input was classified as Unknown/Background class, which is expected.

I should say that I did attempt to try this same example on MacOS and it did not work out (linker error). However, I gave it another shot on the Ubuntu box and it worked fine.

One challenge that we still need to address is how to pipe inputs to the C++ environment. Torchaudio is absent in the C++ realm, so feature processing will be an issue or a lot of work. This motivates us to look into things like CoreML (in Apple ecosystem) and the ONNX runtimes in future posts.

While I do believe PyTorch is the way to go as far as model development goes, it may not be a bad idea to leverage tensorflow lite and things like that in the TF ecosystem which people have shown to run on resource-constrained MCUs.

While this may not the most exciting/interesting topic to post on, it is very critical if you want an efficient path to deployment.
