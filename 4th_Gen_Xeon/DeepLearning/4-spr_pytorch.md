
# Using Intel® Extension for Pytorch* for optimization and Performance boost

Intel® Extension for PyTorch* adds optimizations for extra performance when running PyTorch on Intel hardware.  Most of the optimizations will be included in stock PyTorch releases eventually.  The intention of the extension is to deliver up-to-date features and optimizations for PyTorch on Intel hardware.  Examples include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

Intel® Extension for PyTorch* has been released as an open–source project at [https://github.com/intel/intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch)

## Using PyTorch*, a Deep Learning Framework

Make sure PyTorch is installed so that the extension will work properly. For more information on installation refer to: [https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/installation.html#](https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/installation.html#)

### Deploying PyTorch

Reference: [https://intel.github.io/intel-extension-for-pytorch/latest/index.html](https://intel.github.io/intel-extension-for-pytorch/latest/index.html)

Environment: Python 3.7 or above

Step 1: Visit the official PyTorch website: [https://pytorch.org/](https://pytorch.org/)

Step 2: Select CPU

Currently, Intel oneDNN is integrated into the official version of PyTorch, so there is no need for an additional installation to have accelerated performance on the Intel&reg; Xeon&reg; Scalable Processor platform. Select &ldquo;CPU&rdquo; for Compute Platform. See the figure below for details.

![](images/torch-cpu.png)


Step 3: Installation

``` 
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

```

### Intel® Extension for PyTorch* Installation


You can use either of the following commands to install Intel® Extension for PyTorch*:

```
python -m pip install intel_extension_for_pytorch
```

```
python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
```

## Getting Started

Minor code changes are required for users to get started with Intel® Extension for PyTorch*. Both PyTorch imperative mode and TorchScript mode are supported. 

` For inference ` applies the ipex.optimize function to the model object. 

` For training ` applies the ipex.optimize function to the model object, as well as an optimizer object.

The following code snippet shows training code with BF16/FP32 data types.  More examples for training and inference are available at [Example Page](https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/examples.html)

```
import torch
import intel_extension_for_pytorch as ipex

model = Model()
model = model.to(memory_format=torch.channels_last)
criterion = ...
optimizer = ...
model.train()

# For Float32
model, optimizer = ipex.optimize(model, optimizer=optimizer)

# For BFloat16
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)

# Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
data = data.to(memory_format=torch.channels_last)
optimizer.zero_grad()
output = model(data)
```

### Optimization Recommendations for Training and Inferencing PyTorch-based Deep Learning Models

Although default primitives of PyTorch and Intel® Extension for PyTorch* are highly optimized, there are additional configuration options that can improve performance. Most can be applied by a launch script that automates setting configuration options, mainly for the following:

1. OpenMP library: [Intel OpenMP library (default) | GNU OpenMP library]
2. Memory allocator: [PyTorch default memory allocator | Jemalloc | TCMalloc (default)]
3. Number of instances: [Single instance (default) | Multiple instances]

For more details refer to [Launch Script Usage Guide](https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/performance_tuning/launch_script.html)

Apart from launch scripts there are other hardware settings that include configuring the structure of Intel CPUs, as well as Non-Uniform Memory Access (NUMA).  Software configuration can be set to take advantage of Channels Last memory format, OpenMP, numactl to fully utilize CPU computation resources with Intel® Extension for PyTorch* to boost performance.  For more details refer to [Performance Tuning Guide](https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/performance_tuning.html)

### Enabling BFLOAT16 Inference 

Intel® 4th Generation Intel® Xeon® Scalable Processors support accelerating AI inference by using low precision data types such as BFloat16 and INT8 based on the Intel® Deep Learning Boost and Intel® Advanced Matrix Extension(AMX). There are several instructions such as AMX_BF16, AMX_INT8, AVX512_BF16, AVX512_VNNI to accelerate AI models.

#### Auto Mixed Precision(AMP)

` torch.cpu.amp ` provides convenience for auto data type conversion at runtime. Deep learning workloads can benefit from lower-precision floating point data types such as torch.float16 or torch.bfloat16, because of its lighter calculation workload and smaller memory usage. The Auto Mixed Precision (AMP) feature automates the tuning of data type conversions over all operators.

#### Steps to enable AMX_BF16

To check whether given cpu machine supports AMX_BF16 instructions, use ` lscpu ` command.

` torch.cpu.amp.autocast ` allows scopes of your script to run with mixed precision. In these scopes, operations run in a data type chosen by the autocast class to improve performance while maintaining accuracy.  The following simple network should show a speedup with mixed precision:

```
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        return self.conv(x)
        
```

` torch.cpu.amp.autocast ` is designed to be a context manager that allow the scope of your script to run with mixed precision. AMX_BF16 are newer and have more advanced intrinsics than AVX512_BF16. They offer better performance to support AI applications. Therefore, for the data type BFloat16, AMX_BF16 has the highest execution priority. The AI frameworks optimized by Intel will choose AMX_BF16 first. If it is not available, then AVX512_BF16 will be chosen.
For more details refer to [Auto Mixed Precision (AMP)](https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/features/amp.html?highlight=amp)

```

model = SimpleNet().eval()
x = torch.rand(64, 64, 224, 224)
with torch.cpu.amp.autocast():
    y = model(x)
    
```
To Check whether AMX_BF16 is enabled, check for ` avx512_core_amx_bf16 ` JIT Kernel usage. Review the setting ` ONEDNN_VERBOSE=1 `

The Github link to the Intel® Extension for PyTorch* is:
[Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch)
