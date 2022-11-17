# Using Intel® Extension for TensorFlow* for optimization and Performance boost

Intel® Extension for TensorFlow* adds optimizations for extra performance when running TensorFlow on Intel hardware. The intention of the extension is to deliver up-to-date features and optimizations for TensorFlow on Intel hardware. Examples include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

Intel® Extension for TensorFlow* has been released as an open–source project at https://github.com/intel/intel-extension-for-tensorflow

## Install TensorFlow* and Intel® Extension for TensorFlow*

Refer to [Installation Guide](https://intel.github.io/intel-extension-for-tensorflow/latest/docs/install/experimental/install_for_cpu.html) to install TensorFlow* and Intel® Extension for TensorFlow* for CPU.   
After having installed Intel® Extension for TensorFlow*, it will be activated automatically as a plugin of stock TensorFlow.

## Optimization Recommendations for Training and Inferencing TensorFlow-based Deep Learning Models
Intel® Extension for TensorFlow* is a Python package to extend official TensorFlow, achieve higher performance. Although stock TensorFlow and the default configuration of Intel® Extension for TensorFlow* perform well, there are still something that users can do for performance optimization on specific platforms. Most optimized configurations can be automatically set by the launch script. Mainly for the following:

- NUMA Control: numactl specifies NUMA scheduling and memory placement policy
- Number of instances: [Single instance (default) | Multiple instances]
- Memory allocator: [TCMalloc | JeMalloc | default Malloc] If unspecified, launch script will choose for user.  
- Runtime environment settings: OMP_NUM_THREADS, KMP_AFFINITY, KMP_BLOCKTIME, TF_NUM_INTEROP_THREADS, TF_NUM_INTRAOP_THREADS, ITEX_AUTO_MIXED_PRECISION, ITEX_LAYOUT_OPT, etc.

Refer to [Launch Script Usage Guide](https://intel.github.io/intel-extension-for-tensorflow/latest/docs/guide/launch.html) for more details.

Refer to [Practice Guide](https://intel.github.io/intel-extension-for-tensorflow/latest/docs/guide/practice_guide.html) for better understanding of common optimization tips which are used in launch script. 

## Enabling BFLOAT16
Intel® 4th Generation Intel® Xeon® Scalable Processors support accelerating AI inference by using low precision data types such as BFloat16 and INT8 based on the Intel® Deep Learning Boost and Intel® Advanced Matrix Extension(AMX). There are several instructions such as AMX_BF16, AMX_INT8, AVX512_BF16, AVX512_VNNI to accelerate AI models.

Mixed Precision uses lower-precision data types (such as FP16 or BF16) to make models run faster and with less memory consumption during training and inference.

Stock Tensorflow provides two ways to do this, Grappler Graph Optimization [Auto Mixed Precision](https://www.tensorflow.org/guide/graph_optimization)(AMP) and [Keras mixed precision API](https://www.tensorflow.org/guide/mixed_precision).

Intel® Extension for TensorFlow* is fully compatible with both of above methods in Stock TensorFlow, and provides an Advanced Auto Mixed Precision feature which is enhanced from Auto Mixed Precision of Stock TensorFlow for better performance.

**Intel® Extension for TensorFlow\* will turn off Auto Mixed Precision of Stock TensorFlow if its own Advanced Auto Mixed Precision is enabled explicitly.**

Refer to [Advanced Auto Mixed Precision](https://intel.github.io/intel-extension-for-tensorflow/latest/docs/guide/advanced_auto_mixed_precision.html) for more details.

#### Enable Advanced AMP

There are 2 ways to enable Advanced AMP in Intel® Extension for TensorFlow*: Python API and Environment Variables, choose either one which works for you.

|Python API|Environment Variables|
|-|-|
|`import intel_extension_for_tensorflow as itex`<br><br>`auto_mixed_precision_options = itex.AutoMixedPrecosionOptions()`<br>`auto_mixed_precision_options.data_type = itex.BFLOAT16`<br><br>`graph_options = itex.GraphOptions(auto_mixed_precision_options=auto_mixed_precision_options)`<br>`graph_options.auto_mixed_precision = itex.ON`<br><br>`config = itex.ConfigProto(graph_options=graph_options)`<br>`itex.set_backend("cpu", config)`|`export ITEX_AUTO_MIXED_PRECISION=1`<br>`export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16"`<br>|

### For Inference

For a pre-trained FP32 model (resnet50 from TensorFlow Hub as an example below):

1. Install tensorflow_hub for this sample in addition to TensorFlow* and Intel® Extension for TensorFlow*
   ```
   pip install tensorflow_hub
   ```
2. Enable Advanced AMP (using environment variables as an example here)
   ```
   export ITEX_AUTO_MIXED_PRECISION=1  
   export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16"
   ```
3. Run inference with ```export ONEDNN_VERBOSE=1```, you should be able to see AVX512_BF16 and AMX_BF16 instructions are enabled.

   - Save below code as itex_sample_inf.py
   
     ```
     import os
     import tensorflow as tf
     import tensorflow_hub as tf_hub

     os.environ["TFHUB_CACHE_DIR"] = 'tfhub_models'
     model = tf_hub.KerasLayer('https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5')
     model(tf.random.uniform((1, 224, 224, 3)))
     ```
     
   - Run inference script in terminal
   
     ```
     export ONEDNN_VERBOSE=1
     python itex_sample_inf.py
     ```
     
   - Check the result
   
     ```
     onednn_verbose,exec,cpu,convolution,jit:avx512_core_amx_bf16,forward_training,src_bf16::blocked:acdb:f0 wei_bf16::blocked:Adcb16a:f0 bia_undef::undef::f0 dst_bf16::blocked:acdb:f0,attr-scratchpad:user ,alg:convolution_direct,mb1_ic3oc64_ih224oh112kh7sh2dh0ph3_iw224ow112kw7sw2dw0pw3,2.16406
     onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_bf16::blocked:abcd:f0 dst_f32::blocked:abcd:f0,,,1x112x112x64,0.875
     onednn_verbose,exec,cpu,batch_normalization,bnorm_jit:avx512_core,forward_training,data_f32::blocked:acdb:f0 diff_undef::undef::f0,attr-scratchpad:user ,flags:GCHR,mb1ic64ih112iw112,0.754883
     onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_bf16::blocked:abcd:f0,,,1x112x112x64,0.178955
     onednn_verbose,exec,cpu,pooling_v2,jit:avx512_core_bf16,forward_inference,src_bf16::blocked:acdb:f0 dst_bf16::blocked:acdb:f0 ws_undef::undef::f0,attr-scratchpad:user ,alg:pooling_max,mb1ic64_ih112oh56kh3sh2dh0ph0_iw112ow56kw3sw2dw0pw0,0.283203
     onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_bf16::blocked:cdba:f0 dst_bf16::blocked:AcdB16b64a2b:f0,,,256x64x1x1,1.02393
     onednn_verbose,exec,cpu,convolution,brgconv_1x1:avx512_core_amx_bf16,forward_training,src_bf16::blocked:acdb:f0 wei_bf16::blocked:AcdB16b64a2b:f0 bia_undef::undef::f0 dst_bf16::blocked:acdb:f0,attr-scratchpad:user ,alg:convolution_direct,mb1_ic64oc256_ih56oh56kh1sh1dh0ph0_iw56ow56kw1sw1dw0pw0,0.87207
     ```

### For Training

Same as inference, you can enable Advanced AMP either using Python API or setting environment variables for training. 

1. Enable Advanced AMP (using environment variables as an example here)
   ```
   export ITEX_AUTO_MIXED_PRECISION=1  
   export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16"
   ```
2. Run training with ```export ONEDNN_VERBOSE=1```, you should be able to see AVX512_BF16 and AMX_BF16 instructions are enabled.

   - Save below code as itex_sample_train.py

     ```
     import tensorflow as tf
     from keras.utils import np_utils

     # load data
     cifar10 = tf.keras.datasets.cifar10
     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
     num_classes = 10

     # pre-process
     x_train, x_test = x_train/255.0, x_test/255.0
     y_train = np_utils.to_categorical(y_train, num_classes)
     y_test = np_utils.to_categorical(y_test, num_classes)

     # build model
     feature_extractor_layer = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
     feature_extractor_layer.trainable = False
     model = tf.keras.Sequential([
         tf.keras.layers.Input(shape=(32, 32, 3)),
         feature_extractor_layer,
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(1024, activation='relu'),
         tf.keras.layers.Dropout(0.2),
         tf.keras.layers.Dense(num_classes, activation='softmax')
     ])
     model.compile(
       optimizer=tf.keras.optimizers.Adam(),
       loss=tf.keras.losses.CategoricalCrossentropy(),
       metrics=['acc'])

     # train model
     model.fit(x_train, y_train,
         batch_size = 128,
         validation_data=(x_test, y_test), 
         epochs=1)

     model.save('resnet_bf16_model')
     ```
     
   - Run training script in terminal
   
     ```
     export ONEDNN_VERBOSE=1
     python itex_sample_train.py
     ```
     
   - Check the result

     ```
     onednn_verbose,exec,cpu,convolution,jit:avx512_core_amx_bf16,forward_training,src_bf16::blocked:acdb:f0 wei_bf16::blocked:Adcb16a:f0 bia_bf16::blocked:a:f0 dst_bf16::blocked:acdb:f0,attr-scratchpad:user ,alg:convolution_direct,mb128_ic3oc64_ih32oh16kh7sh2dh0ph3_iw32ow16kw7sw2dw0pw3,1.802
     onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_bf16::blocked:abcd:f0 dst_f32::blocked:abcd:f0,,,128x16x16x64,6.74292
     onednn_verbose,exec,cpu,batch_normalization,bnorm_jit:avx512_core,forward_training,data_f32::blocked:acdb:f0 diff_undef::undef::f0,attr-scratchpad:user ,flags:GCHR,mb128ic64ih16iw16,2.65112
     onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_bf16::blocked:abcd:f0,,,128x16x16x64,0.845947
     onednn_verbose,exec,cpu,pooling_v2,jit:avx512_core_bf16,forward_inference,src_bf16::blocked:acdb:f0 dst_bf16::blocked:acdb:f0 ws_undef::undef::f0,attr-scratchpad:user ,alg:pooling_max,mb128ic64_ih18oh8kh3sh2dh0ph0_iw18ow8kw3sw2dw0pw0,0.195068
     onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_bf16::blocked:cdba:f0 dst_bf16::blocked:AcdB16b64a2b:f0,,,64x64x1x1,1.72998
     onednn_verbose,exec,cpu,convolution,brgconv_1x1:avx512_core_amx_bf16,forward_training,src_bf16::blocked:acdb:f0 wei_bf16::blocked:AcdB16b64a2b:f0 bia_bf16::blocked:a:f0 dst_bf16::blocked:acdb:f0,attr-scratchpad:user ,alg:convolution_direct,mb128_ic64oc64_ih8oh8kh1sh1dh0ph0_iw8ow8kw1sw1dw0pw0,5.64795
     ```

## Enabling INT8

Intel® Extension for TensorFlow* co-works with [Intel® Neural Compressor](https://intel.github.io/neural-compressor) >= 1.14.1 to provide compatible TensorFlow INT8 quantization solution support with same user experience.
