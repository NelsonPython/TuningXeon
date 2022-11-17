
## Environment

Tested hardware and software for this tuning guide include:

### Hardware

The configuration described in this article is based on 4th Generation Intel® Xeon® processor hardware (codenamed Saphire Rapids). The server platform, memory, hard drives, and network interface cards can be determined according to your usage requirements.

| Hardware | Model |
|----------------------------------|------------------------------------|
| Server Platform Name/Brand/Model | Intel® Eagle Stream Server Platform |
| CPU | Intel® Xeon® PLATINUM 8479 CPU @ 2.00GHz | 
| Memory | 8*32 GB DDR5, 4800 MT/s | 

### Software

| Software | Version |
|------------------|-------------|
| Operating System | Ubuntu 22.04.1 LTS | 
| Kernel | 5.17.6 |
| Frequency Driver | intel_pstate |
| Frequency Governor | performance |

Note: The configuration described in this article is based on 4th Generation Intel Xeon processor hardware. Server platform, memory, hard drives, network interface cards can be determined according to customer usage requirements.  

### BIOS Settings

|Configuration item|Recommended value|
|------------------|-----------------|
| <b>Socket Configuration → Processor Configuration||
| &emsp; (New BIOS) Enable LP | single thread |
| &emsp; (Old BIOS) Hyper-Threading [ALL] | disable |
| <b>Socket Configuration → Advanced Power Management Configuration||
| &emsp; CPU P State Control | Set following items to 'Disable': <br> SpeedStep (Pstates), Energy Efficient Turbo, CPU Flex Ratio Override, Perf P-Limit |
| &emsp; CPU C State Control | Set following items to 'Disable': <br> CPU C1 auto demotion, CPU C1 auto undemotion, CPU C6 report, Enhanced Halt State (C1E) |
| &emsp; Package C State Control → Package C State | C0/C1 state |
| <b>Socket Configuration → Advanced Power Management Configuration → Advanced PM Tuning → Energy Perf BIAS||
| &emsp; Power Performance Tuning | OS Controls EPB |

### Memory

Use all available memory channels

### CPU

The 4th Generation of Intel&reg; Xeon&reg; Scalable processor comes with the all new Intel&reg; AMX (Advanced Matrix Extensions) instruction set. Intel&reg; AMX provides acceleration for mixed precision deep learning training and inference workloads. The 4th Generation of Intel&reg; Xeon&reg; Scalable processor provides two instruction sets viz. AMX_BF16 and AMX_INT8 which provides acceleration for bfloat16 and int8 operations respectively

Note:  To confirm that AMX_BF16 and AMX_INT8 are supported by the CPU, enter the following command on the bash terminal and look for AMX in the "flags" section. In case AMX instructions are not getting listed, consider updating the Linux kernel to 5.17 and above

```
$ cat /proc/cpuinfo
```

### Network

If cross-node training clusters are required, choose high-speed networks, such as 25G/100G, for better scalability.

### Hard drive

For high I/O efficiency, use SSDs and drives with higher read and write speeds.

## Linux Operating System Optimization

To speed up processing tune the Linux operating system for parallel programming.

### OpenMP Parameter Settings

The [OpenMP](https://www.openmp.org/) is a specification for parallel programming.  In case your application implements OpenMP based threading, experiment with the following environment variables and find the best fit values:

- OMP_NUM_THREADS = &ldquo;number of cpu cores in container&rdquo;
- KMP_BLOCKTIME = 1 or 0 (set according to actual type of model)
- KMP_AFFINITY=granularity=fine, verbose, compact,1,0

### Number of CPU cores

Consider the impact on inference performance based on the number of CPU cores being used, as follows:

&bull; When batchsize is small (in online services for instance) the increase in inference throughput gradually weakens as the number of CPU cores increases.  In practice, 8-16 CPU cores is recommended for service deployment depending on the model used.

&bull; When batchsize is large (in offline services for instance) the inference throughput can increase linearly as the number of CPU cores increases.  In practice, more than 20 CPU cores is recommended for service deployment.

``` # taskset -C xxx-xxx –p pid (limits the number of CPU cores used in service) ```

### NUMA Configuration

For NUMA-based servers, there is usually a 5-10% increase in performance when configuring NUMA on the same node compared to using it on different nodes.

``` #numactl -N NUMA_NODE -l command args ... (controls NUMA nodes running in service) ```

### Configuration of Linux Performance Governor

Efficiency is the key consideration.  Set the CPU frequency to its peak for the best performance.

``` # cpupower frequency-set -g performance ```

### CPU C-States Settings

There are several power modes available for each CPU which are collectively referred to as C-states or C-modes.  To reduce power consumption when the CPU is idle, the CPU can be placed in the low-power mode.  Disabling C-States can increase performance.

``` #cpupower idle-set -d 2,3 ```

