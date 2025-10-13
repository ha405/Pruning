# Assignment 1A - Task 0: Baseline Profiling

## Methodology

The following metrics were collected as per the assignment requirements:
- **Model Size:** The serialized size of the model's `state_dict` in megabytes (MB).
- **MACs:** The number of Multiply-Accumulate operations for a single forward pass, measured in millions (M).
- **Memory Usage:** Peak and average GPU memory consumption during inference, measured in megabytes (MB) using `torch.profiler`.
- **Latency:** The average end-to-end inference time per batch, measured in milliseconds (ms).
- **Energy Footprint:** The total energy consumed during the profiling run, measured in millijoules (mJ) using `pyJoules`.
- **Accuracy:** Top-1 and Top-5 accuracy evaluated on the respective test sets.

## Baseline Performance Results

The table below summarizes the profiling results for the unpruned VGG16-BN models.

| Metric                 | VGG16-BN on CIFAR-10 | VGG16-BN on CIFAR-100 |
| ---------------------- | -------------------- | --------------------- |
| **Model Size (MB)**    | 58.251               | 58.427                |
| **MACs (M)**           | 314.002              | 314.049               |
| **Peak Memory (MB)**   | 457.388              | 516.827               |
| **Average Memory (MB)**| 73.249               | 132.705               |
| **Latency (ms)**       | 15.461               | 13.204                |
| **Energy (mJ)**        | 77,151               | 79,778                |
| **Top-1 Accuracy (%)** | 82.92                | 63.57                 |
| **Top-5 Accuracy (%)** | 98.27                | 83.71                 |


## Verification of Baseline Accuracy

The baseline accuracies for both the CIFAR-10 (82.92% Top-1) and CIFAR-100 (63.57% Top-1) models are consistent with the values typically reported for this architecture on these datasets. This confirms that the model loading, data preprocessing, and evaluation pipelines are set up correctly for both tasks, providing a reliable baseline for the subsequent pruning experiments.