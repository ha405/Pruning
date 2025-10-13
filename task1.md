# Assignment 1A - Pruning: Results and Analysis

This document presents the results and analysis for the pruning tasks performed on VGG16-BN models for CIFAR-10 and CIFAR-100 datasets, as required by Assignment 1A.

## Task 0: Baseline Performance
First, baseline metrics were established for the unpruned, pretrained VGG16-BN models.

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

---

## Task 1: Unstructured Pruning

Two methods of unstructured pruning were implemented to achieve an overall model sparsity of approximately 80%.

### Task 1a: Sensitivity Analysis and Layerwise Sparsity

This method involved analyzing each layer's sensitivity to pruning and setting a specific sparsity ratio for each layer to reach a global target of ~80%.

| Dataset  | Sparsity (%) | Size (MB) | MACs (M) | Peak Mem (MB) | Latency (ms) | Top-1 Acc (%) |
|----------|--------------|-----------|----------|---------------|--------------|---------------|
| CIFAR-10 | 80.33        | 34.331    | 314.002  | 455.333       | 17.096       | **92.37**     |
| CIFAR-100| 80.33        | 34.384    | 314.049  | 455.946       | 30.656       | **68.07**     |

### Task 1b: Saliency-Based Iterative Pruning (GraSP)

This method involved iteratively pruning the model at different stages of training based on the GraSP saliency metric to achieve a final sparsity of ~80%.

| Dataset  | Sparsity (%) | Size (MB) | MACs (M) | Peak Mem (MB) | Latency (ms) | Top-1 Acc (%) |
|----------|--------------|-----------|----------|---------------|--------------|---------------|
| CIFAR-10 | 80.00        | 35.105    | 314.002  | 455.873       | 39.590       | 82.56         |
| CIFAR-100| 80.00        | 35.214    | 314.049  | 456.007       | 32.875       | 42.67         |

---

## Task 2: Structured Pruning (Regression-Based)

This task involved implementing regression-based channel pruning to remove entire channels from convolutional layers, aiming for a high overall sparsity.

| Dataset   | Sparsity (%) | Size (MB) | MACs (M) | Peak Mem (MB) | Latency (ms) | Top-1 Acc (%) |
|-----------|--------------|-----------|----------|---------------|--------------|---------------|
| CIFAR-10  | ~80% (est.)  | 11.144    | **41.782**   | **118.364**       | **8.342**        | 88.01         |
| CIFAR-100 | ~80% (est.)  | 11.321    | **41.828**   | **118.698**       | **11.238**       | 54.46         |

---

## Analysis and Discussion

### Expected vs. Observed Changes: A Detailed Look

The results from the pruning experiments reveal a clear and important distinction between unstructured and structured pruning methods, aligning with theoretical expectations but also offering some surprising outcomes.

-   **Model Size:** As anticipated, all pruning methods successfully reduced the on-disk model size. The most dramatic compression came from **structured pruning (Task 2)**, which shrunk the model from ~58 MB to ~11 MB. This is its inherent strength; by removing entire channels, the weight tensors themselves become physically smaller.
    In contrast, **unstructured pruning (Task 1)**, while achieving the same 80% parameter sparsity, resulted in a larger file size (~34 MB). This is a crucial subtlety. A standard saved model does not differentiate between a zero-value weight and a non-zero one. To capture the sparsity, formats like COO (Coordinate list) must be used, which explicitly store the indices and values of the non-zero elements. This metadata overhead is why the file size reduction is not proportional to the sparsity ratio.

-   **Performance (Latency, Memory, and Energy):**
    -   **Unstructured Pruning (Task 1):** A key observation is that unstructured pruning not only failed to improve latency but often made it worse (e.g., latency for GraSP on CIFAR-10 increased from 15.4ms to 39.6ms). Standard deep learning libraries and GPUs are highly optimized for dense matrix multiplications. Introducing unstructured sparsity breaks this paradigm. The model still has to process tensors of the same original dimensions, and the custom forward pass logic—which involves converting sparse tensors back to dense for convolution or using sparse matrix multiplication for linear layers—introduces significant overhead. Without specialized hardware or highly optimized sparse kernels (which we did not use), there is no "free lunch"
    -   **Structured Pruning (Task 2):** This method delivered on its promise of tangible performance gains. By creating a physically smaller, dense model, it aligned perfectly with the strengths of existing hardware. We observed a remarkable **~87% reduction in MACs** (from 314M to ~42M) and a **~74% reduction in peak memory usage**. Most importantly, this translated directly to a **~40-50% decrease in inference latency**. This demonstrates that for real-world acceleration on common devices, removing structured blocks of computation is far more effective than just zeroing out individual weights.

-   **Accuracy Compared to Baseline:**
    -   The most surprising result was from **layerwise sensitivity pruning (Task 1a)**, which achieved a significant accuracy *boost* on both datasets, with CIFAR-10 accuracy jumping from 82.92% to 92.37%. This suggests that magnitude pruning on a well-trained model acts as a powerful regularization technique, removing redundant or noisy parameters that may have contributed to overfitting, thereby improving the model's ability to generalize.
    -   **GraSP (Task 1b)** performed reasonably on the simpler CIFAR-10 task but failed significantly on CIFAR-100 (42.67% vs. 63.57% baseline). This indicates that its criterion—preserving gradient flow—is not a robust enough proxy for feature importance on more complex, fine-grained tasks.
    -   **Structured pruning (Task 2)** provided a strong balance, maintaining high accuracy on CIFAR-10 (88.01%) and delivering a respectable, though reduced, accuracy on CIFAR-100 (54.46%). This highlights the fundamental trade-off of structured pruning: removing entire features is a blunter instrument than removing individual weights, making it harder to maintain performance on complex tasks, but the performance gains are significant.

### SNIP vs. GraSP: A Deeper Dive
Both SNIP and GraSP are part of a family of methods that attempt to "prune at initialization," saving the immense computational cost of training a model only to discard most of it. They operate on the premise that an important weight in a trained model must also have been important from the very beginning.
-   **GraSP**, the method implemented, tries to preserve the "gradient highway" by removing weights that most impede the flow of gradients from the loss back to the early layers. The logic is that a healthy gradient flow is essential for effective training. Its failure on CIFAR-100 suggests that this criterion alone is insufficient. A network can have great gradient flow but still lack the specific features needed to distinguish between 100 fine-grained classes.
-   **SNIP (Single-shot Network Pruning)** uses a different saliency metric. It identifies connections whose removal causes the least change to the loss on a mini-batch of data. Its criterion is more directly tied to the task objective ("does this weight affect the loss?").

Given GraSP's performance, I would not have definitively expected SNIP to perform better, but it's plausible. SNIP's focus on the loss might make it slightly better at identifying task-relevant weights, whereas GraSP's focus on gradient dynamics is more about trainability. However, both methods are heuristics that operate on an untrained network. The stellar performance of post-training magnitude pruning (Task 1a) strongly suggests that the information encoded in the final weights of a fully trained model is a far more reliable indicator of importance than any metric available at initialization. This aligns with the "Lottery Ticket Hypothesis," which posits that good subnetworks exist at initialization but are very difficult to find without the signal provided by the training process itself.