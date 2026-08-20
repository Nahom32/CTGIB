# TGIB GPU Optimization Roadmap

## Overview

The current TGIB implementation suffers from severe GPU starvation: the GPU is idle for the vast majority of training time due to CPU-bound neighbor lookups, excessive per-element CPU→GPU transfers, batch-size-1 training, and synchronous metric computation after every step. The optimizations below are ordered by expected impact (highest first) and grouped into phases for incremental adoption.

---

## Phase 1 — Eliminate the Biggest Bottlenecks

### 1.1 True Mini-Batch Training (Critical)

**Problem:** The training loop in `learn_edge.py:236` iterates `for k in range(1, len(train_src_l))` — processing a **single event per forward/backward pass** despite `--bs 200` being defined but never used. The GPU receives one tiny computation graph, executes it, then waits for the CPU to prepare the next sample. With a single event, the compute is negligible relative to launch overhead.

**Fix:** Restructure the training loop to process `BATCH_SIZE` events per iteration. Because TGIB uses a growing temporal context (`train_src_l[:k+1]`), true random batching is not possible — but we can:

- Process consecutive slices of `BATCH_SIZE` events at each step, each time using the full history up to the last event in the slice.
- Accumulate gradients across the mini-batch before calling `optimizer.step()`.
- Use `torch.cuda.amp.GradScaler` for mixed-precision to further reduce memory and increase throughput.

**Files:** `learn_edge.py` (training loop), `module.py` (forward pass must accept batched inputs)

### 1.2 Batched Neighbor Lookup (Critical)

**Problem:** `graph.py:get_temporal_neighbor` runs a Python `for` loop over every node individually (`learn_edge.py:107`), performing binary search per node in pure Python/NumPy. This is the dominant CPU bottleneck — for a single forward pass with 2-hop recursion, hundreds of individual neighbor lookups are executed sequentially.

**Fix:** Rewrite `get_temporal_neighbor` to operate on arrays in bulk:

- Pre-sort all adjacency lists once (already done in `init_off_set`).
- Replace the Python `for` loop with vectorized NumPy operations using `np.searchsorted` on the sorted timestamp arrays for batch binary search.
- Alternatively, use Numba JIT compilation (`@njit`) on the `find_before` inner loop.
- Consider moving the adjacency structure to a CUDA-based temporal neighbor finder (e.g., using `torch.cumsearch` or custom CUDA kernels).

**Files:** `graph.py`, `learn_edge.py`

### 1.3 Reduce CPU→GPU Transfer Granularity (Critical)

**Problem:** The forward pass in `module.py:463-579` makes ~24 individual `torch.from_numpy(...).to(device)` calls, each creating a separate CUDA transfer. The recursive `tem_conv` (`module.py:605-661`) adds ~2 more per recursion level per node — with `num_layers=2` and `num_neighbors=20`, a single event triggers ~421 separate CPU→GPU transfers. Each transfer incurs kernel launch overhead and PCIe latency.

**Fix:**

- Pre-allocate contiguous GPU tensors for all neighbor data before the forward pass.
- Batch all `from_numpy` + `.to(device)` calls: collect all numpy arrays that need transfer, concatenate them, transfer once, then split on GPU.
- Cache the `node_raw_embed` and `edge_raw_embed` lookups: since embeddings are frozen (`freeze=True`), look up all needed IDs in one `torch.cat` call, transfer once, and index on GPU.
- Eliminate the per-`tem_conv` call transfer by passing pre-transferred tensors through the recursion.

**Files:** `module.py` (forward, tem_conv), `learn_edge.py`

---

## Phase 2 — Reduce Redundant Computation

### 2.1 Avoid Synchronous Metric Computation Every Step (High)

**Problem:** After every single training step (`learn_edge.py:256-265`), the code does:
```python
pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), ...])
```
This forces a GPU→CPU synchronization **every iteration**, stalling the GPU pipeline. Computing `average_precision_score` and `roc_auc_score` on 2-element arrays is also statistically meaningless.

**Fix:**

- Accumulate predictions in a list during training; compute metrics only at epoch end.
- Wrap metric computation in `torch.cuda.event` timing to confirm the stall.
- Alternatively, compute loss-only metrics during training and defer AP/AUC to a periodic evaluation pass.

**Files:** `learn_edge.py`

### 2.2 Cache Repeated Embedding Lookups in `tem_conv` (High)

**Problem:** `tem_conv` is called recursively and also called separately for the target node, its fake counterpart, and all 1-hop/2-hop neighbors. The same node IDs may be looked up multiple times across these calls. Each call independently does `torch.from_numpy(u_emb).long().to(device)` and `self.node_raw_embed(u_emb_th)`.

**Fix:**

- Maintain a memoization cache (LRU or dictionary) mapping node IDs to their GPU-resident embeddings.
- Before the forward pass, collect all unique node IDs that will be needed (from the 2-hop neighborhood), do a single batched `node_raw_embed` lookup, and store results in a GPU tensor indexed by node ID.
- This eliminates redundant embedding lookups and reduces the number of `.to(device)` transfers from O(num_nodes × num_layers) to O(unique_nodes).

**Files:** `module.py`

### 2.3 Avoid Repeated `tem_conv` Recursion for Shared Subgraphs (High)

**Problem:** When computing the 1-hop neighborhood of `u_emb_k`, the code calls `tem_conv` on each of the `num_neighbors` neighbors. Then for the 2-hop neighborhood, `tem_conv` is called again on `num_neighbors^2` nodes. Many of these nodes overlap — the same node's embedding is computed from scratch each time it appears.

**Fix:**

- Flatten all unique node IDs across all hops before calling `tem_conv`.
- Call `tem_conv` once on the deduplicated set of nodes.
- Index into the result to reconstruct the per-neighbor structure.
- This reduces the number of recursive neighbor lookups from O(num_neighbors^2) to O(unique_nodes) per forward pass.

**Files:** `module.py` (forward, tem_conv)

---

## Phase 3 — Leverage GPU Parallelism

### 3.1 Mixed Precision Training (AMP) (Medium)

**Problem:** All computations run in FP32, doubling memory usage and reducing GPU throughput compared to FP16/BF16 on Tensor Cores.

**Fix:**

- Wrap the forward pass in `torch.cuda.amp.autocast()` and use `torch.cuda.amp.GradScaler` for the optimizer.
- The TGIB model dimensions (100-dim embeddings, small MLP heads) are well-suited for mixed precision.
- Expected speedup: 1.5-2x on modern GPUs with Tensor Core support.

**Files:** `learn_edge.py` (training loop)

### 3.2 `torch.compile()` for Model Graph Optimization (Medium)

**Problem:** The model's forward pass is not compiled — PyTorch executes each operation with individual kernel launches.

**Fix:**

- Apply `torch.compile(tgib, mode="reduce-overhead")` after model initialization.
- This fuses small operations (element-wise additions, activations, layer norms) into single CUDA kernels, reducing launch overhead.
- Note: the recursive `tem_conv` with dynamic shapes may limit compile effectiveness — consider restructuring to make shapes static first.

**Files:** `learn_edge.py`, `module.py`

### 3.3 CUDA Graphs for Fixed-Shape Operations (Low-Medium)

**Problem:** The attention computation in `AttnModel.forward` launches many small CUDA kernels (reshape, permute, bmm, softmax) that could be captured as a CUDA graph.

**Fix:**

- Identify the most frequently called operations with static shapes (attention layers, MergeLayer, TimeEncode).
- Capture them as CUDA graphs to eliminate per-kernel launch overhead.
- Requires careful handling of dynamic shapes in the rest of the model.

**Files:** `module.py`

### 3.4 Custom CUDA Kernel for Temporal Neighbor Sampling (Medium)

**Problem:** The binary-search-based temporal neighbor finder in `graph.py` is the primary CPU bottleneck. Even with Numba JIT, it remains on CPU and serializes the pipeline.

**Fix:**

- Write a custom CUDA kernel that performs batched temporal neighbor sampling:
  - Store adjacency lists in a GPU-resident CSR format.
  - Each thread block handles one source node; threads within the block cooperatively search for neighbors before the cutoff time.
  - Use `torch.utils.cpp_extension` to compile the kernel.
- Alternatively, use PyTorch's `torch.searchsorted` on GPU-resident sorted timestamp arrays for approximate batch binary search.

**Files:** `graph.py`, new file for CUDA extension

---

## Phase 4 — Data Pipeline & Memory Optimization

### 4.1 Pin Memory and Persistent Workers (Low-Medium)

**Problem:** The codebase does not use `torch.utils.data.DataLoader`. All data is loaded as raw numpy arrays and manually sliced, missing out on PyTorch's optimized data loading infrastructure.

**Fix:**

- Create a `torch.utils.data.Dataset` subclass wrapping the training arrays.
- Use `DataLoader` with `pin_memory=True`, `num_workers=4`, and `persistent_workers=True`.
- `pin_memory=True` enables faster CPU→GPU transfers by using page-locked memory.
- Even with batch-size-1 training, `num_workers > 0` moves neighbor lookups to background processes, overlapping CPU work with GPU computation.

**Files:** `learn_edge.py`, new file for dataset class

### 4.2 Gradient Accumulation for Effective Larger Batches (Low-Medium)

**Problem:** Even with true mini-batch restructuring (Optimization 1.1), the effective batch size may be limited by GPU memory due to the growing computation graph.

**Fix:**

- Implement gradient accumulation: run N micro-batches of size 1, accumulate gradients, then step the optimizer every N steps.
- This simulates a larger batch size without additional memory for activations.
- Combine with AMP (Optimization 3.1) for maximum throughput.

**Files:** `learn_edge.py`

### 4.3 Precompute Adjacency Structures on GPU (Low)

**Problem:** The `NeighborFinder` stores adjacency data as Python lists of numpy arrays. Each `get_temporal_neighbor` call creates new numpy output arrays, which are then transferred to GPU.

**Fix:**

- Store the adjacency structure (node indices, timestamps, edge indices, offsets) as CUDA tensors at initialization.
- Perform neighbor lookups directly on GPU using `torch.searchsorted` on the sorted timestamp arrays.
- Store output arrays as pre-allocated GPU tensors to avoid allocation overhead.

**Files:** `graph.py`, `learn_edge.py`

---

## Phase 5 — Algorithmic / Architecture Optimizations

### 5.1 Approximate 2-Hop Neighborhood (Medium)

**Problem:** The 2-hop neighborhood computation is O(num_neighbors^2) per node — with `num_neighbors=20`, this means 400 neighbor lookups and embedding computations for the 2-hop layer alone. The recursive `tem_conv` compounds this further.

**Fix:**

- Use importance sampling or graph attention to select a subset of 2-hop neighbors rather than exhaustively expanding all.
- Alternatively, reduce `num_neighbors` for the 2-hop layer (e.g., 5 instead of 20) while keeping the 1-hop layer at 20.
- Consider a single-hop architecture for faster training with periodic full-graph evaluation.

**Files:** `module.py`, `learn_edge.py`

### 5.2 Checkpointed Recursion (Low-Medium)

**Problem:** The recursive `tem_conv` stores the full computation graph for all levels, consuming GPU memory for intermediate activations that are only needed during the backward pass.

**Fix:**

- Apply `torch.utils.checkpoint.checkpoint` to the recursive `tem_conv` calls.
- This trades compute (recomputing during backward) for memory, enabling larger effective batch sizes.
- Particularly effective when combined with gradient accumulation.

**Files:** `module.py`

---

## Implementation Priority Matrix

| # | Optimization | Impact | Effort | Risk |
|---|---|---|---|---|
| 1.1 | True mini-batch training | **Critical** | High | Medium |
| 1.2 | Batched neighbor lookup | **Critical** | Medium | Low |
| 1.3 | Reduce transfer granularity | **Critical** | Medium | Low |
| 2.1 | Defer metric computation | High | Low | None |
| 2.2 | Cache embedding lookups | High | Medium | Low |
| 2.3 | Deduplicate tem_conv calls | High | Medium | Medium |
| 3.1 | Mixed precision (AMP) | Medium | Low | Low |
| 3.2 | torch.compile | Medium | Low | Medium |
| 3.3 | CUDA graphs | Low-Med | High | Medium |
| 3.4 | CUDA neighbor kernel | Medium | High | Medium |
| 4.1 | DataLoader + pin_memory | Low-Med | Medium | Low |
| 4.2 | Gradient accumulation | Low-Med | Low | Low |
| 4.3 | GPU-resident adjacency | Low | High | Medium |
| 5.1 | Approximate 2-hop | Medium | High | High |
| 5.2 | Checkpointed recursion | Low-Med | Low | Low |

---

## Expected Outcomes

- **Phase 1 alone** should reduce training time by **5-10x** by eliminating the batch-size-1 bottleneck, the Python-loop neighbor lookups, and the hundreds of per-element GPU transfers.
- **Phase 2** adds another **1.5-2x** by removing redundant computation and synchronous GPU stalls.
- **Phase 3** provides a further **1.5-2x** through kernel fusion and mixed precision.
- **Phases 4-5** provide incremental gains and enable scaling to larger datasets.

The combined effect should transform training from GPU-idle to GPU-saturated, potentially achieving **15-30x end-to-end speedup** on the same hardware.
