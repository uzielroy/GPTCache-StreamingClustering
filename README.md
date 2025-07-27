# 🚀 GPTCache Streaming Clustering
*Adaptive Embedding Caching for Large Language Models*

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![Conda Environment](https://img.shields.io/badge/Conda-env.yaml-success)](env.yaml)

> **“GPTCache Streaming Clustering”** - a streaming‑clustering cache policy for LLM embeddings built on top of [GPTCache](https://github.com/zilliztech/GPTCache).

---

## 📌 Overview  
This repository implements a **streaming clustering** caching policy that groups similar embeddings into dynamically updated clusters.  
The result: **lower cache‑lookup latency** and **higher throughput** for LLM‑powered applications.

---

## 📂 Directory Layout
    ├── env.yaml                     # Reproducible Conda environment
    ├── download_models.py           # One‑click model & dataset fetcher
    ├── benchmark_clustering_speedup.py  # Synthetic benchmark (clustering vs FAISS)
    ├── gptcache_benchmark.py        # Real‑world benchmark (AG News, CNN/DailyMail)
    ├── results/                     # All benchmark outputs live here
    └── README.md                    # You are here 📖

---

## 🛠️ Setup

### 1️⃣ Clone & enter the repo
    git clone https://github.com/uzielroy/GPTCache-Streaming-Clustering.git
    cd GPTCache-Streaming-Clustering

### 2️⃣ Create the Conda environment
    conda env create -f env.yaml
    conda activate gptcache-env

### 3️⃣ Download models & datasets
    python download_models.py

*Tip:* To use a local HF cache or mirror, edit `download_models.py`.

---

## 🧪 Synthetic Benchmark – `benchmark_clustering_speedup.py`

| Argument             | Description                     | Default          |
|----------------------|---------------------------------|------------------|
| `--n_clusters_list`  | List of cluster counts          | `500 1000 2000`  |
| `--per_cluster_list` | Points per cluster              | `50 100`         |
| `--dim`              | Embedding dimension             | `128`            |
| `--tau`              | Cluster similarity threshold    | `0.75`           |
| `--alpha`            | EMA centroid update rate        | `0.1`            |
| `--n_queries`        | Query count (hits + misses)      | `3000`           |
| `--seed`             | RNG seed                        | `0`              |

Example:
---
    python benchmark_clustering_speedup.py \
      --n_clusters_list 500 1000 \
      --per_cluster_list 50 100 \
      --dim 128 --tau 0.75 --alpha 0.1 \
      --n_queries 3000 --seed 42

---

## 📈 Real‑Data Benchmark – `gptcache_benchmark.py`

| Argument            | Description                                    | Default                      |
|---------------------|------------------------------------------------|------------------------------|
| `--models`          | LLM back‑ends (`phi2`, `tinyllama`, `falcon`, …) | `phi2 tinyllama falcon`     |
| `--embedders`       | Embedding models (`bge-large`, `minilm`, …)     | `bge-large minilm`          |
| `--thrs`            | Cache similarity threshold                      | `0.68`                      |
| `--caches`          | Cache sizes                                     | `3000`                      |
| `--epochs`          | Dataset passes                                  | `1`                         |
| `--seed`            | RNG seed                                        | `42`                        |
| `--cluster_cache`   | Enable streaming clustering (flag)              | disabled                    |
| `--tau_inter`       | Inter‑cluster threshold                         | `0.5`                       |
| `--tau_intra`       | Intra‑cluster threshold                         | `0.63`                      |
| `--cluster_alpha`   | EMA centroid rate                               | `0.1`                       |

Clustered‑cache example:
---
    python gptcache_benchmark.py \
      --models phi2 tinyllama \
      --embedders bge-large \
      --thrs 0.68 --caches 1000 \
      --cluster_cache \
      --tau_inter 0.75 --tau_intra 0.65 \
      --cluster_alpha 0.1 \
      --epochs 1 --seed 42

---

## 📊 Output Layout
    results/
    └── <run_tag>/
        ├── requests.csv                 # Per‑request log
        ├── latency_throughput_stats.txt # Latency & throughput summary
        ├── latency_cdf.png              # CDF of request latencies
        ├── hit_rate.png                 # Running hit‑rate graph
        └── cluster_stats.txt            # Cluster diagnostics
    benchmark_summary.csv                # Aggregated summary across runs

---

### 🗂️ Output File Descriptions

| File | What it contains |
|------|------------------|
| **`requests.csv`** | One row per request. Columns: `latency_ms` (end‑to‑end latency), `hit` (1 = cache hit, 0 = miss), `gpu_mb` (GPU memory at request time), `ram_mb` (host RAM), plus sequential `idx`. |
| **`latency_throughput_stats.txt`** | Plain‑text summary of latency statistics (`avg_latency`, `median_latency`, `p95`, `p99`) and overall throughput (`throughput_qps`), along with totals for `hits`, `misses`, and final accuracy/score. |
| **`latency_cdf.png`** | Cumulative Distribution Function plot of per‑request latencies-quick visual insight into tail‑latency behaviour. |
| **`hit_rate.png`** | Line chart of cumulative cache hit‑rate over time (only produced when caching is enabled). |
| **`cluster_stats.txt`** | Final snapshot of clustering state: total clusters, non‑empty clusters, mean & max cluster size, plus a histogram of cluster sizes. A running log (`cluster_stats_progress.txt`) is also appended to during long runs. |
| **`benchmark_summary.csv`** | Aggregated run‑level metrics-each row corresponds to a single run tag and includes model, workload, cache settings, accuracy, latency percentiles, throughput, memory usage, and clustering stats. |


