#!/usr/bin/env python3
"""
benchmark_clustering_speedup.py
--------------------------------
Sweep two-stage clustering (no eviction) across multiple synthetic dataset configurations
to measure speedup vs. flat FAISS with separate hit/miss timings and per-budget FAISS baselines.

• Sweeps:
    - Number of clusters: --n_clusters_list
    - Points per cluster: --per_cluster_list
• No eviction: all points always stay in clusters
• Metrics: hit-rate, mean hit-latency, mean miss-latency, mean overall latency, speed-up vs flat FAISS
• Includes FAISS (flat) timings for each budget
• Progress bars via tqdm and detailed printouts

Usage:
    pip install numpy faiss-cpu tqdm
    python benchmark_clustering_speedup.py \
        --n_clusters_list 500 1000 2000 \
        --per_cluster_list 50 100 \
        --dim 128 --tau 0.75 --alpha 0.1 --n_queries 3000 --seed 0
"""
import time
import numpy as np
import faiss
import argparse
from tqdm.auto import tqdm
from typing import List, Dict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage clustering speedup (no eviction) sweep"
    )
    parser.add_argument("--n_clusters_list", type=int, nargs="+",
                        default=[500, 1000, 2000],
                        help="Synthetic ground-truth cluster counts")
    parser.add_argument("--per_cluster_list", type=int, nargs="+",
                        default=[50, 100],
                        help="Points per cluster to test")
    parser.add_argument("--dim", type=int, default=128,
                        help="Vector dimensionality")
    parser.add_argument("--tau", type=float, default=0.75,
                        help="Similarity threshold for cluster creation")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="EMA rate for centroid updates")
    parser.add_argument("--n_queries", type=int, default=3000,
                        help="Number of hit/miss queries each")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    return parser.parse_args()

def _unit(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32", copy=False)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def generate_synthetic(dim, n_clusters, per_cluster, seed, n_queries):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype("float32")
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    vecs = []
    for c in centers:
        noise = rng.normal(scale=0.02, size=(per_cluster, dim)).astype("float32")
        pts = c + noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        vecs.append(pts)
    all_vecs = np.vstack(vecs)
    hit_idxs = rng.choice(all_vecs.shape[0], size=n_queries, replace=False)
    HIT = all_vecs[hit_idxs]
    MISS = rng.normal(size=(n_queries, dim)).astype("float32")
    MISS /= np.linalg.norm(MISS, axis=1, keepdims=True)
    return all_vecs, HIT, MISS

mean_us = lambda lat: sum(lat) / len(lat) / 1e3

def bench_flat(vecs, HIT, MISS):
    """
    Baseline flat FAISS over the full dataset, returning separate hit/miss/all latency.
    """
    idx = faiss.IndexFlatIP(vecs.shape[1]); idx.add(vecs)
    hit_times, miss_times = [], []
    for v in tqdm(HIT, desc="flat hit", leave=False):
        t0 = time.perf_counter_ns(); idx.search(v[None,:],1)
        hit_times.append(time.perf_counter_ns() - t0)
    for v in tqdm(MISS, desc="flat miss", leave=False):
        t0 = time.perf_counter_ns(); idx.search(v[None,:],1)
        miss_times.append(time.perf_counter_ns() - t0)
    return {
        'flat_hit_us': mean_us(hit_times),
        'flat_miss_us': mean_us(miss_times),
        'flat_all_us': mean_us(hit_times + miss_times)
    }

class ClusteredIndex:
    def __init__(self, dim, tau, alpha):
        self.dim, self.tau, self.alpha = dim, tau, alpha
        self.cent = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.cvecs: List[np.ndarray] = []
        self.clus: List[faiss.IndexFlatIP] = []

    def insert(self, v: np.ndarray):
        if self.cent.ntotal == 0:
            cid, sim = None, 0.0
        else:
            D, I = self.cent.search(v[None,:],1)
            cid, sim = int(I[0,0]), float(D[0,0])
        if cid is None or sim < self.tau:
            cid = len(self.cvecs)
            self.cent.add_with_ids(v[None,:], np.array([cid],dtype=np.int64))
            self.cvecs.append(v.copy())
            idx = faiss.IndexFlatIP(self.dim)
            self.clus.append(idx)
        else:
            old = self.cvecs[cid]
            newc = _unit((1-self.alpha)*old + self.alpha*v)
            self.cvecs[cid] = newc
            self.cent.remove_ids(np.array([cid],dtype=np.int64))
            self.cent.add_with_ids(newc[None,:], np.array([cid],dtype=np.int64))
        self.clus[cid].add(v[None,:])

    def lookup_times(self, HIT, MISS) -> Dict[str, float]:
        """
        Return separate average hit_us, miss_us, all_us for clustering.
        """
        hit_times, miss_times = [], []
        hits = 0
        for v in tqdm(HIT, desc="cluster hit", leave=False):
            t0 = time.perf_counter_ns(); ok = self._lookup(v)
            hit_times.append(time.perf_counter_ns() - t0)
            hits += int(ok)
        for v in tqdm(MISS, desc="cluster miss", leave=False):
            t0 = time.perf_counter_ns(); _ = self._lookup(v)
            miss_times.append(time.perf_counter_ns() - t0)
        return {
            'cluster_hit_us': mean_us(hit_times),
            'cluster_miss_us': mean_us(miss_times),
            'cluster_all_us': mean_us(hit_times + miss_times),
            'hit_rate': 100 * hits / len(HIT)
        }

    def _lookup(self, v: np.ndarray) -> bool:
        D, I = self.cent.search(v[None,:],1)
        cid, sim = int(I[0,0]), float(D[0,0])
        if sim < self.tau:
            return False
        D2, _ = self.clus[cid].search(v[None,:],1)
        return float(D2[0,0]) >= self.tau

def bench_clustered(vecs, HIT, MISS, tau, alpha):
    print("\n[Clustering] Building two-stage index (no eviction)…")
    idx = ClusteredIndex(vecs.shape[1], tau, alpha)
    for v in tqdm(vecs, desc="cluster insert", leave=False):
        idx.insert(_unit(v))
    return idx.lookup_times(_unit(HIT), _unit(MISS))  # returns dict with times

def main():
    args = parse_args()
    summary = []
    for pc in args.per_cluster_list:
        for nc in args.n_clusters_list:
            print(f"\n=== per_cluster={pc}, n_clusters={nc} ===")
            vecs, HIT, MISS = generate_synthetic(
                args.dim, nc, pc, args.seed, args.n_queries
            )
            flat = bench_flat(vecs, HIT, MISS)
            clustered = bench_clustered(vecs, HIT, MISS, args.tau, args.alpha)
            speedup = flat['flat_all_us'] / clustered['cluster_all_us']
            row = {
                'per_cluster': pc, 'n_clusters': nc, 'speedup': speedup
            }
            row.update(flat)
            row.update(clustered)
            summary.append(row)
    # Print summary table
    headers = [
        "per_cluster","n_clusters",
        "flat_hit_us","flat_miss_us","flat_all_us",
        "cluster_hit_us","cluster_miss_us","cluster_all_us",
        "hit_rate","speedup"
    ]
    print("\nSUMMARY TABLE:")
    print("  ".join(f"{h:>15}" for h in headers))
    for r in summary:
        print("  ".join(
            f"{r[h]:15.2f}" if isinstance(r[h], float) else f"{r[h]:>15}"
            for h in headers
        ))

if __name__ == "__main__":
    main()
