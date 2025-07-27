#!/usr/bin/env python3
"""
benchmark_clustering_speedup.py
--------------------------------
Extended two-stage clustering benchmarking script with data analysis, explicit breakdowns, and ablation visualizations.

• Sweeps:
    - Number of clusters: --n_clusters_list
    - Points per cluster: --per_cluster_list
    - Similarity thresholds: --tau_list
    - EMA rates: --alpha_list
• No eviction: all points always stay in clusters
• Metrics: hit-rate, mean/median/percentile latencies, inter-/intra-cluster breakdown, speedup vs flat FAISS
• Visualizations:
    - CDF of latencies
    - Hit-rate curves
    - Inter/Intra latency breakdown bar charts
    - Ablation line plots showing τ/α impact on latency & hit-rate

Usage:
    pip install numpy faiss-cpu tqdm matplotlib pandas
    python benchmark_clustering_speedup.py \
        --n_clusters_list 500 1000 2000 \
        --per_cluster_list 50 100 \
        --tau_list 0.5 0.75 0.9 \
        --alpha_list 0.01 0.1 0.5 \
        --dim 128 --n_queries 3000 --seed 0
"""
import os
import time
import argparse
from typing import List, Dict

import numpy as np
import faiss
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage clustering speedup (no eviction) with analysis and ablation"
    )
    parser.add_argument("--n_clusters_list", type=int, nargs="+",
                        default=[500, 1000, 2000],
                        help="Synthetic ground-truth cluster counts")
    parser.add_argument("--per_cluster_list", type=int, nargs="+",
                        default=[50, 100],
                        help="Points per cluster to test")
    parser.add_argument("--dim", type=int, default=128,
                        help="Vector dimensionality")
    parser.add_argument("--n_queries", type=int, default=3000,
                        help="Number of hit/miss queries each")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--tau_list", type=float, nargs="+",
                        default=[0.75],
                        help="Similarity thresholds to test")
    parser.add_argument("--alpha_list", type=float, nargs="+",
                        default=[0.1],
                        help="EMA rates to test")
    return parser.parse_args()


def _unit(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32", copy=False)
    # Normalize along the last dimension; works for both 1D and 2D inputs
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


# Helpers: convert ns to µs & percentiles
mean_us = lambda x: np.mean(x) / 1e3
pct_us = lambda x, p: np.percentile(x, p) / 1e3


def bench_flat_times(vecs: np.ndarray, HIT: np.ndarray, MISS: np.ndarray) -> Dict[str, np.ndarray]:
    idx = faiss.IndexFlatIP(vecs.shape[1]); idx.add(vecs)
    hit_times, miss_times = [], []
    for v in tqdm(HIT, desc="flat hit", leave=False):
        t0 = time.perf_counter_ns(); idx.search(v[None, :], 1)
        hit_times.append(time.perf_counter_ns() - t0)
    for v in tqdm(MISS, desc="flat miss", leave=False):
        t0 = time.perf_counter_ns(); idx.search(v[None, :], 1)
        miss_times.append(time.perf_counter_ns() - t0)
    return {
        'hit_times': np.array(hit_times),
        'miss_times': np.array(miss_times),
        'all_times': np.array(hit_times + miss_times)
    }


class ClusteredIndex:
    def __init__(self, dim: int, tau: float, alpha: float):
        self.dim, self.tau, self.alpha = dim, tau, alpha
        self.cent = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.cvecs: List[np.ndarray] = []
        self.clus: List[faiss.IndexFlatIP] = []

    def insert(self, v: np.ndarray):
        if self.cent.ntotal == 0:
            cid, sim = None, 0.0
        else:
            D, I = self.cent.search(v[None, :], 1)
            cid, sim = int(I[0, 0]), float(D[0, 0])
        if cid is None or sim < self.tau:
            cid = len(self.cvecs)
            self.cent.add_with_ids(v[None, :], np.array([cid], dtype=np.int64))
            self.cvecs.append(v.copy())
            self.clus.append(faiss.IndexFlatIP(self.dim))
        else:
            old = self.cvecs[cid]
            newc = _unit((1 - self.alpha) * old + self.alpha * v)
            self.cvecs[cid] = newc
            self.cent.remove_ids(np.array([cid], dtype=np.int64))
            self.cent.add_with_ids(newc[None, :], np.array([cid], dtype=np.int64))
        self.clus[cid].add(v[None, :])

    def lookup_times(self, HIT: np.ndarray, MISS: np.ndarray) -> Dict[str, np.ndarray]:
        inter, intra, hits = [], [], []
        # HITs
        for v in tqdm(HIT, desc="cluster hit", leave=False):
            t0 = time.perf_counter_ns(); D, I = self.cent.search(v[None, :], 1)
            it = time.perf_counter_ns() - t0
            if float(D[0, 0]) < self.tau:
                jt, ok = 0, False
            else:
                t1 = time.perf_counter_ns(); D2, _ = self.clus[int(I[0, 0])].search(v[None, :], 1)
                jt = time.perf_counter_ns() - t1
                ok = float(D2[0, 0]) >= self.tau
            inter.append(it); intra.append(jt); hits.append(ok)
        # MISSes
        for v in tqdm(MISS, desc="cluster miss", leave=False):
            t0 = time.perf_counter_ns(); D, I = self.cent.search(v[None, :], 1)
            it = time.perf_counter_ns() - t0
            if float(D[0, 0]) < self.tau:
                jt = 0
            else:
                t1 = time.perf_counter_ns(); _ = self.clus[int(I[0, 0])].search(v[None, :], 1)
                jt = time.perf_counter_ns() - t1
            inter.append(it); intra.append(jt)
        return {
            'inter_times': np.array(inter),
            'intra_times': np.array(intra),
            'all_times': np.array(inter) + np.array(intra),
            'hit_flags': np.array(hits)
        }


def bench_clustered_times(vecs, HIT, MISS, tau, alpha):
    print(f"\n[Clustering] Building index (τ={tau}, α={alpha})…")
    idx = ClusteredIndex(vecs.shape[1], tau, alpha)
    for v in tqdm(_unit(vecs), desc="cluster insert", leave=False): idx.insert(v)
    return idx.lookup_times(_unit(HIT), _unit(MISS))


def plot_latency_distributions(flat_all, cl_all, pc, nc, tau, alpha):
    flat_x, flat_y = np.sort(flat_all/1e3), np.arange(len(flat_all))/len(flat_all)
    cl_x, cl_y     = np.sort(cl_all/1e3),    np.arange(len(cl_all))/len(cl_all)
    plt.figure();
    plt.step(flat_x, flat_y, where='post', label='Flat');
    plt.step(cl_x,   cl_y,   where='post', label='Clustered');
    plt.xlabel('Latency (µs)'); plt.ylabel('Empirical CDF');
    plt.title(f'Latency CDF pc={pc},nc={nc},τ={tau},α={alpha}');
    plt.legend(); plt.grid(True);
    plt.savefig(f"plots/cdf_pc{pc}_nc{nc}_tau{tau}_alpha{alpha}.png"); plt.close()


def plot_hit_rate_curve(hit_flags, pc, nc, tau, alpha):
    cr = np.cumsum(hit_flags) / np.arange(1, len(hit_flags)+1)
    plt.figure(); plt.plot(cr); plt.xlabel('Query Index');
    plt.ylabel('Cumulative Hit Rate'); plt.ylim(0,1);
    plt.title(f'Hit-Rate Curve pc={pc},nc={nc},τ={tau},α={alpha}');
    plt.grid(True);
    plt.savefig(f"plots/hitrate_pc{pc}_nc{nc}_tau{tau}_alpha{alpha}.png"); plt.close()


def plot_inter_intra_breakdown(inter_times, intra_times, pc, nc, tau, alpha):
    m_inter, m_intra = mean_us(inter_times), mean_us(intra_times)
    plt.figure(); plt.bar(['Inter','Intra'], [m_inter, m_intra]);
    plt.ylabel('Mean Latency (µs)');
    plt.title(f'Inter vs Intra Latency pc={pc},nc={nc},τ={tau},α={alpha}');
    plt.grid(axis='y');
    plt.savefig(f"plots/breakdown_pc{pc}_nc{nc}_tau{tau}_alpha{alpha}.png"); plt.close()


def plot_ablation(df: pd.DataFrame, metric: str, by: str, savepath: str):
    plt.figure()
    for val in sorted(df[by].unique()):
        sub = df[df[by]==val]
        plt.plot(sub['tau'] if by=='alpha' else sub['alpha'], sub[metric], marker='o', label=f"{by}={val}")
    plt.xlabel('τ' if by=='alpha' else 'α'); plt.ylabel(metric);
    plt.title(f'{metric} vs {"τ" if by=="alpha" else "α"} for varying {by}');
    plt.legend(); plt.grid(True);
    plt.savefig(savepath); plt.close()


def main():
    args = parse_args()
    os.makedirs('plots', exist_ok=True)
    results = []
    for pc in args.per_cluster_list:
      for nc in args.n_clusters_list:
        for tau in args.tau_list:
          for alpha in args.alpha_list:
            print(f"\n=== pc={pc}, nc={nc}, τ={tau}, α={alpha} ===")
            vecs, HIT, MISS = generate_synthetic(args.dim, nc, pc, args.seed, args.n_queries)
            flat = bench_flat_times(vecs, HIT, MISS)
            cl   = bench_clustered_times(vecs, HIT, MISS, tau, alpha)

            res = {
                'per_cluster': pc, 'n_clusters': nc, 'tau': tau, 'alpha': alpha,
                'flat_mean_us': mean_us(flat['all_times']),
                'flat_median_us': pct_us(flat['all_times'],50), 'flat_p95_us': pct_us(flat['all_times'],95), 'flat_p99_us': pct_us(flat['all_times'],99),
                'cluster_mean_us': mean_us(cl['all_times']), 'cluster_median_us': pct_us(cl['all_times'],50), 'cluster_p95_us': pct_us(cl['all_times'],95), 'cluster_p99_us': pct_us(cl['all_times'],99),
                'hit_rate_pct': np.mean(cl['hit_flags'])*100,
                'inter_mean_us': mean_us(cl['inter_times']), 'intra_mean_us': mean_us(cl['intra_times']),
                'speedup_mean': mean_us(flat['all_times'])/mean_us(cl['all_times']),
                'speedup_median': pct_us(flat['all_times'],50)/pct_us(cl['all_times'],50)
            }
            results.append(res)
            # Plots
            plot_latency_distributions(flat['all_times'], cl['all_times'], pc, nc, tau, alpha)
            plot_hit_rate_curve(cl['hit_flags'], pc, nc, tau, alpha)
            plot_inter_intra_breakdown(cl['inter_times'], cl['intra_times'], pc, nc, tau, alpha)
    # Save and display summary
    df = pd.DataFrame(results)
    df.to_csv('results_summary.csv', index=False)
    print("\nAblation results saved to results_summary.csv")
    print(df)
    # Ablation visualizations
    plot_ablation(df, 'cluster_mean_us', by='alpha', savepath='plots/ablation_latency_vs_tau.png')
    plot_ablation(df, 'hit_rate_pct',    by='alpha', savepath='plots/ablation_hitrate_vs_tau.png')
    plot_ablation(df, 'cluster_mean_us', by='tau',   savepath='plots/ablation_latency_vs_alpha.png')
    plot_ablation(df, 'hit_rate_pct',    by='tau',   savepath='plots/ablation_hitrate_vs_alpha.png')

if __name__ == '__main__':
    main()
