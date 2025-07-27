#!/usr/bin/env python3
"""
GPT-Cache Benchmark with Streaming Clustering Option, Dual Thresholds, and LRU Eviction

Usage Example:
python gptcache_benchmark.py --cluster_cache --tau_inter 0.7 --tau_intra 0.55 --cluster_alpha 0.1 --caches 1000
"""

import argparse, itertools, os, random, tempfile, time, re, sys
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import psutil
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from gptcache import cache, Config
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation import NumpyNormEvaluation
from collections import defaultdict, Counter, deque

def _unit(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32", copy=False)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

class ClusteredIndex:
    def __init__(self, dim, tau_inter, tau_intra, alpha, cache_size):
        import faiss
        self.dim = dim
        self.tau_inter = tau_inter
        self.tau_intra = tau_intra
        self.alpha = alpha
        self.cache_size = cache_size
        self.cent = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.cvecs: list[np.ndarray] = []
        self.clus: list = []
        self.ans_map = defaultdict(list)
        self.lru = deque()
        self.next_id = defaultdict(int)
        self.id2ansidx = defaultdict(dict)

    def insert(self, v: np.ndarray, answer=None):
        import faiss
        if self.cent.ntotal == 0:
            cid, sim = None, 0.0
        else:
            D, I = self.cent.search(v[None,:], 1)
            cid, sim = int(I[0,0]), float(D[0,0])
        if cid is None or sim < self.tau_inter:
            cid = len(self.cvecs)
            self.cent.add_with_ids(v[None,:], np.array([cid],dtype=np.int64))
            self.cvecs.append(v.copy())
            idx = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))  # << FIXED HERE
            self.clus.append(idx)
        else:
            old = self.cvecs[cid]
            newc = _unit((1-self.alpha)*old + self.alpha*v)
            self.cvecs[cid] = newc
            self.cent.remove_ids(np.array([cid],dtype=np.int64))
            self.cent.add_with_ids(newc[None,:], np.array([cid],dtype=np.int64))
        local_id = self.next_id[cid]
        self.next_id[cid] += 1
        self.clus[cid].add_with_ids(v[None,:], np.array([local_id],dtype=np.int64))
        self.lru.append((cid, local_id))
        ans_idx = len(self.ans_map[cid])
        if answer is not None:
            self.ans_map[cid].append(answer)
            self.id2ansidx[cid][local_id] = ans_idx
        else:
            self.ans_map[cid].append(None)
            self.id2ansidx[cid][local_id] = ans_idx
        # LRU eviction
        while len(self.lru) > self.cache_size:
            evict_cid, evict_lid = self.lru.popleft()
            try:
                self.clus[evict_cid].remove_ids(np.array([evict_lid], dtype=np.int64))
                ansidx = self.id2ansidx[evict_cid].pop(evict_lid, None)
                if ansidx is not None and ansidx < len(self.ans_map[evict_cid]):
                    self.ans_map[evict_cid][ansidx] = None  # Mark as removed
            except Exception:
                pass

    def search(self, v: np.ndarray, top_k: int):
        D, I = self.cent.search(v[None,:], 1)
        cid, sim = int(I[0,0]), float(D[0,0])
        if sim < self.tau_inter:
            return []
        D2, I2 = self.clus[cid].search(v[None,:], top_k)
        if float(D2[0,0]) < self.tau_intra:
            return []
        ans = "<HIT_FROM_CLUSTER>"
        if I2.size > 0:
            idx = int(I2[0,0])   # convert to Python int, not array!
            ansidx = self.id2ansidx[cid].get(idx, None)
            if ansidx is not None:
                a = self.ans_map[cid][ansidx]
                if a is not None:
                    ans = a
        return [(cid, D2[0,0], int(I2[0,0]), ans)]

    def get_cluster_stats(self):
        size_list = [c.ntotal for c in self.clus]
        n_clusters = len(self.clus)
        nonzero = sum(1 for x in size_list if x > 0)
        return n_clusters, nonzero, size_list

    def flush(self):
        # For compatibility with gptcache's cache.flush()
        # You can optionally implement actual clearing here if needed.
        self.__init__(self.dim, self.tau_inter, self.tau_intra, self.alpha, self.cache_size)
DEFAULT_SEED = 42
GEN_ARGS     = dict(max_new_tokens=8, do_sample=False)
DEBUG_N      = 5

EMB_MODELS = {
    "bge-large": "BAAI/bge-large-en-v1.5",
    "minilm":    "sentence-transformers/all-MiniLM-L6-v2",
}
MODEL_ID = {
    "phi2":      "phi-2",
    "tinyllama": "TinyLlama-1.1B-Chat-v1.0",
    "falcon":    "falcon-rw-1b",
}
AG_LABELS = ["world", "sports", "business", "sci/tech"]

@dataclass
class RunStats:
    model: str; workload: str; embedder: str; cache_enabled: bool
    cache_size: int; sim_thr: float; clustered: bool
    tau_inter: float; tau_intra: float
    hits: int; misses: int; accuracy: float
    avg_latency: float; median_latency: float; p95: float; p99: float; throughput_qps: float
    gpu_mem_mb: int; ram_mb: int
    n_clusters: int = 0; n_clusters_nonzero: int = 0
    cluster_size_mean: float = 0.0; cluster_size_max: int = 0


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def make_encoder(model_name: str, device: str):
    st_model = SentenceTransformer(model_name, device=device)
    def encode(texts, *, convert_to_tensor=False, **kw):
        return st_model.encode(
            texts, convert_to_tensor=convert_to_tensor,
            normalize_embeddings=True, **kw)
    encode.dim = st_model.get_sentence_embedding_dimension()
    return encode

def setup_cache(encode_fn, cache_size, sim_thr,
                use_clustered=False, tau_inter=0.7, tau_intra=0.55, cluster_alpha=0.1):
    if getattr(cache, "data_manager", None):
        cache.flush()
    if use_clustered:
        cache.data_manager = ClusteredIndex(encode_fn.dim, tau_inter, tau_intra, cluster_alpha, cache_size)
        cache.clustered = True
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        scalar = CacheBase("sqlite", sql_url="sqlite:///:memory:")
        vector = VectorBase(
            "faiss", dimension=encode_fn.dim, top_k=1,
            index_path=os.path.join(tmp_dir.name, "faiss.index"),
            metric_type="IP")
        manager = get_data_manager(scalar, vector,
                                   max_size=cache_size,
                                   clean_size=int(cache_size * 0.2))
        cache.init(
            embedding_func=encode_fn, data_manager=manager,
            similarity_evaluation=NumpyNormEvaluation(enable_normal=True),
            config=Config(similarity_threshold=sim_thr))
        cache._tmp_dir = tmp_dir
        cache.clustered = False

def clean_label(x): return x.strip().lower()
def task_metric(wl):
    if wl == "repetitive-short":
        return "cls"
    return "cos"

def init_state(enc):
    return dict(correct=0, seen=0, sum=0.0, enc=enc)

def update(state, pred, ref, wl):
    t = task_metric(wl)
    if t == "cls":
        state["correct"] += int(clean_label(pred) == clean_label(ref))
        state["seen"] += 1
    else:
        state["sum"] += util.cos_sim(
            state["enc"]([pred], convert_to_tensor=True),
            state["enc"]([ref],  convert_to_tensor=True)
        )[0, 0].item()
        state["seen"] += 1

def final(state, wl):
    t = task_metric(wl)
    if t == "cls":
        return state["correct"] / max(1, state["seen"])
    return state["sum"] / max(1, state["seen"])

def mname(wl):
    return {"cls": "acc", "cos": "cos"}[task_metric(wl)]

def load_workload_dataset(name):
    if name == "novel-long":
        ds = load_dataset("cnn_dailymail", "3.0.0", split="train[:5000]")
        return ds["article"], ds["highlights"]
    if name == "repetitive-short":
        ds = load_dataset("ag_news", split="train[:5000]")
        return ds["text"], [AG_LABELS[i] for i in ds["label"]]
    raise ValueError(name)

def benchmark(model_path, workload, embed_name, cache_on, thr, cache_size,
              epochs, seed: int, out_root: Path,
              use_clustered=False, tau_inter=0.7, tau_intra=0.55, cluster_alpha=0.1):

    set_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode = make_encoder(embed_name, device)
    if cache_on:
        setup_cache(encode, cache_size, thr, use_clustered, tau_inter, tau_intra, cluster_alpha)

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = (AutoModelForCausalLM
             .from_pretrained(model_path, torch_dtype="auto")
             .to(device).eval())

    tasks, refs = load_workload_dataset(workload)
    order = list(range(len(tasks))) * epochs
    random.shuffle(order)

    cache_type_str = "clustered" if (cache_on and use_clustered) else "faiss"
    run_tag = (
        f"{model_path.name}_{workload}_{Path(embed_name).name}_"
        f"thr{thr}_cs{cache_size}_cache{int(cache_on)}_{cache_type_str}_"
        f"tauinter{tau_inter}_tauintra{tau_intra}_alpha{cluster_alpha}"
    )
    run_dir = out_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    if cache_on:
        print(f"\n[INFO] Cache backend: {'CLUSTERED (streaming)' if use_clustered else 'FAISS (flat)'}\n")

    hits = misses = 0
    lat, log_rows = [], []
    elapsed = 0.0
    mstate = init_state(encode)

    bar = tqdm(
        order, total=len(order), desc=run_tag, dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
    )

    for idx in bar:
        article, ref = tasks[idx], refs[idx]
        prompt = article
        if workload == "repetitive-short":
            prompt = (
                "Classify the following news article into "
                "[World, Sports, Business, Sci/Tech]. Respond with one word.\n\n"
                f"Article:\n{article}\n\nCategory:"
            )

        qv = encode(article)
        t0 = time.time()
        hit_flag = False
        ans = None

        if cache_on:
            if getattr(cache, 'clustered', False):
                res = cache.data_manager.search(_unit(qv), top_k=1)
                if res:
                    ans = res[0][3]
                    hit_flag = True
            else:
                res = cache.data_manager.search(qv, top_k=1)
                if res:
                    cd = cache.data_manager.get_scalar_data(res[0])
                    if cd and cosine(qv, np.asarray(cd.embedding_data)) >= thr:
                        ans = cd.answers[0].answer
                        hit_flag = True

        if not hit_flag:
            toks = tok(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.inference_mode():
                out = model.generate(**toks, **GEN_ARGS,
                                     pad_token_id=tok.pad_token_id)
            gen_ids = out[0][toks["input_ids"].shape[1]:]
            ans = tok.decode(gen_ids, skip_special_tokens=True).strip()
            if workload == "repetitive-short":
                ans = re.split(r'\W+', ans, 1)[0]
            if cache_on:
                if getattr(cache, 'clustered', False):
                    cache.data_manager.insert(_unit(qv), answer=ans)
                else:
                    cache.data_manager.save(article, ans, qv)

        dt = time.time() - t0
        elapsed += dt
        lat.append(dt)
        hits += int(hit_flag)
        misses += int(not hit_flag)

        if len(log_rows) < DEBUG_N:
            print(f"\n--- sample {len(log_rows)} ---")
            print("PRED:", ans[:120])
            print("GT  :", str(ref)[:120])
            print("---------------------------")

        update(mstate, ans, ref, workload)
        live = final(mstate, workload)

        hr = hits / (hits + misses) if (hits + misses) > 0 else 0.0
        bar.set_postfix(hits=hits, misses=misses, hr=f"{hr:.2%}", **{mname(workload): f"{live:.3f}"})

        log_rows.append({
            "idx": len(log_rows),
            "latency_ms": dt * 1000,
            "hit": int(hit_flag),
            "gpu_mb": (torch.cuda.memory_allocated() // 2**20
                       if torch.cuda.is_available() else 0),
            "ram_mb": psutil.Process().memory_info().rss // 2**20,
        })

        if (cache_on and getattr(cache, 'clustered', False) and (len(log_rows) % 500 == 0 or len(log_rows) == 1)):
            n_clusters, n_clusters_nonzero, size_list = cache.data_manager.get_cluster_stats()
            cluster_size_max = max(size_list) if size_list else 0
            cluster_size_mean = float(np.mean(size_list)) if size_list else 0.0
            print(f"\n[{len(log_rows)} requests] Clustered Cache Stats:")
            print(f"  Total clusters:   {n_clusters}")
            print(f"  Nonzero clusters: {n_clusters_nonzero}")
            print(f"  Cluster mean sz:  {cluster_size_mean:.2f}")
            print(f"  Cluster max sz:   {cluster_size_max}")
            with open(run_dir / "cluster_stats_progress.txt", "a") as fout:
                print(f"{len(log_rows)}\t{n_clusters}\t{n_clusters_nonzero}\t{cluster_size_mean:.2f}\t{cluster_size_max}", file=fout)

    bar.close()

    # ---------- Statistics Calculation ----------
    lat_array = np.array(lat)
    lat_mean = float(np.mean(lat_array))
    lat_median = float(np.median(lat_array))
    lat_p95 = float(np.percentile(lat_array, 95))
    lat_p99 = float(np.percentile(lat_array, 99))
    throughput = (hits + misses) / (elapsed + 1e-9)  # requests per second (QPS)
    # --------------------------------------------

    acc_final = final(mstate, workload)
    pd.DataFrame(log_rows).to_csv(run_dir / "requests.csv", index=False)

    stats_dict = dict(
        avg_latency=lat_mean,
        median_latency=lat_median,
        p95=lat_p95,
        p99=lat_p99,
        throughput_qps=throughput,
        hits=hits,
        misses=misses,
        accuracy=acc_final,
    )
    with open(run_dir / "latency_throughput_stats.txt", "w") as fout:
        for k, v in stats_dict.items():
            print(f"{k}: {v}", file=fout)

    n_clusters = n_clusters_nonzero = cluster_size_max = 0
    cluster_size_mean = 0.0
    if cache_on and use_clustered:
        n_clusters, n_clusters_nonzero, size_list = cache.data_manager.get_cluster_stats()
        cluster_size_max = max(size_list) if size_list else 0
        cluster_size_mean = float(np.mean(size_list)) if size_list else 0.0
        print(f"\n[CLUSTERED CACHE STATS]")
        print(f"  Total clusters:         {n_clusters}")
        print(f"  Nonzero clusters:       {n_clusters_nonzero}")
        print(f"  Cluster size (mean):    {cluster_size_mean:.2f}")
        print(f"  Cluster size (max):     {cluster_size_max}")
        with open(run_dir / "cluster_stats.txt", "w") as fout:
            print(f"Total clusters:         {n_clusters}", file=fout)
            print(f"Nonzero clusters:       {n_clusters_nonzero}", file=fout)
            print(f"Cluster size (mean):    {cluster_size_mean:.2f}", file=fout)
            print(f"Cluster size (max):     {cluster_size_max}", file=fout)
            c = Counter(size_list)
            print("\nSize histogram (count : num_clusters):", file=fout)
            for sz, count in sorted(c.items()):
                print(f"{sz:5d} : {count:5d}", file=fout)

    return RunStats(
        model_path.name, workload, embedder=embed_name, cache_enabled=cache_on,
        cache_size=cache_size, sim_thr=thr, clustered=bool(cache_on and use_clustered),
        tau_inter=tau_inter, tau_intra=tau_intra,
        hits=hits, misses=misses, accuracy=acc_final,
        avg_latency=lat_mean, median_latency=lat_median,
        p95=lat_p95, p99=lat_p99, throughput_qps=throughput,
        gpu_mem_mb=(torch.cuda.max_memory_allocated()//2**20
                    if torch.cuda.is_available() else 0),
        ram_mb=psutil.Process().memory_info().rss//2**20,
        n_clusters=n_clusters,
        n_clusters_nonzero=n_clusters_nonzero,
        cluster_size_mean=cluster_size_mean,
        cluster_size_max=cluster_size_max,
    )

def quick_plots(out_root: Path, run_stats: RunStats):
    run_tag = (
        f"{run_stats.model}_{run_stats.workload}_{Path(run_stats.embedder).name}_"
        f"thr{run_stats.sim_thr}_cs{run_stats.cache_size}_cache{int(run_stats.cache_enabled)}"
        f"_{'clustered' if run_stats.clustered else 'faiss'}"
        f"_tauinter{run_stats.tau_inter}_tauintra{run_stats.tau_intra}_alpha{getattr(run_stats, 'cluster_alpha', 0.1)}"
    )
    run_dir = out_root / run_tag
    req_log = pd.read_csv(run_dir / "requests.csv")

    lat_sorted = np.sort(req_log["latency_ms"])
    p = np.linspace(0, 1, len(lat_sorted))
    plt.figure(); plt.plot(lat_sorted, p)
    plt.xlabel("Latency (ms)"); plt.ylabel("CDF"); plt.title("Latency CDF")
    plt.grid(True); plt.tight_layout()
    plt.savefig(run_dir / "latency_cdf.png"); plt.close()

    if run_stats.cache_enabled:
        hr = req_log["hit"].cumsum() / (np.arange(len(req_log))+1)
        plt.figure(); plt.plot(hr)
        plt.xlabel("Request #"); plt.ylabel("Cumulative hit-rate")
        plt.ylim(0,1); plt.grid(True); plt.tight_layout()
        plt.savefig(run_dir / "hit_rate.png"); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, default="models")
    ap.add_argument("--models",     nargs="+", default=list(MODEL_ID.keys()))
    ap.add_argument("--embedders",  nargs="+", default=list(EMB_MODELS.keys()))
    ap.add_argument("--thrs", "--thr", nargs="+", type=float, default=[0.68])
    ap.add_argument("--caches", "--cs",  nargs="+", type=int,   default=[3_000])
    ap.add_argument("--out",    type=Path,   default="results")
    ap.add_argument("--epochs", type=int,    default=1)
    ap.add_argument("--seed",   type=int,    default=DEFAULT_SEED)
    ap.add_argument("--cluster_cache", action="store_true",
                    help="Use streaming clustered cache instead of FAISS")
    ap.add_argument("--tau_inter", type=float, default=0.5,
                    help="Threshold for assigning to/creating clusters (default: 0.5)")
    ap.add_argument("--tau_intra", type=float, default=0.63,
                    help="Threshold for intra-cluster cache hit (default: 0.65)")
    ap.add_argument("--cluster_alpha", type=float, default=0.1,
                    help="EMA update rate for centroids (alpha, default: 0.1)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = list(itertools.product(
        args.models,
        ["novel-long","repetitive-short"],
        args.embedders,
        [True],
        args.thrs,
        args.caches,
    ))

    summary = []
    for m, wl, emb, cache_on, thr, cs in jobs:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        stats = benchmark(
            args.model_dir / MODEL_ID[m],
            wl, EMB_MODELS[emb],
            cache_on, thr, cs,
            epochs=args.epochs,
            seed=args.seed,
            out_root=args.out,
            use_clustered=args.cluster_cache,
            tau_inter=args.tau_inter,
            tau_intra=args.tau_intra,
            cluster_alpha=args.cluster_alpha,
        )

        summary.append(asdict(stats))
        quick_plots(args.out, stats)
        print(f"✓ plots saved → "
              f"{args.out/(stats.model+'_'+wl+'_'+Path(stats.embedder).name+'_thr'+str(thr)+'_cs'+str(cs)+'_cache1_' + ('clustered' if stats.clustered else 'faiss') + f'_tauinter{stats.tau_inter}_tauintra{stats.tau_intra}_alpha{args.cluster_alpha}')}\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(summary).to_csv(args.out / "benchmark_summary.csv", index=False)
    print(f"Summary CSV → {args.out/'benchmark_summary.csv'}")

if __name__ == "__main__":
    main()
