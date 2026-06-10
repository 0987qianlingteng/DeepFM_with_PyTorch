import argparse
import csv
import os
import random
from itertools import product
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from data.dataset import CriteoDataset
from data.movielens import (
    NpzCTRDataset,
    get_ml1m_paths,
    load_meta,
    prepare_movielens_1m,
)
from model.DeepFM import DeepFM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_sizes(path: str) -> List[int]:
    return np.loadtxt(path, delimiter=",", dtype=int).tolist()


def save_history(history, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_auc", "epoch_time_sec"],
        )
        writer.writeheader()
        writer.writerows(history)


def build_model(args, feature_sizes):
    return DeepFM(
        feature_sizes=feature_sizes,
        embedding_size=args.embedding_size,
        hidden_dims=[int(x) for x in args.hidden_dims.split(",")],
        dropout=[float(x) for x in args.dropout.split(",")],
        use_cuda=args.use_cuda,
    )


def _train_loop(model, train_loader, val_loader, args, model_path, history_path):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)
    history = model.fit(
        train_loader,
        val_loader,
        optimizer,
        epochs=args.epochs,
        scheduler=scheduler,
        model_path=model_path,
        early_stop_patience=args.patience,
    )
    save_history(history, history_path)
    best = max(history, key=lambda x: (-1 if np.isnan(x["val_auc"]) else x["val_auc"]))
    avg_epoch_time = float(np.mean([h["epoch_time_sec"] for h in history]))
    print("\n=== Training Summary ===")
    print(f"best_epoch: {best['epoch']}")
    print(f"best_val_auc: {best['val_auc']:.6f}")
    print(f"best_val_acc: {best['val_acc']:.4f}")
    print(f"avg_epoch_time_sec: {avg_epoch_time:.3f}")
    print(f"history_saved: {history_path}")
    print(f"best_model_saved: {model_path}")
    return best, avg_epoch_time


def train_criteo(args):
    set_seed(args.seed)
    feature_sizes = load_feature_sizes(os.path.join(args.data_dir, args.feature_sizes))

    train_ds = CriteoDataset(args.data_dir, train=True, train_file=args.train_file, test_file=args.test_file)
    y = np.array(train_ds.target, dtype=np.int64)
    all_idx = np.arange(len(train_ds))
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y,
    )

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.use_cuda,
    )
    val_loader = DataLoader(
        Subset(train_ds, val_idx),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.use_cuda,
    )

    model = build_model(args, feature_sizes)
    _train_loop(model, train_loader, val_loader, args, args.model_path, args.history_path)


def recommend_criteo(args):
    set_seed(args.seed)
    feature_sizes = load_feature_sizes(os.path.join(args.data_dir, args.feature_sizes))
    model = build_model(args, feature_sizes)
    model.load_state_dict(torch.load(args.model_path, map_location=model.device))
    model.to(model.device)
    model.eval()

    test_ds = CriteoDataset(args.data_dir, train=False, train_file=args.train_file, test_file=args.test_file)
    total = len(test_ds)
    test_id = args.test_id
    if test_id is None:
        test_id = int(input(f"请输入测试ID (0 ~ {total-1}): ").strip())
    if test_id < 0 or test_id >= total:
        raise ValueError(f"test_id out of range: {test_id}, expected [0, {total-1}]")

    xi_all, xv_all = [], []
    for i in range(total):
        xi, xv = test_ds.get_feature_pair_by_id(i)
        xi_all.append(xi)
        xv_all.append(xv)
    xi_all = torch.stack(xi_all, dim=0).to(model.device)
    xv_all = torch.stack(xv_all, dim=0).to(model.device)

    probs, reps = [], []
    with torch.no_grad():
        for s in range(0, len(xi_all), args.eval_batch_size):
            xi_b = xi_all[s : s + args.eval_batch_size]
            xv_b = xv_all[s : s + args.eval_batch_size]
            probs.append(torch.sigmoid(model(xi_b, xv_b)))
            reps.append(model.get_representation(xi_b, xv_b))

    probs = torch.cat(probs, dim=0)
    reps = torch.cat(reps, dim=0)
    query_rep = reps[test_id : test_id + 1]
    sim = torch.nn.functional.cosine_similarity(query_rep, reps, dim=1)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    final_score = args.alpha * probs + (1.0 - args.alpha) * sim
    final_score[test_id] = -1.0
    topk = torch.topk(final_score, k=min(args.top_n, total - 1)).indices.cpu().numpy().tolist()

    print(f"\n输入测试ID: {test_id}")
    print(f"Top-{len(topk)} 推荐结果:")
    print("rank\tsample_id\tscore\tp(click)\tsimilarity\tmovie_token")
    for rank, idx in enumerate(topk, start=1):
        movie_token = int(test_ds.features[idx][13]) if test_ds.features.shape[1] > 13 else -1
        print(
            f"{rank}\t{idx}\t{float(final_score[idx]):.6f}\t"
            f"{float(probs[idx]):.6f}\t{float(sim[idx]):.6f}\t{movie_token}"
        )


def prepare_ml1m(args):
    set_seed(args.seed)
    ml1m_dir = args.ml1m_dir
    if not os.path.isdir(ml1m_dir):
        raise RuntimeError(f"MovieLens directory not found: {ml1m_dir}")
    paths = prepare_movielens_1m(
        ml1m_dir=ml1m_dir,
        output_dir=args.data_dir,
        split_strategy=args.split_strategy,
        val_ratio=args.val_ratio,
        random_state=args.seed,
        positive_threshold=args.positive_threshold,
        task_type=args.task_type,
        neg_ratio=args.neg_ratio,
        max_rows=args.max_rows,
    )
    meta = load_meta(paths.meta_json)
    print("MovieLens 1M prepared.")
    print(f"train file: {paths.train_npz}")
    print(f"val file: {paths.val_npz}")
    print(f"meta file: {paths.meta_json}")
    print(f"stats: {meta['stats']}")


def _load_ml1m_ready(args):
    paths = get_ml1m_paths(args.data_dir)
    if args.force_prepare or (not os.path.exists(paths.train_npz)) or (not os.path.exists(paths.meta_json)):
        prepare_ml1m(args)
    meta = load_meta(paths.meta_json)
    return paths, meta


def train_ml1m(args):
    set_seed(args.seed)
    paths, meta = _load_ml1m_ready(args)
    train_ds = NpzCTRDataset(paths.train_npz)
    val_ds = NpzCTRDataset(paths.val_npz)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.use_cuda,
    )
    model = build_model(args, meta["feature_sizes"])
    _train_loop(model, train_loader, val_loader, args, args.model_path, args.history_path)


def recommend_ml1m(args):
    set_seed(args.seed)
    paths, meta = _load_ml1m_ready(args)
    model = build_model(args, meta["feature_sizes"])
    model.load_state_dict(torch.load(args.model_path, map_location=model.device))
    model.to(model.device)
    model.eval()

    val_rows = pd.read_csv(paths.val_rows_csv)
    movies_meta = pd.read_csv(paths.movies_csv)
    user_map = meta["mappings"]["user_map"]
    genre_map = meta["mappings"]["genre_map"]
    year_map = meta["mappings"]["year_map"]
    user_profiles = meta["user_profiles"]
    user_history = meta["user_history_pos_movies"]

    test_id = args.test_id
    if test_id is None:
        test_id = int(input(f"请输入测试ID (0 ~ {len(val_rows)-1}): ").strip())
    if test_id < 0 or test_id >= len(val_rows):
        raise ValueError(f"test_id out of range: {test_id}, expected [0, {len(val_rows)-1}]")

    row = val_rows.iloc[test_id]
    user_raw = str(row["user_id"])
    user_idx = user_map[user_raw]
    profile = user_profiles[str(user_idx)]
    seen = set(user_history.get(str(user_idx), []))

    candidates = movies_meta[~movies_meta["movie_idx"].isin(seen)].copy()
    candidates["genre_idx"] = candidates["genre_primary"].astype(str).map(genre_map).fillna(0).astype(np.int64)
    candidates["year_idx"] = candidates["year"].astype(str).map(year_map).fillna(0).astype(np.int64)

    xi = np.stack(
        [
            np.full(len(candidates), user_idx, dtype=np.int64),
            np.full(len(candidates), profile["gender_idx"], dtype=np.int64),
            np.full(len(candidates), profile["age_idx"], dtype=np.int64),
            np.full(len(candidates), profile["occupation_idx"], dtype=np.int64),
            np.full(len(candidates), profile["zip_idx"], dtype=np.int64),
            candidates["movie_idx"].values.astype(np.int64),
            candidates["genre_idx"].values.astype(np.int64),
            candidates["year_idx"].values.astype(np.int64),
        ],
        axis=1,
    )
    xv = np.ones_like(xi, dtype=np.float32)

    xi_t = torch.from_numpy(xi).long().to(model.device)
    xv_t = torch.from_numpy(xv).float().to(model.device)
    with torch.no_grad():
        p = torch.sigmoid(model(xi_t, xv_t)).cpu().numpy()
    candidates["score"] = p
    top = candidates.sort_values("score", ascending=False).head(args.top_n).reset_index(drop=True)

    print(f"\n输入测试ID: {test_id}")
    print(f"对应用户: {user_raw} | 当前样本电影: {row['title']} ({row['genres']})")
    print(f"Top-{len(top)} 电影推荐:")
    print("rank\ttitle\tgenres\tscore")
    for i, r in top.iterrows():
        print(f"{i+1}\t{r['title']}\t{r['genres']}\t{float(r['score']):.6f}")


def grid_search_ml1m(args):
    set_seed(args.seed)
    embeddings = [int(x) for x in args.grid_embeddings.split(",")]
    hidden_choices = args.grid_hidden.split(";")
    lrs = [float(x) for x in args.grid_lrs.split(",")]
    split_choices = args.grid_splits.split(",")
    dropouts = [float(x) for x in args.dropout.split(",")]

    results = []
    trial = 0
    for split, emb, hid_str, lr in product(split_choices, embeddings, hidden_choices, lrs):
        trial += 1
        print(f"\n===== Grid Trial {trial} | split={split} emb={emb} hidden={hid_str} lr={lr} =====")
        prepare_movielens_1m(
            ml1m_dir=args.ml1m_dir,
            output_dir=args.data_dir,
            split_strategy=split,
            val_ratio=args.val_ratio,
            random_state=args.seed,
            positive_threshold=args.positive_threshold,
            task_type=args.task_type,
            neg_ratio=args.neg_ratio,
            max_rows=args.max_rows,
        )
        paths = get_ml1m_paths(args.data_dir)
        meta = load_meta(paths.meta_json)

        train_ds = NpzCTRDataset(paths.train_npz)
        val_ds = NpzCTRDataset(paths.val_npz)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

        local_args = argparse.Namespace(**vars(args))
        local_args.embedding_size = emb
        local_args.hidden_dims = hid_str
        local_args.lr = lr
        local_args.epochs = args.grid_epochs

        model = DeepFM(
            feature_sizes=meta["feature_sizes"],
            embedding_size=emb,
            hidden_dims=[int(x) for x in hid_str.split(",")],
            dropout=dropouts,
            use_cuda=args.use_cuda,
        )
        os.makedirs(args.grid_dir, exist_ok=True)
        model_path = os.path.join(args.grid_dir, f"ml1m_trial_{trial}.pt")
        history_path = os.path.join(args.grid_dir, f"ml1m_trial_{trial}.csv")
        best, avg_t = _train_loop(model, train_loader, val_loader, local_args, model_path, history_path)

        results.append(
            {
                "trial": trial,
                "split": split,
                "embedding_size": emb,
                "hidden_dims": hid_str,
                "lr": lr,
                "best_epoch": int(best["epoch"]),
                "best_val_auc": float(best["val_auc"]),
                "best_val_acc": float(best["val_acc"]),
                "avg_epoch_time_sec": float(avg_t),
                "model_path": model_path,
                "history_path": history_path,
            }
        )

    df = pd.DataFrame(results).sort_values("best_val_auc", ascending=False).reset_index(drop=True)
    out_csv = os.path.join(args.grid_dir, "ml1m_grid_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("\n===== Grid Search Done =====")
    print(df.head(10).to_string(index=False))
    print(f"saved: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFM for Criteo + MovieLens 1M")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_dir", type=str, default="./data")
    common.add_argument("--model_path", type=str, default="best_deepfm.pt")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--use_cuda", action="store_true")
    common.add_argument("--embedding_size", type=int, default=16)
    common.add_argument("--hidden_dims", type=str, default="256,128")
    common.add_argument("--dropout", type=str, default="0.2,0.2")
    common.add_argument("--eval_batch_size", type=int, default=4096)
    common.add_argument("--batch_size", type=int, default=4096)
    common.add_argument("--epochs", type=int, default=10)
    common.add_argument("--lr", type=float, default=1e-3)
    common.add_argument("--weight_decay", type=float, default=1e-6)
    common.add_argument("--patience", type=int, default=3)
    common.add_argument("--num_workers", type=int, default=0)
    common.add_argument("--history_path", type=str, default="history_auc.csv")
    common.add_argument("--val_ratio", type=float, default=0.1)

    p_train = subparsers.add_parser("train", parents=[common], help="Train on Criteo")
    p_train.add_argument("--train_file", type=str, default="train.txt")
    p_train.add_argument("--test_file", type=str, default="test.txt")
    p_train.add_argument("--feature_sizes", type=str, default="feature_sizes.txt")

    p_rec = subparsers.add_parser("recommend", parents=[common], help="Recommend on Criteo test ids")
    p_rec.add_argument("--train_file", type=str, default="train.txt")
    p_rec.add_argument("--test_file", type=str, default="test.txt")
    p_rec.add_argument("--feature_sizes", type=str, default="feature_sizes.txt")
    p_rec.add_argument("--test_id", type=int, default=None)
    p_rec.add_argument("--top_n", type=int, default=5)
    p_rec.add_argument("--alpha", type=float, default=0.7)

    p_prep_ml = subparsers.add_parser("ml_prepare", parents=[common], help="Prepare MovieLens 1M")
    p_prep_ml.add_argument("--ml1m_dir", type=str, default="./data/ml-1m/ml-1m")
    p_prep_ml.add_argument("--split_strategy", type=str, default="random", choices=["random", "time", "user_leave_one_out"])
    p_prep_ml.add_argument("--positive_threshold", type=int, default=4)
    p_prep_ml.add_argument("--task_type", type=str, default="implicit", choices=["implicit", "explicit"])
    p_prep_ml.add_argument("--neg_ratio", type=int, default=1)
    p_prep_ml.add_argument("--max_rows", type=int, default=0)

    p_train_ml = subparsers.add_parser("ml_train", parents=[common], help="Train on MovieLens 1M")
    p_train_ml.add_argument("--ml1m_dir", type=str, default="./data/ml-1m/ml-1m")
    p_train_ml.add_argument("--split_strategy", type=str, default="random", choices=["random", "time", "user_leave_one_out"])
    p_train_ml.add_argument("--positive_threshold", type=int, default=4)
    p_train_ml.add_argument("--task_type", type=str, default="implicit", choices=["implicit", "explicit"])
    p_train_ml.add_argument("--neg_ratio", type=int, default=1)
    p_train_ml.add_argument("--max_rows", type=int, default=0)
    p_train_ml.add_argument("--force_prepare", action="store_true")

    p_rec_ml = subparsers.add_parser("ml_recommend", parents=[common], help="Top-N Movie recommendation with real titles")
    p_rec_ml.add_argument("--ml1m_dir", type=str, default="./data/ml-1m/ml-1m")
    p_rec_ml.add_argument("--split_strategy", type=str, default="random", choices=["random", "time", "user_leave_one_out"])
    p_rec_ml.add_argument("--positive_threshold", type=int, default=4)
    p_rec_ml.add_argument("--task_type", type=str, default="implicit", choices=["implicit", "explicit"])
    p_rec_ml.add_argument("--neg_ratio", type=int, default=1)
    p_rec_ml.add_argument("--max_rows", type=int, default=0)
    p_rec_ml.add_argument("--force_prepare", action="store_true")
    p_rec_ml.add_argument("--test_id", type=int, default=None)
    p_rec_ml.add_argument("--top_n", type=int, default=5)

    p_grid = subparsers.add_parser("ml_grid", parents=[common], help="Grid search on MovieLens 1M")
    p_grid.add_argument("--ml1m_dir", type=str, default="./data/ml-1m/ml-1m")
    p_grid.add_argument("--positive_threshold", type=int, default=4)
    p_grid.add_argument("--task_type", type=str, default="implicit", choices=["implicit", "explicit"])
    p_grid.add_argument("--neg_ratio", type=int, default=1)
    p_grid.add_argument("--max_rows", type=int, default=0)
    p_grid.add_argument("--grid_splits", type=str, default="random,time,user_leave_one_out")
    p_grid.add_argument("--grid_embeddings", type=str, default="8,16")
    p_grid.add_argument("--grid_hidden", type=str, default="128,64;256,128")
    p_grid.add_argument("--grid_lrs", type=str, default="0.001,0.0005")
    p_grid.add_argument("--grid_epochs", type=int, default=5)
    p_grid.add_argument("--grid_dir", type=str, default="./grid_runs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "train":
        train_criteo(args)
    elif args.cmd == "recommend":
        recommend_criteo(args)
    elif args.cmd == "ml_prepare":
        prepare_ml1m(args)
    elif args.cmd == "ml_train":
        train_ml1m(args)
    elif args.cmd == "ml_recommend":
        recommend_ml1m(args)
    elif args.cmd == "ml_grid":
        grid_search_ml1m(args)

