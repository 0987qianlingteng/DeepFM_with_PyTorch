import json
import os
import re
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _build_index(values: pd.Series) -> Dict[str, int]:
    uniq = sorted(values.astype(str).unique().tolist())
    return {v: i for i, v in enumerate(uniq)}


def _extract_year(title: str) -> str:
    m = YEAR_RE.search(str(title))
    return m.group(1) if m else "UNK"


def _extract_primary_genre(genres: str) -> str:
    g = str(genres).split("|")
    return g[0] if g else "Unknown"


@dataclass
class ML1MPaths:
    root: str
    processed_dir: str
    train_npz: str
    val_npz: str
    meta_json: str
    movies_csv: str
    val_rows_csv: str


def get_ml1m_paths(data_dir: str) -> ML1MPaths:
    processed = os.path.join(data_dir, "processed")
    return ML1MPaths(
        root=data_dir,
        processed_dir=processed,
        train_npz=os.path.join(processed, "ml1m_train.npz"),
        val_npz=os.path.join(processed, "ml1m_val.npz"),
        meta_json=os.path.join(processed, "ml1m_meta.json"),
        movies_csv=os.path.join(processed, "ml1m_movies.csv"),
        val_rows_csv=os.path.join(processed, "ml1m_val_rows.csv"),
    )


def _load_raw_ml1m(ml1m_dir: str) -> pd.DataFrame:
    ratings = pd.read_csv(
        os.path.join(ml1m_dir, "ratings.dat"),
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    users = pd.read_csv(
        os.path.join(ml1m_dir, "users.dat"),
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
    )
    movies = pd.read_csv(
        os.path.join(ml1m_dir, "movies.dat"),
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    movies["year"] = movies["title"].map(_extract_year)
    movies["genre_primary"] = movies["genres"].map(_extract_primary_genre)
    df = ratings.merge(users, on="user_id", how="left").merge(
        movies[["movie_id", "title", "genres", "genre_primary", "year"]],
        on="movie_id",
        how="left",
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def _build_feature_maps(df: pd.DataFrame):
    user_map = _build_index(df["user_id"])
    movie_map = _build_index(df["movie_id"])
    gender_map = _build_index(df["gender"])
    age_map = _build_index(df["age"])
    occ_map = _build_index(df["occupation"])
    zip_map = _build_index(df["zip"])
    genre_map = _build_index(df["genre_primary"])
    year_map = _build_index(df["year"])

    df["user_idx"] = df["user_id"].astype(str).map(user_map).astype(np.int64)
    df["movie_idx"] = df["movie_id"].astype(str).map(movie_map).astype(np.int64)
    df["gender_idx"] = df["gender"].astype(str).map(gender_map).astype(np.int64)
    df["age_idx"] = df["age"].astype(str).map(age_map).astype(np.int64)
    df["occupation_idx"] = df["occupation"].astype(str).map(occ_map).astype(np.int64)
    df["zip_idx"] = df["zip"].astype(str).map(zip_map).astype(np.int64)
    df["genre_idx"] = df["genre_primary"].astype(str).map(genre_map).astype(np.int64)
    df["year_idx"] = df["year"].astype(str).map(year_map).astype(np.int64)

    maps = {
        "user_map": user_map,
        "movie_map": movie_map,
        "gender_map": gender_map,
        "age_map": age_map,
        "occ_map": occ_map,
        "zip_map": zip_map,
        "genre_map": genre_map,
        "year_map": year_map,
    }
    return df, maps


def _split_indices(n_rows: int, split_strategy: str, val_ratio: float, timestamps: np.ndarray, random_state: int):
    if split_strategy == "random":
        rng = np.random.default_rng(random_state)
        idx = np.arange(n_rows)
        rng.shuffle(idx)
        cut = int((1.0 - val_ratio) * n_rows)
        return idx[:cut], idx[cut:]
    if split_strategy == "time":
        order = np.argsort(timestamps)
        cut = int((1.0 - val_ratio) * n_rows)
        return order[:cut], order[cut:]
    raise ValueError(f"Unsupported split_strategy for current task: {split_strategy}")


def prepare_movielens_1m(
    ml1m_dir: str,
    output_dir: str,
    split_strategy: str = "random",
    val_ratio: float = 0.1,
    random_state: int = 42,
    positive_threshold: int = 4,
    task_type: str = "implicit",
    neg_ratio: int = 1,
    max_rows: int = 0,
) -> ML1MPaths:
    os.makedirs(output_dir, exist_ok=True)
    paths = get_ml1m_paths(output_dir)
    os.makedirs(paths.processed_dir, exist_ok=True)

    df = _load_raw_ml1m(ml1m_dir)
    df["label"] = (df["rating"] >= positive_threshold).astype(np.float32)
    df, maps = _build_feature_maps(df)

    feature_fields = [
        "user_idx",
        "gender_idx",
        "age_idx",
        "occupation_idx",
        "zip_idx",
        "movie_idx",
        "genre_idx",
        "year_idx",
    ]
    feature_sizes = [
        len(maps["user_map"]),
        len(maps["gender_map"]),
        len(maps["age_map"]),
        len(maps["occ_map"]),
        len(maps["zip_map"]),
        len(maps["movie_map"]),
        len(maps["genre_map"]),
        len(maps["year_map"]),
    ]

    movies_meta = (
        df[["movie_id", "movie_idx", "title", "genres", "genre_primary", "year", "genre_idx", "year_idx"]]
        .drop_duplicates(subset=["movie_idx"])
        .sort_values("movie_idx")
        .reset_index(drop=True)
    )
    movies_meta.to_csv(paths.movies_csv, index=False, encoding="utf-8")

    if task_type == "explicit":
        table = df[
            feature_fields + ["label", "timestamp", "user_id", "movie_id", "title", "genres"]
        ].copy()
    elif task_type == "implicit":
        pos = df[df["label"] > 0].copy()
        all_movie_idx = np.array(sorted(movies_meta["movie_idx"].unique().tolist()), dtype=np.int64)
        movie_feat = movies_meta.set_index("movie_idx")[["genre_idx", "year_idx", "movie_id", "title", "genres"]]

        rng = np.random.default_rng(random_state)
        neg_records = []
        for user_idx, grp in pos.groupby("user_idx"):
            seen = set(int(x) for x in grp["movie_idx"].tolist())
            candidates = np.array([m for m in all_movie_idx if m not in seen], dtype=np.int64)
            if len(candidates) == 0:
                continue
            need = len(grp) * max(1, int(neg_ratio))
            sampled = rng.choice(candidates, size=need, replace=len(candidates) < need)
            base = grp.iloc[0]
            ts = grp["timestamp"].to_numpy()
            ts_sample = rng.choice(ts, size=need, replace=True)
            for i, movie_idx in enumerate(sampled):
                info = movie_feat.loc[int(movie_idx)]
                neg_records.append(
                    {
                        "user_idx": int(user_idx),
                        "gender_idx": int(base["gender_idx"]),
                        "age_idx": int(base["age_idx"]),
                        "occupation_idx": int(base["occupation_idx"]),
                        "zip_idx": int(base["zip_idx"]),
                        "movie_idx": int(movie_idx),
                        "genre_idx": int(info["genre_idx"]),
                        "year_idx": int(info["year_idx"]),
                        "label": 0.0,
                        "timestamp": int(ts_sample[i]),
                        "user_id": base["user_id"],
                        "movie_id": int(info["movie_id"]),
                        "title": str(info["title"]),
                        "genres": str(info["genres"]),
                    }
                )

        pos_table = pos[feature_fields + ["label", "timestamp", "user_id", "movie_id", "title", "genres"]].copy()
        neg_table = pd.DataFrame(neg_records)
        table = pd.concat([pos_table, neg_table], ignore_index=True).sample(
            frac=1.0, random_state=random_state
        ).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    if max_rows and max_rows > 0 and len(table) > max_rows:
        table = table.sample(n=int(max_rows), random_state=random_state).reset_index(drop=True)

    if split_strategy == "user_leave_one_out":
        split_strategy = "random"
    train_idx, val_idx = _split_indices(
        n_rows=len(table),
        split_strategy=split_strategy,
        val_ratio=val_ratio,
        timestamps=table["timestamp"].to_numpy(),
        random_state=random_state,
    )

    Xi = table[feature_fields].to_numpy(dtype=np.int64)
    Xv = np.ones_like(Xi, dtype=np.float32)
    y = table["label"].to_numpy(dtype=np.float32)

    np.savez_compressed(paths.train_npz, Xi=Xi[train_idx], Xv=Xv[train_idx], y=y[train_idx])
    np.savez_compressed(paths.val_npz, Xi=Xi[val_idx], Xv=Xv[val_idx], y=y[val_idx])
    table.iloc[val_idx][["user_id", "movie_id", "title", "genres", "timestamp", "label"]].reset_index(drop=True).to_csv(
        paths.val_rows_csv, index=False, encoding="utf-8"
    )

    user_history = (
        table[table["label"] > 0]
        .groupby("user_idx")["movie_idx"]
        .apply(lambda s: sorted(set(int(x) for x in s.tolist())))
        .to_dict()
    )
    user_profiles = (
        table.groupby("user_idx")[["gender_idx", "age_idx", "occupation_idx", "zip_idx"]]
        .first()
        .astype(np.int64)
        .to_dict(orient="index")
    )

    meta = {
        "feature_fields": feature_fields,
        "feature_sizes": feature_sizes,
        "split_strategy": split_strategy,
        "val_ratio": val_ratio,
        "positive_threshold": positive_threshold,
        "task_type": task_type,
        "neg_ratio": neg_ratio,
        "max_rows": int(max_rows) if max_rows else 0,
        "mappings": maps,
        "user_history_pos_movies": {str(k): v for k, v in user_history.items()},
        "user_profiles": {
            str(k): {
                "gender_idx": int(v["gender_idx"]),
                "age_idx": int(v["age_idx"]),
                "occupation_idx": int(v["occupation_idx"]),
                "zip_idx": int(v["zip_idx"]),
            }
            for k, v in user_profiles.items()
        },
        "stats": {
            "num_rows": int(len(table)),
            "train_rows": int(len(train_idx)),
            "val_rows": int(len(val_idx)),
            "num_users": int(len(maps["user_map"])),
            "num_movies": int(len(maps["movie_map"])),
            "positive_rate": float(y.mean()),
        },
    }
    with open(paths.meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return paths


class NpzCTRDataset(Dataset):
    def __init__(self, npz_path: str):
        obj = np.load(npz_path)
        self.Xi = obj["Xi"].astype(np.int64)
        self.Xv = obj["Xv"].astype(np.float32)
        self.y = obj["y"].astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.Xi[idx]).long(),
            torch.from_numpy(self.Xv[idx]).float(),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def load_meta(meta_json: str) -> Dict:
    with open(meta_json, "r", encoding="utf-8") as f:
        return json.load(f)
