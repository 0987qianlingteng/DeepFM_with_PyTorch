import argparse
import json
import mimetypes
import os
import random
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import torch

from data.movielens import get_ml1m_paths, load_meta
from model.DeepFM import DeepFM


class MovieLensRecommender:
    def __init__(self, args):
        self.args = args
        self.paths = get_ml1m_paths(args.data_dir)
        self.meta = load_meta(self.paths.meta_json)
        self.val_rows = pd.read_csv(self.paths.val_rows_csv)
        self.movies_meta = pd.read_csv(self.paths.movies_csv)
        self.model = self._load_model()

    def _load_model(self):
        model = DeepFM(
            feature_sizes=self.meta["feature_sizes"],
            embedding_size=self.args.embedding_size,
            hidden_dims=[int(x) for x in self.args.hidden_dims.split(",")],
            dropout=[float(x) for x in self.args.dropout.split(",")],
            use_cuda=self.args.use_cuda,
        )
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.args.model_path}. "
                "Please train the model first or pass --model_path."
            )
        model.load_state_dict(torch.load(self.args.model_path, map_location=model.device))
        model.to(model.device)
        model.eval()
        return model

    def status(self):
        stats = self.meta.get("stats", {})
        return {
            "model_path": self.args.model_path,
            "num_samples": int(len(self.val_rows)),
            "num_movies": int(stats.get("num_movies", len(self.movies_meta))),
            "num_users": int(stats.get("num_users", 0)),
            "best_auc": self._best_auc(),
        }

    def _best_auc(self):
        if not self.args.history_path or not os.path.exists(self.args.history_path):
            return None
        hist = pd.read_csv(self.args.history_path)
        if "val_auc" not in hist.columns or hist.empty:
            return None
        return float(hist["val_auc"].max())

    def random_test_id(self):
        return random.randint(0, len(self.val_rows) - 1)

    def recommend(self, test_id, top_n):
        if test_id < 0 or test_id >= len(self.val_rows):
            raise ValueError(f"test_id must be between 0 and {len(self.val_rows) - 1}")

        row = self.val_rows.iloc[test_id]
        user_raw = str(row["user_id"])
        mappings = self.meta["mappings"]
        user_idx = mappings["user_map"][user_raw]
        profile = self.meta["user_profiles"][str(user_idx)]
        seen = set(self.meta["user_history_pos_movies"].get(str(user_idx), []))

        genre_map = mappings["genre_map"]
        year_map = mappings["year_map"]
        candidates = self.movies_meta[~self.movies_meta["movie_idx"].isin(seen)].copy()
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

        with torch.no_grad():
            xi_t = torch.from_numpy(xi).long().to(self.model.device)
            xv_t = torch.from_numpy(xv).float().to(self.model.device)
            scores = torch.sigmoid(self.model(xi_t, xv_t)).cpu().numpy()

        candidates["score"] = scores
        top = candidates.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        recs = [
            {
                "rank": int(i + 1),
                "title": str(r["title"]),
                "genres": str(r["genres"]),
                "score": float(r["score"]),
            }
            for i, r in top.iterrows()
        ]
        return {
            "test_id": int(test_id),
            "user_id": user_raw,
            "sample_movie": {
                "title": str(row["title"]),
                "genres": str(row["genres"]),
                "label": float(row.get("label", 0.0)),
            },
            "recommendations": recs,
        }


def make_handler(recommender, web_dir):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=web_dir, **kwargs)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                self._send_json(recommender.status())
                return
            if parsed.path == "/api/random":
                self._send_json({"test_id": recommender.random_test_id()})
                return
            if parsed.path == "/api/recommend":
                self._handle_recommend(parsed.query)
                return
            if parsed.path == "/":
                self.path = "/index.html"
            super().do_GET()

        def _handle_recommend(self, query):
            try:
                qs = parse_qs(query)
                test_id = int(qs.get("test_id", ["0"])[0])
                top_n = int(qs.get("top_n", ["5"])[0])
                top_n = max(1, min(top_n, 20))
                self._send_json(recommender.recommend(test_id, top_n))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _send_json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def guess_type(self, path):
            if path.endswith(".js"):
                return "application/javascript"
            if path.endswith(".css"):
                return "text/css"
            return mimetypes.guess_type(path)[0] or "application/octet-stream"

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive MovieLens recommendation UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--model_path", default="ml1m_dual_target_try5.pt")
    parser.add_argument("--history_path", default="ml1m_dual_target_try5.csv")
    parser.add_argument("--embedding_size", type=int, default=24)
    parser.add_argument("--hidden_dims", default="256,128")
    parser.add_argument("--dropout", default="0.2,0.2")
    parser.add_argument("--use_cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    web_dir = os.path.join(root, "web")
    recommender = MovieLensRecommender(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(recommender, web_dir))
    print(f"Recommendation UI: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
