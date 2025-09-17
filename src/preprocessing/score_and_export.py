
# -*- coding: utf-8 -*-
"""
src/preprocessing/score_and_export.py

역할:
- config.USE_TORCH 플래그 또는 실행 모드에 따라 선택적으로 점수화
  1) USE_TORCH=True → Torch 모델(.pt), 실패 시 Sklearn(joblib) → 휴리스틱
  2) USE_TORCH=False → Sklearn(joblib), 실패 시 휴리스틱
- 입력: embedding_text 포함 DF/CSV
- 출력: active/culture/nature/relaxation 컬럼 추가 + 저장 옵션

사용 예:
python -m src.preprocessing.score_and_export \
  --input ./data/raw/tour_raw_with_texts.csv \
  --output ./data/processed/tour_scored.csv
"""

import os, warnings, argparse
from typing import Union, Optional, List, Dict
import numpy as np
import pandas as pd
from src.config import RAW_WITH_TEXTS_CSV, SCORED_CSV, MODELS_DIR, USE_TORCH
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

def _softmax_rows(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - X.max(axis=1, keepdims=True)
    ex = np.exp(X)
    return ex / (ex.sum(axis=1, keepdims=True) + eps)

# 휴리스틱 키워드(간단)
KEYWORDS: Dict[str, List[str]] = {
    "active":  ["등산","트레킹","하이킹","캠핑","서핑","마라톤","자전거","패러글라이딩","래프팅","체험","레포츠","스포츠","클라이밍","암벽","산행","러닝","승마","카약","카누","서바이벌"],
    "culture": ["박물관","미술관","전시","전통","역사","궁궐","고궁","사찰","유적","기념관","공연","연극","뮤지컬","문화재","예술","아트","문화센터","도서관","아카이브","축제","페스티벌"],
    "nature":  ["산","공원","바다","해변","호수","강","계곡","숲","정원","습지","폭포","해안","둘레길","자연","하천","산책로","초원","국립공원","자연공원","생태공원","갯벌"],
    "relax":   ["스파","온천","리조트","휴양림","힐링","명상","요가","찜질방","호텔","카페","정원 산책","휴식","야경","전망대","호캉스","글램핑","풀빌라","리트릿","쉼터","휴양"],
}
def _heuristic_scores(texts: List[str]) -> np.ndarray:
    scores = []
    for t in texts:
        t0 = (t or "").lower()
        def any_in(keys): return any(k.lower() in t0 for k in keys)
        v = np.array([1.0 if any_in(KEYWORDS[k]) else 0.0 for k in ["active","culture","nature","relax"]], dtype=np.float64)
        if v.sum() == 0: v[:] = 0.25
        scores.append(v)
    return _softmax_rows(np.vstack(scores))

def _find_torch_model() -> Optional[str]:
    names = ["theme_regressor.pt", "theme_regressor_best.pt"]
    cands = [Path(MODELS_DIR), Path.cwd(), Path.cwd()/ "models"]
    for d in cands:
        for n in names:
            p = d / n
            if p.exists(): return str(p)
    return None

class TorchPredictor:
    def __init__(self, ckpt_path: str):
        self.ckpt = torch.load(ckpt_path, map_location="cpu")
        self.sbert_name = self.ckpt.get("sbert_name","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.enc = SentenceTransformer(self.sbert_name)
        in_dim = self.ckpt.get("in_dim", self.enc.encode(["x"], normalize_embeddings=True, convert_to_numpy=True).shape[1])
        hidden = self.ckpt.get("hidden", 512); dropout = self.ckpt.get("dropout", 0.1)
        import torch.nn as nn
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 4), nn.LogSoftmax(dim=-1)
        )
        self.model.load_state_dict(self.ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        B = 1024; out = []
        for i in range(0, len(X), B):
            xb = torch.from_numpy(X[i:i+B]).float()
            logp = self.model(xb); p = torch.exp(logp).cpu().numpy()
            out.append(p)
        P = np.vstack(out) if out else np.zeros((len(texts),4))
        return P

def _load_joblib_if_any():
    try:
        import joblib
        for name in ["tour_theme_regressor.joblib", "tour_theme_regressor_sgd.joblib"]:
            p = Path(MODELS_DIR) / name
            if p.exists():
                return joblib.load(str(p))
        return None
    except Exception:
        return None

from src.config import USE_TORCH

def encode_and_score(texts: List[str]) -> np.ndarray:
    if USE_TORCH:  # 🔘 토치 강제
        ckpt = _find_torch_model()
        if ckpt:
            try:
                return TorchPredictor(ckpt).predict(texts)
            except Exception as e:
                warnings.warn(f"Torch 예측 실패: {e} → Sklearn fallback")
        else:
            warnings.warn("Torch 모델 없음 → Sklearn fallback")

    # 🔘 기본: Sklearn
    reg = _load_joblib_if_any()
    if reg is not None:
        try:
            from sentence_transformers import SentenceTransformer
            enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            X = enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            if isinstance(reg, list):
                P = np.column_stack([m.predict(X) for m in reg])
            else:
                P = reg.predict(X)
            return _softmax_rows(P)
        except Exception as e:
            warnings.warn(f"Sklearn 예측 실패: {e}")

    # 🔘 마지막 안전망
    return _heuristic_scores(texts)


def score_and_save(df_or_csv: Union[pd.DataFrame, str],
                   out_csv: Optional[str] = str(SCORED_CSV)) -> pd.DataFrame:
    if isinstance(df_or_csv, str): df = pd.read_csv(df_or_csv)
    else: df = df_or_csv.copy()

    if "embedding_text" not in df.columns:
        raise ValueError("입력에 'embedding_text' 컬럼이 없습니다. 2단계를 먼저 실행하세요.")

    texts = df["embedding_text"].fillna("").astype(str).tolist()
    P = encode_and_score(texts)
    P = np.clip(P, 0, None)
    s = P.sum(axis=1, keepdims=True) + 1e-9
    P = P / s

    df["active"], df["culture"], df["nature"], df["relaxation"] = P[:,0], P[:,1], P[:,2], P[:,3]
    if out_csv:
        Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ CSV 저장: {out_csv} (행:{len(df)})")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(RAW_WITH_TEXTS_CSV))
    ap.add_argument("--output", type=str, default=str(SCORED_CSV))
    a = ap.parse_args()
    score_and_save(a.input, out_csv=a.output)

if __name__ == "__main__":
    main()
