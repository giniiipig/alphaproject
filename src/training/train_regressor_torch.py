
# -*- coding: utf-8 -*-
"""
train_auto_torch.py
-------------------
여떠잼: "신규 학습(기본)" vs "추가 학습(재학습)" 자동 구분 스크립트

규칙
- 기본 경로(models/theme_regressor.pt)에 체크포인트가 없으면 → 신규 학습
  * train_labels_auto.csv + train_labels_human_input.csv 병합 → train_labels_merged.csv
  * 병합본으로 학습
  * 저장: models/theme_regressor_base.pt (+ 최신 모델 별칭: models/theme_regressor.pt 로도 업데이트)

- 체크포인트가 이미 있거나, --additional_csv 를 지정하면 → 추가 학습(이어학습)
  * --additional_csv (기본: train_labels_template_for_humans.csv가 있으면 그것, 없으면 --train_csv) 로 이어학습
  * (옵션) --replay_csv 로 과거 큰 데이터 일부 섞기(--replay_ratio)
  * 저장: models/theme_regressor_additional_YYYYMMDDHHMM.pt (+ 최신 모델 별칭 업데이트)

설치
pip install torch sentence-transformers pandas scikit-learn

사용 예시
1) 자동 모드(권장):
python train_auto_torch.py

2) 명시적으로 추가 학습(새 소량 라벨로 이어하기):
python train_auto_torch.py --additional_csv ./train_labels_template_for_humans.csv --epochs 10 --lr 1e-4

3) 추가 학습 + 리플레이 혼합:
python train_auto_torch.py --additional_csv ./new_small_labels.csv --replay_csv ./train_labels_merged.csv --replay_ratio 0.3 --epochs 12 --lr 1e-4
"""

import os
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer


# ===================== 공통 유틸 =====================

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_prob_norm(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    y = np.clip(y, 0, None)
    s = y.sum(axis=-1, keepdims=True) + eps
    return y / s

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """text, active, culture, nature, relaxation 컬럼 강제 정규화"""
    colmap = {c.lower(): c for c in df.columns}
    need = ["text","active","culture","nature","relaxation"]
    miss = [c for c in need if c not in colmap]
    if miss:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {miss}. 실제 컬럼: {list(df.columns)}")
    df = df.rename(columns={
        colmap["text"]: "text",
        colmap["active"]: "active",
        colmap["culture"]: "culture",
        colmap["nature"]: "nature",
        colmap["relaxation"]: "relaxation",
    })
    df["text"] = df["text"].astype(str)
    for c in ["active","culture","nature","relaxation"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def merge_auto_human(auto_csv: str, human_csv: str, out_csv: str) -> pd.DataFrame:
    """auto + human_input 병합. 사람이 쓴 값 우선, 결측은 auto 값 사용. 행별 확률 정규화."""
    auto = ensure_cols(pd.read_csv(auto_csv))
    human = ensure_cols(pd.read_csv(human_csv))

    # 기준: text
    merged = auto.merge(human, on="text", how="outer", suffixes=("_auto","_human"))
    out = pd.DataFrame({"text": merged["text"]})
    for c in ["active","culture","nature","relaxation"]:
        ca, ch = f"{c}_auto", f"{c}_human"
        va = merged[ca].fillna(0.0)
        vh = merged[ch].fillna(np.nan)
        # 사람이 쓴 값 우선. 사람이 없으면 auto 사용
        v = vh.where(~vh.isna(), va)
        out[c] = v.astype(float)

    # 행별 확률 정규화
    P = out[["active","culture","nature","relaxation"]].to_numpy(dtype=float)
    out[["active","culture","nature","relaxation"]] = safe_prob_norm(P)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 병합 CSV 저장: {out_csv} (행:{len(out)})")
    return out


# ===================== 데이터셋/모델 =====================

class ThemeDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, sbert_name: str):
        self.labels = labels.astype(np.float32)
        self.enc = SentenceTransformer(sbert_name)
        self.X = self.enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.labels[idx]).float()

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 4),
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, x):
        return self.log_softmax(self.net(x))


# ===================== 학습 루프 =====================

@dataclass
class TrainCfg:
    sbert_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.0
    hidden: int = 512
    dropout: float = 0.1
    seed: int = 42

def kldiv_loss(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.kl_div(log_probs, targets, reduction="batchmean")

def distribution_metrics(log_probs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        preds = torch.exp(log_probs)
        mae = torch.mean(torch.abs(preds - targets)).item()
        cos = torch.nn.functional.cosine_similarity(preds, targets, dim=-1).mean().item()
    return mae, cos

def prepare_model(sbert_name: str, hidden: int, dropout: float, device: torch.device) -> Tuple[MLPRegressor, int, SentenceTransformer]:
    enc = SentenceTransformer(sbert_name)
    in_dim = enc.encode(["hello"], normalize_embeddings=True, convert_to_numpy=True).shape[1]
    model = MLPRegressor(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
    return model, in_dim, enc

def train_one(df: pd.DataFrame, cfg: TrainCfg,
              resume_ckpt: Optional[str],
              out_ckpt: str) -> str:
    set_seed(cfg.seed)

    df = ensure_cols(df)
    texts = df["text"].tolist()
    labels = safe_prob_norm(df[["active","culture","nature","relaxation"]].to_numpy(dtype=float))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # enc은 Dataset 내부에서 새로 로드 → 여기선 차원 확인용만 사용
    sample_enc = SentenceTransformer(cfg.sbert_name)
    in_dim = sample_enc.encode(["x"], normalize_embeddings=True, convert_to_numpy=True).shape[1]

    dataset = ThemeDataset(texts, labels, cfg.sbert_name)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    model = MLPRegressor(in_dim=in_dim, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")

    # ===== 재개 옵션 =====
    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optim_state" in ckpt:
            try:
                optim.load_state_dict(ckpt["optim_state"])
            except Exception:
                pass
        print(f"🔁 이어서 학습 재개: {resume_ckpt}")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad()
            logp = model(xb)
            loss = kldiv_loss(logp, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item(); n += 1
        print(f"[Epoch {ep:02d}] train_loss={tot/max(n,1):.4f}")

    # 저장
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "sbert_name": cfg.sbert_name,
        "in_dim": in_dim,
        "hidden": cfg.hidden,
        "dropout": cfg.dropout,
        "optim_state": optim.state_dict(),
    }, out_ckpt)
    print(f"✅ 모델 저장: {out_ckpt}")
    return out_ckpt


# ===================== 메인 로직 =====================

def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto_mode", type=lambda x: str(x).lower() != "false", default=True,
                    help="True면 자동 판단(기본: True)")
    ap.add_argument("--model_latest", type=str, default="./models/theme_regressor.pt",
                    help="최신 모델 별칭 경로")
    ap.add_argument("--base_ckpt_name", type=str, default="./models/theme_regressor_base.pt",
                    help="신규(기본) 학습 저장 경로")
    ap.add_argument("--additional_dir", type=str, default="./models",
                    help="추가학습 저장 디렉토리")

    # 데이터셋 경로들
    ap.add_argument("--auto_csv", type=str, default="./train_labels_auto.csv")
    ap.add_argument("--human_csv", type=str, default="./train_labels_human_input.csv")
    ap.add_argument("--merged_csv", type=str, default="./train_labels_merged.csv",
                    help="신규 학습 시 생성되는 병합 CSV 경로")
    ap.add_argument("--train_csv", type=str, default="./train_labels_human_input.csv",
                    help="수동/기타용 기본 학습 입력(자동 모드에선 보통 병합본 사용)")
    ap.add_argument("--additional_csv", type=str, default=None,
                    help="추가학습 전용 CSV (기본: train_labels_template_for_humans.csv 있으면 그것, 없으면 --train_csv)")
    ap.add_argument("--replay_csv", type=str, default=None,
                    help="추가학습 시 과거 데이터 일부 섞을 때 사용")
    ap.add_argument("--replay_ratio", type=float, default=0.0,
                    help="추가학습 시 (신규:리플레이) 비율; 0이면 비활성")

    # 학습 설정
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # 수동 모드
    ap.add_argument("--mode", type=str, choices=["fit","continue"], default="fit",
                    help="auto_mode=False에서만 사용")

    args = ap.parse_args()

    cfg = TrainCfg(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, hidden=args.hidden, dropout=args.dropout, seed=args.seed
    )

    # 추가학습 CSV 기본값(auto-detect)
    if args.additional_csv is None:
        default_add = "./train_labels_template_for_humans.csv"
        args.additional_csv = default_add if file_exists(default_add) else args.train_csv

    # 의사결정: 신규 vs 추가
    if args.auto_mode:
        latest_exists = file_exists(args.model_latest)
        if not latest_exists:
            # ===== 신규(기본) 학습 =====
            print("🆕 신규(기본) 학습 모드")
            if not (file_exists(args.auto_csv) and file_exists(args.human_csv)):
                raise FileNotFoundError("신규 학습에는 train_labels_auto.csv, train_labels_human_input.csv 가 모두 필요합니다.")

            df_merged = merge_auto_human(args.auto_csv, args.human_csv, args.merged_csv)
            out_path = args.base_ckpt_name
            _ = train_one(df_merged, cfg, resume_ckpt=None, out_ckpt=out_path)

            # 최신 별칭 업데이트
            try:
                os.makedirs(os.path.dirname(args.model_latest) or ".", exist_ok=True)
                # 심볼릭 링크 실패 시 덮어쓰기 복사
                import shutil
                shutil.copyfile(out_path, args.model_latest)
            except Exception:
                pass
            print(f"🔗 최신 모델 별칭 업데이트: {args.model_latest} -> {out_path}")

        else:
            # ===== 추가 학습(이어학습) =====
            print("🔁 추가 학습(이어학습) 모드")
            # 베이스로는 기존 latest ckpt 사용
            resume_ckpt = args.model_latest
            # 입력 데이터 구성
            df_add = ensure_cols(pd.read_csv(args.additional_csv))

            if args.replay_csv and file_exists(args.replay_csv) and args.replay_ratio > 0:
                df_rep = ensure_cols(pd.read_csv(args.replay_csv))
                # 비율대로 샘플링 결합
                n_new = len(df_add)
                n_rep = int(n_new * args.replay_ratio / (1 - args.replay_ratio))
                if n_rep > 0 and len(df_rep) > 0:
                    idx = np.random.choice(len(df_rep), size=min(n_rep, len(df_rep)), replace=False)
                    df_add = pd.concat([df_add, df_rep.iloc[idx]], ignore_index=True)
                    print(f"📌 리플레이 샘플 추가: {len(idx)}")

            ts = datetime.now().strftime("%Y%m%d%H%M")
            out_name = f"theme_regressor_additional_{ts}.pt"
            out_path = os.path.join(args.additional_dir, out_name)
            _ = train_one(df_add, cfg, resume_ckpt=resume_ckpt, out_ckpt=out_path)

            # 최신 별칭 업데이트
            try:
                os.makedirs(os.path.dirname(args.model_latest) or ".", exist_ok=True)
                import shutil
                shutil.copyfile(out_path, args.model_latest)
            except Exception:
                pass
            print(f"🔗 최신 모델 별칭 업데이트: {args.model_latest} -> {out_path}")

    else:
        # 수동 모드
        if args.mode == "fit":
            print("🆕 수동: 신규(기본) 학습 모드")
            df = ensure_cols(pd.read_csv(args.train_csv))
            out_path = args.base_ckpt_name
            _ = train_one(df, cfg, resume_ckpt=None, out_ckpt=out_path)
            import shutil
            try:
                shutil.copyfile(out_path, args.model_latest)
            except Exception:
                pass
        else:
            print("🔁 수동: 추가 학습(이어학습) 모드")
            resume_ckpt = args.model_latest if file_exists(args.model_latest) else args.base_ckpt_name
            df = ensure_cols(pd.read_csv(args.additional_csv))
            ts = datetime.now().strftime("%Y%m%d%H%M")
            out_name = f"theme_regressor_additional_{ts}.pt"
            out_path = os.path.join(args.additional_dir, out_name)
            _ = train_one(df, cfg, resume_ckpt=resume_ckpt, out_ckpt=out_path)
            import shutil
            try:
                shutil.copyfile(out_path, args.model_latest)
            except Exception:
                pass

if __name__ == "__main__":
    main()
