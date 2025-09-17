# train_regressor_cli.py
# -*- coding: utf-8 -*-
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDRegressor
from src.config import LABEL_AUTO as DEFAULT_CSV, MODEL_PATH as DEFAULT_MODEL_PATH

N_OUT = 4  # active, culture, nature, relaxation

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need_cols = {"text", "active", "culture", "nature", "relaxation"}
    missing = need_cols - set(map(str.lower, df.columns))
    # 관용적으로 대소문자 섞여도 허용
    colmap = {c.lower(): c for c in df.columns}
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")
    return df.rename(columns={colmap["text"]:"text",
                              colmap["active"]:"active",
                              colmap["culture"]:"culture",
                              colmap["nature"]:"nature",
                              colmap["relaxation"]:"relaxation"})

def embed_texts(texts: List[str]) -> np.ndarray:
    print("📌 임베딩 시작...")
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    X = enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return X

def init_models(random_state: int = 42) -> List[SGDRegressor]:
    base_kwargs = dict(loss="squared_error", max_iter=1000, tol=1e-3, random_state=random_state)
    return [SGDRegressor(**base_kwargs) for _ in range(N_OUT)]

def save_models(models: List[SGDRegressor], path: str):
    joblib.dump(models, path)
    print(f"✅ 모델 저장 완료: {path}")

def load_models(path: str) -> Optional[List[SGDRegressor]]:
    if not os.path.exists(path):
        return None
    obj = joblib.load(path)
    # 기존에 MultiOutputRegressor로 저장했더라도 list로 변환 지원
    if isinstance(obj, list):
        return obj
    # 예전 Ridge/MultiOutput 저장물을 로드했다면 호환 불가 → None 처리(새로 학습)
    try:
        # 모델 객체에 predict가 있더라도 partial_fit 없는 경우가 많음
        # 점진학습을 일관되게 쓰려면 list[SGDRegressor]만 허용
        return None
    except Exception:
        return None

def fit_or_incremental(models: List[SGDRegressor], X: np.ndarray, Y: np.ndarray, mode: str) -> List[SGDRegressor]:
    """
    mode: 'fit' | 'incremental'
    """
    if mode == "fit":
        print("📌 새로 학습(fit) 진행...")
        models = init_models()  # 항상 초기화해서 완전 재학습
        for i in range(N_OUT):
            models[i].fit(X, Y[:, i])
    else:
        print("📌 점진적 학습(partial_fit) 진행...")
        # models가 None이면 새로 초기화
        if models is None:
            models = init_models()
        for i in range(N_OUT):
            models[i].partial_fit(X, Y[:, i])
    return models

def main():
    parser = argparse.ArgumentParser(description="Train theme regressor (fit / incremental / auto).")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="학습용 CSV 경로 (text, active, culture, nature, relaxation)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="모델 저장 경로 (.joblib)")
    parser.add_argument("--mode", type=str, choices=["auto", "fit", "incremental"], default="auto",
                        help="auto: 모델 있으면 incremental, 없으면 fit")
    parser.add_argument("--shuffle", action="store_true", help="학습 전 데이터 셔플")
    args = parser.parse_args()

    # 1) 데이터 로드
    df = load_csv(args.csv)
    texts = df["text"].astype(str).tolist()
    Y = df[["active","culture","nature","relaxation"]].to_numpy(dtype=float)

    if args.shuffle:
        idx = np.random.permutation(len(df))
        Y = Y[idx]
        texts = [texts[i] for i in idx]

    # 2) 임베딩
    X = embed_texts(texts)

    # 3) 모델 로드/초기화 및 모드 결정
    models = load_models(args.model)
    if args.mode == "auto":
        mode = "incremental" if models is not None else "fit"
    else:
        mode = args.mode
    print(f"📌 동작 모드: {mode} (모델 경로: {args.model})")

    # 4) 학습
    models = fit_or_incremental(models, X, Y, mode)

    # 5) 저장
    save_models(models, args.model)

if __name__ == "__main__":
    main()
