# -*- coding: utf-8 -*-
# merge_human_labels.py
import pandas as pd
from src.config import LABEL_DIR

AUTO = LABEL_DIR / "train_labels_auto.csv"
HUMAN = LABEL_DIR / "train_labels_human_input.csv"
OUT  = LABEL_DIR / "train_labels_merged.csv"

def as_float_or_none(x):
    try:
        return float(x)
    except:
        return None

def main():
    auto = pd.read_csv(AUTO)
    human = pd.read_csv(HUMAN)

    # 정리
    human["text"] = human["text"].astype(str)
    # 숫자 변환 못하면 None → 이후 자동 값 사용
    for c in ["active","culture","nature","relaxation"]:
        human[c] = human[c].apply(as_float_or_none)

    # text 기준 merge
    merged = auto.merge(human, on="text", how="left", suffixes=("_auto","_human"))

    # 사람이 쓴 값 있으면 그걸 사용, 아니면 자동 값 사용
    out = pd.DataFrame({"text": merged["text"]})
    for c in ["active","culture","nature","relaxation"]:
        out[c] = merged[f"{c}_human"].combine_first(merged[f"{c}_auto"])

    # 행별로 합이 0이거나 NaN이면 자동 값으로 대체(안전망)
    out[["active","culture","nature","relaxation"]] = out[["active","culture","nature","relaxation"]].fillna(0)
    s = out[["active","culture","nature","relaxation"]].sum(axis=1).replace(0, 1)
    out["active"]      = out["active"] / s
    out["culture"]     = out["culture"] / s
    out["nature"]      = out["nature"] / s
    out["relaxation"]  = out["relaxation"] / s

    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"✅ 병합 저장: {OUT} (행:{len(out)})")

if __name__ == "__main__":
    main()
