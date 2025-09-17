# -*- coding: utf-8 -*-
# generate_train_labels_from_step2.py
import pandas as pd
import numpy as np
import re
from src.config import RAW_WITH_TEXTS_CSV, LABEL_DIR

SRC = RAW_WITH_TEXTS_CSV
OUT_AUTO = LABEL_DIR / "train_labels_auto.csv"
OUT_TEMPLATE = LABEL_DIR / "train_labels_human_input.csv"


def build_text(row):
    t = str(row.get("embedding_text") or "").strip()
    if not t:
        title = str(row.get("title") or "")
        cats  = " ".join([str(row.get(c) or "") for c in ["cat1","cat2","cat3"]]).strip()
        addr  = str(row.get("addr1") or "")
        t = " ".join([title, cats, addr]).strip()
    t = re.sub(r"\s+", " ", t)
    return t[:2000]

KEYWORDS = {
    "active":  ["등산","트레킹","하이킹","캠핑","서핑","마라톤","자전거","패러글라이딩","래프팅","체험","레포츠","스포츠","클라이밍","암벽","산행","러닝","승마","카약","카누","서바이벌"],
    "culture": ["박물관","미술관","전시","전통","역사","궁궐","고궁","사찰","유적","기념관","공연","연극","뮤지컬","문화재","예술","아트","문화센터","도서관","아카이브","축제","페스티벌"],
    "nature":  ["산","공원","바다","해변","호수","강","계곡","숲","정원","습지","폭포","해안","둘레길","자연","하천","산책로","초원","국립공원","자연공원","생태공원","갯벌"],
    "relax":   ["스파","온천","리조트","휴양림","힐링","명상","요가","찜질방","호텔","카페","정원 산책","휴식","야경","전망대","호캉스","글램핑","풀빌라","리트릿","쉼터","휴양"],
}

def score_from_keywords(text):
    counts = []
    for k in ["active","culture","nature","relax"]:
        c = sum(1 for kw in KEYWORDS[k] if kw in text)
        counts.append(c)
    v = np.array(counts, dtype=float)
    v = v + 0.1           # 스무딩 (전부 0 방지)
    v = v / v.sum()       # 확률화
    return v

def main():
    df = pd.read_csv(SRC)
    texts = df.apply(build_text, axis=1)
    scores = np.vstack([score_from_keywords(t) for t in texts])

    out = pd.DataFrame({
        "text": texts,
        "active": scores[:,0],
        "culture": scores[:,1],
        "nature": scores[:,2],
        "relaxation": scores[:,3],
    }).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # 크기가 너무 크면 상한 설정(예: 최대 2000행)
    out = out.head(2000)

    out.to_csv(OUT_AUTO, index=False, encoding="utf-8-sig")
    # 사람 보정용 템플릿: 점수 칸 비워서 줌
    tmpl = out.copy()
    tmpl[["active","culture","nature","relaxation"]] = ""
    tmpl.to_csv(OUT_TEMPLATE, index=False, encoding="utf-8-sig")

    print(f"✅ 자동 라벨 저장: {OUT_AUTO} (행:{len(out)})")
    print(f"✅ 보정 템플릿 저장: {OUT_TEMPLATE}")

if __name__ == "__main__":
    main()
