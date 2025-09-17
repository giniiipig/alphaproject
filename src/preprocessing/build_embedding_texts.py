# -*- coding: utf-8 -*-
"""
Step 2: embedding_text 생성 모듈
- overview 사용
- overview 없으면 intro 조합
- 그래도 없으면 타이틀/주소/분류 기반 rich fallback

출력: 원본 DF + embedding_text 컬럼
"""

import pandas as pd
import time, re
from typing import Union, Optional, List
from tqdm import tqdm

# ====== 유틸 ======
def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]*>", " ", text).strip()

def _get_intro(contentid: str, contenttypeid: int) -> dict:
    """
    TODO: TourAPI intro 요청 함수 자리
    현재는 빈 dict 리턴 (실사용 시 API 연동)
    """
    return {}

def _synthesize_from_intro(ctype: int, intro: dict) -> str:
    """
    intro dict에서 의미 있는 텍스트 조합
    """
    parts = []
    if not intro:
        return ""
    for k, v in intro.items():
        if v and isinstance(v, str) and len(v.strip()) > 0:
            parts.append(f"{k}:{v.strip()}")
    return " ".join(parts)

def _rich_fallback(row) -> str:
    """overview/intro가 없는 경우 풍부한 문장 생성"""
    title = str(row.get("title") or "")
    addr = str(row.get("addr1") or "")
    cat1, cat2, cat3 = str(row.get("cat1") or ""), str(row.get("cat2") or ""), str(row.get("cat3") or "")
    ctype = str(row.get("contenttypeid") or "")

    if ctype == "15":  # 행사
        return f"{title} (행사, {addr}). 다양한 체험과 공연이 열리는 이벤트 장소입니다."
    elif cat1 == "A01":  # 자연관광지
        return f"{title} (자연관광지, {addr}). 산, 숲, 계곡 등 자연경관을 체험할 수 있는 곳입니다."
    elif cat1 == "A02":  # 문화관광지
        return f"{title} (문화관광지, {addr}). 역사와 문화를 배울 수 있는 명소입니다."
    elif cat1 == "A03":  # 레포츠
        return f"{title} (레포츠, {addr}). 다양한 액티비티와 체험을 즐길 수 있습니다."
    else:
        return f"{title} ({addr}). 분류: {cat1}/{cat2}/{cat3}."

# ====== 메인 함수 ======
def add_embedding_texts(
    df_or_csv: Union[pd.DataFrame, str],
    qps_delay: float = 0.05,
    save_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    입력 DataFrame(또는 CSV 경로)에 대해 embedding_text 컬럼을 생성해 반환.
    반환: 원본 컬럼 + embedding_text
    """
    if isinstance(df_or_csv, str):
        df = pd.read_csv(df_or_csv)
    else:
        df = df_or_csv.copy()

    need_intro_idx = []
    texts: List[str] = []

    # 1) overview 우선
    for i, row in tqdm(df.iterrows(), total=len(df), desc="임베딩 텍스트 생성"):
        ctype = int(row.get("contenttypeid", 0)) if pd.notna(row.get("contenttypeid")) else 0
        overview = _strip_html(str(row.get("overview") or ""))

        if overview:
            # 행사면 기간 프리픽스 붙이기
            if ctype == 15:
                es = str(row.get("eventstartdate") or "").strip()
                ee = str(row.get("eventenddate") or "").strip()
                if es or ee:
                    overview = f"[행사기간 {es}~{ee}] " + overview
            texts.append(overview)
            continue

        # overview 없음 → intro 후보
        need_intro_idx.append(i)
        texts.append("")  # placeholder

    # 2) intro 조회
    for i in tqdm(need_intro_idx, desc="intro 조회"):
        row = df.loc[i]
        cid = str(row.get("contentid"))
        ctype = int(row.get("contenttypeid", 0)) if pd.notna(row.get("contenttypeid")) else 0

        intro = _get_intro(cid, ctype)
        intro_txt = _synthesize_from_intro(ctype, intro)

        if intro_txt:
            texts[i] = intro_txt
        else:
            texts[i] = _rich_fallback(row)

        time.sleep(qps_delay)

    # 3) 결과 저장
    df["embedding_text"] = texts

    if save_csv:
        df.to_csv(save_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 저장 완료: {save_csv} (행:{len(df)})")

    return df

# ====== CLI 실행 예시 ======
if __name__ == "__main__":
    df = add_embedding_texts("tour_raw_seoul.csv", save_csv="tour_raw_with_texts.csv")
    print(df[["title", "embedding_text"]].head())
