# -*- coding: utf-8 -*-
"""
Step 1: TourAPI 수집 모듈
- areaBasedList2로 목록 가져오기
- 각 항목에 detailCommon2(이미지/주소/좌표/개요 등) 병합
- ALLOW_TYPES 필터링 ([12,14,15,25])
- pandas.DataFrame으로 반환 + CSV 저장(옵션)

사용 예:
    from fetch_tourapi import fetch_items_to_df
    df = fetch_items_to_df(area_code=1, max_pages=3, rows_per_page=100, save_csv="tour_raw_seoul.csv")

사전 준비:
    pip install requests certifi pandas numpy tqdm
    (Windows OpenSSL 이슈 있으면 TLSAdapter 그대로 사용)
"""

import os, ssl, re, time, certifi, requests
from typing import Dict, Any, List, Tuple, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from src.config import RAW_CSV
from tqdm import tqdm

# ===================== 설정 =====================
BASE = "https://apis.data.go.kr/B551011/KorService2"
# 환경변수 TOURAPI_KEY 사용
# KEY  = os.getenv("TOURAPI_KEY") 나중에 환경변수 사용시
KEY  = "lKVgn2HAR6u0nXIaIUJqk9prAM9IDVZiNwtrgUtswKHnXPZ9KlGYT2xuRgygQan9LFxgYmFuxpdlj56jphC81Q=="

TYPE_FLAGS = {
    12: "Y",  # 관광지
    14: "Y",  # 문화시설
    15: "N",  # 축제 공연행사
    25: "Y",  # 여행코스
    28: "Y",  # 레저스포츠
    32: "N",  # 숙박
    38: "Y",  # 쇼핑
    39: "N",  # 음식점
}

# 실제 사용할 때는 숫자 리스트로 변환
ALLOW_TYPES = [t for t, flag in TYPE_FLAGS.items() if flag == "Y"]

# ===================== TLS 어댑터 =====================
class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

def _build_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", TLSAdapter())
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

SESSION = _build_session()

# ===================== 유틸 =====================
def _as_list(x):
    if isinstance(x, list): return x
    if isinstance(x, dict): return [x]
    return []

def _strip_html(x: str) -> str:
    if not isinstance(x, str): return ""
    x = re.sub(r"<br\s*/?>", "\n", x, flags=re.I)
    x = re.sub(r"<[^>]+>", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _tour_get(path: str, params: Dict[str, Any], timeout: int = 12) -> Dict[str, Any]:
    base_params = {
        "serviceKey": KEY,
        "MobileOS": "ETC",
        "MobileApp": "YeotteoJaem",
        "_type": "json",
    }
    p = {**base_params, **params}
    r = SESSION.get(f"{BASE}/{path}", params=p, verify=certifi.where(), timeout=timeout)
    r.raise_for_status()
    return r.json()

# ===================== API 래퍼 =====================
def fetch_area_list(area_code: int, page_no: int, rows: int) -> Tuple[List[Dict[str, Any]], int]:
    """
    areaBasedList2 호출 (한 페이지)
    Returns: (items, totalCount)
    """
    d = _tour_get("areaBasedList2", {
        "areaCode": area_code,
        "pageNo": page_no,
        "numOfRows": rows
    })
    body = d.get("response", {}).get("body", {}) or {}
    items = _as_list(body.get("items", {}).get("item", []))
    total = int(body.get("totalCount", 0))
    # 타입 필터
    items = [it for it in items if int(it.get("contenttypeid", 0)) in ALLOW_TYPES]
    return items, total

def fetch_common(contentid: str, contenttypeid: Optional[int] = None) -> Dict[str, Any]:
    params = { "contentId": contentid }  # v4.3: 이것만 필수
    # contentTypeId는 이제 요청 항목에서 빠졌으므로 보내지 않아도 됨
    d = _tour_get("detailCommon2", params)
    it = _as_list(d.get("response", {}).get("body", {}).get("items", {}).get("item", []))
    return it[0] if it else {}


# ===================== 상위 수준 수집 함수 =====================
def fetch_items_to_df(
    area_code: int = 1,
    max_pages: int = 100,
    rows_per_page: int = 100,
    qps_delay: float = 0.05,
    save_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    지역(area_code) 기준으로 페이지네이션 수집:
      1) 목록 호출
      2) 각 항목 detailCommon2로 확장
      3) 정규화된 DataFrame 반환
    옵션: save_csv 경로 주면 CSV 저장

    반환 컬럼(예):
      contentid, contenttypeid, title, addr1, mapx, mapy, firstimage,
      overview, eventstartdate, eventenddate, cat1, cat2, cat3, areacode, sigungucode
    """
    all_rows: List[Dict[str, Any]] = []
    total_seen = 0
    total_from_api = None

    for page in range(1, max_pages + 1):
        items, total = fetch_area_list(area_code, page, rows_per_page)
        if total_from_api is None:
            total_from_api = total
            print(f"[Info] 서버 totalCount: {total_from_api}")

        if not items:
            print(f"[Info] 페이지 {page}: 항목 없음. 중단")
            break

        for it in tqdm(items, desc=f"페이지 {page} 상세 수집"):
            cid = str(it.get("contentid"))
            ctyp = int(it.get("contenttypeid", 0)) if it.get("contenttypeid") else None

            cmn = fetch_common(cid, ctyp)
            row = {
                "contentid": cid,
                "contenttypeid": int(cmn.get("contenttypeid", it.get("contenttypeid", 0))),
                "title": cmn.get("title", it.get("title", "")),
                "addr1": cmn.get("addr1", it.get("addr1", "")),
                "mapx": cmn.get("mapx", it.get("mapx", None)),
                "mapy": cmn.get("mapy", it.get("mapy", None)),
                "firstimage": cmn.get("firstimage", it.get("firstimage", "")),
                "overview": _strip_html(cmn.get("overview", "")),
                "eventstartdate": cmn.get("eventstartdate", ""),
                "eventenddate": cmn.get("eventenddate", ""),
                "cat1": cmn.get("cat1", it.get("cat1", "")),
                "cat2": cmn.get("cat2", it.get("cat2", "")),
                "cat3": cmn.get("cat3", it.get("cat3", "")),
                "areacode": cmn.get("areacode", it.get("areacode", "")),
                "sigungucode": cmn.get("sigungucode", it.get("sigungucode", "")),
            }
            # 숫자형 변환(가능하면)
            try:
                if row["mapx"] is not None: row["mapx"] = float(row["mapx"])
                if row["mapy"] is not None: row["mapy"] = float(row["mapy"])
            except Exception:
                pass

            all_rows.append(row)
            time.sleep(qps_delay)  # 과도한 호출 방지

        total_seen += len(items)
        if total_seen >= total_from_api:
            break

    df = pd.DataFrame(all_rows)
    if save_csv:
        df.to_csv(save_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 저장 완료: {save_csv} (행:{len(df)})")
    return df

# ===================== CLI 예시 =====================
if __name__ == "__main__":
    from fetch_tourapi import fetch_items_to_df
    fetch_items_to_df(
        area_code=1,
        max_pages=3,
        rows_per_page=100,
        save_csv=str(RAW_CSV)
    )
