import pandas as pd, numpy as np, joblib
from math import radians, sin, cos, sqrt, atan2
# from sentence_transformers import SentenceTransformer
# from src.config import SCORED_CSV, MODEL_PATH
from src.config import SCORED_CSV, PREF_SHARPEN_ENABLE, PREF_SHARPEN_GAMMA, USE_TORCH
import warnings

# ---------------- Config ----------------
CSV_PATH = str(SCORED_CSV) # 통합 CSV(점수 컬럼+좌표 포함)
USER_PREF = [0, 0, 0, 0]          # [active, culture, nature, relaxation]
TOP_K = 8
START_LATLON = (37.5665, 126.9780)  # (위도, 경도) 예: 서울시청 시작지점
TRAVEL_MODE = "metro"             # walk/bike/car/metro
SPEED_KMPH = {"walk":4.5, "bike":15, "car":30, "metro":25}[TRAVEL_MODE]

# ---------------- Utilities ----------------


def _sharpen_prob(p, gamma=1.6, eps=1e-9):
    p = np.clip(np.asarray(p, float), eps, None)
    p = p ** gamma
    return p / p.sum()

def _maybe_sharpen(p):
    return _sharpen_prob(p, PREF_SHARPEN_GAMMA) if PREF_SHARPEN_ENABLE else p

def _softmax_rows(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - X.max(axis=1, keepdims=True)
    ex = np.exp(X)
    return ex / (ex.sum(axis=1, keepdims=True) + eps)

def encode_user_pref_text(text: str):
    """
    사용자 입력 텍스트를 벡터로 인코딩하고,
    Torch 또는 Sklearn 모델을 통해 4차원 선호 분포로 변환
    """
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    X = enc.encode([text], normalize_embeddings=True, convert_to_numpy=True)

    if USE_TORCH:
        # Torch 모델 우선
        import torch
        ckpt_path = MODELS_DIR / "theme_regressor.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                import torch.nn as nn
                model = nn.Sequential(
                    nn.Linear(X.shape[1], 512), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(256, 4), nn.LogSoftmax(dim=-1)
                )
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                with torch.no_grad():
                    logp = model(torch.from_numpy(X).float())
                    p = torch.exp(logp).cpu().numpy()
                return p[0]
            except Exception as e:
                warnings.warn(f"Torch 예측 실패: {e} → Sklearn fallback")

    # Sklearn 기본
    try:
        import joblib
        reg = joblib.load(MODEL_PATH)
        if isinstance(reg, list):
            P = np.column_stack([m.predict(X) for m in reg])
        else:
            P = reg.predict(X)
        return _softmax_rows(P)[0]
    except Exception as e:
        warnings.warn(f"Sklearn 예측 실패: {e} → 휴리스틱 대체")
        return np.array([0.25,0.25,0.25,0.25])



def haversine_km(lat1, lon1, lat2, lon2):
    """두 좌표(위도(lat)/경도(lon)) 사이 구면거리(km)"""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(1-a), sqrt(a))
    return R * c

# ---------------- Pipeline ----------------
def load_and_score(csv_path, user_pref):
    df = pd.read_csv(csv_path)
    for col in ["mapy","mapx","active","culture","nature","relaxation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pref = np.array(user_pref, dtype=float)
    s = np.nansum(pref)
    if s <= 0:
        pref = np.ones_like(pref) / len(pref)
    else:
        pref = pref / s

    # 🔘 샤프닝 토글: config로 on/off
    pref = _maybe_sharpen(pref)

    theme = df[["active","culture","nature","relaxation"]].to_numpy(dtype=float)
    df["pref_score"] = theme @ pref   # 기존 공식 유지:contentReference[oaicite:2]{index=2}
    geo = df[df["mapy"].notna() & df["mapx"].notna()].copy()
    return geo

def build_distance_matrix(lat, lon):
    n = len(lat)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_km(lat[i], lon[i], lat[j], lon[j])
            D[i, j] = D[j, i] = d
    return D

def build_itinerary(df_selected, start_latlon, speed_kmph):
    if len(df_selected) == 0:
        return pd.DataFrame()

    lat = df_selected['mapy'].to_numpy()
    lon = df_selected['mapx'].to_numpy()

    legs = [haversine_km(start_latlon[0], start_latlon[1], lat[0], lon[0])]
    for i in range(len(df_selected) - 1):
        legs.append(haversine_km(lat[i], lon[i], lat[i+1], lon[i+1]))

    out = df_selected.copy().reset_index(drop=True)
    out["leg_km"]  = np.round(legs, 2) #이동 구간
    out["cum_km"]  = np.round(np.cumsum(out["leg_km"]), 2) #이동 합계
    out["eta_min"] = (out["leg_km"] / speed_kmph * 60).round().astype(int) #걸리는 시간

    cols = [c for c in ["title","pref_score","addr1","mapy","mapx","leg_km","cum_km","eta_min"] if c in out.columns]
    return out[cols]


def get_candidates(current_latlon, remaining_df, how="score_then_distance", k=5,
                   alpha=1.0, beta=0.3):
    """
    현재 위치 기준으로 remaining_df에서 상위 k개 후보를 반환
    how:
      - "score_then_distance": pref_score 내림차순으로 상위 50~100개 추린 후 현재 위치와의 거리 오름차순
      - "hybrid": alpha*pref_score - beta*distance_km 점수로 상위 k
      - "distance_only": 거리만 가까운 순
    """
    if remaining_df.empty:
        return remaining_df

    # 현재 위치로부터 거리 계산(임시 컬럼)
    distances = remaining_df.apply(
        lambda r: haversine_km(current_latlon[0], current_latlon[1], r["mapy"], r["mapx"]),
        axis=1
    )
    tmp = remaining_df.copy()
    tmp["_dist_km"] = distances

    if how == "score_then_distance":
        # 취향점수 상위 N(적당히 크게) → 거리 가까운 순
        N = min(len(tmp), max(k * 4, 20))
        top_by_score = tmp.sort_values("pref_score", ascending=False).head(N)
        picked = top_by_score.sort_values("_dist_km", ascending=True).head(k)

    elif how == "hybrid":
        # 혼합 점수
        tmp["_hybrid"] = alpha * tmp["pref_score"] - beta * tmp["_dist_km"]
        picked = tmp.sort_values("_hybrid", ascending=False).head(k)

    elif how == "distance_only":
        picked = tmp.sort_values("_dist_km", ascending=True).head(k)

    else:
        # 기본: 취향점수 내림차순 상위 k
        picked = tmp.sort_values("pref_score", ascending=False).head(k)

    return picked.drop(columns=[c for c in ["_dist_km","_hybrid"] if c in picked.columns])

def interactive_loop(csv_path=CSV_PATH, user_pref=USER_PREF,
                     start_latlon=START_LATLON, speed_kmph=SPEED_KMPH,
                     top_k_max=TOP_K, how="score_then_distance", k_candidates=5):
    """
    콘솔 대화형 루프:
      1) 현재 위치에서 후보 k개 보여줌
      2) 사용자에게 1~k 중 선택(또는 q로 종료)
      3) 선택한 장소를 경로에 추가하고, 현재 위치를 갱신
      4) (선택 수 == top_k_max) 또는 남은 후보 없음 → 종료
      5) 최종 결과표 출력
    """
    df = load_and_score(csv_path, user_pref)
    remaining = df.copy()
    route_rows = []
    current = start_latlon

    step = 1
    while len(route_rows) < top_k_max and not remaining.empty:
        candidates = get_candidates(current, remaining, how=how, k=k_candidates)

        if candidates.empty:
            print("\n더 이상 후보가 없습니다.")
            break

        # 화면 출력용 번호 매기기
        print(f"\n[{step}] 후보 {len(candidates)}곳 (번호를 골라주세요, 종료:q)")
        view_cols = [c for c in ["title","addr1","pref_score","mapy","mapx"] if c in candidates.columns]
        cand_view = candidates.reset_index().rename(columns={"index":"_orig_idx"})  # 원래 인덱스 보존
        for i, row in cand_view.iterrows():
            title = row["title"] if "title" in row else f"Place {i+1}"
            addr  = row["addr1"] if "addr1" in row else ""
            psc   = f"{row['pref_score']:.3f}" if "pref_score" in row else "-"
            print(f"  {i+1}. {title} | {addr} | score: {psc}")

        choice = input("선택(1~{0}) 또는 종료(q): ".format(len(cand_view))).strip().lower()
        if choice in ["q","quit","exit"]:
            break

        try:
            idx = int(choice)
            assert 1 <= idx <= len(cand_view)
        except Exception:
            print("잘못된 입력입니다. 다시 선택해주세요.")
            continue

        chosen = cand_view.iloc[idx-1]
        orig_idx = chosen["_orig_idx"]

        # 경로에 추가
        route_rows.append(remaining.loc[orig_idx])

        # 현재 위치 갱신
        current = (chosen["mapy"], chosen["mapx"])

        # remaining에서 제거
        remaining = remaining.drop(index=orig_idx)

        step += 1

    if len(route_rows) == 0:
        print("\n선택된 장소가 없습니다.")
        return pd.DataFrame()

    df_selected = pd.DataFrame(route_rows)
    itinerary = build_itinerary(df_selected, start_latlon, speed_kmph)

    print("\n최종 코스:")
    print(itinerary.to_string(index=False))
    return itinerary



# ---------------- Run ----------------
if __name__ == "__main__":
    mode = input("여행 선호를 어떻게 입력하시겠습니까?\n"
                 "1. 숫자(0~5 척도)\n"
                 "2. 원하는 여행 스타일 키워드(예: 활동적, 힐링, 문화)\n"
                 "선택(1/2): ").strip()
    if(mode == "1"):
        scale = 5.0
        print("0~5 척도로 입력하세요.")
        active = int(input("활동적인 여행 선호도 (active): "))
        culture = int(input("문화/역사 탐방 선호도 (culture): "))
        nature = int(input("자연/야외 선호도 (nature): "))
        relaxation = int(input("휴식/힐링 선호도 (relaxation): "))

        USER_PREF = [active, culture, nature, relaxation]
        USER_PREF = [x / scale for x in USER_PREF]
        print("입력된 사용자 성향:", USER_PREF) #사용자 성향에 대해 말할떄 특정 텍스트로 만들면 좋을듯 "문화와 역사를 좋아하는 ~" 이런식으로


    else:
        user_pref_text = input("원하시는 여행 스타일을 입력해주세요 : ")
        USER_PREF = encode_user_pref_text(user_pref_text)
        print("입력된 사용자 성향:", USER_PREF)

    # 대화형 코스 구성 시작
    # how 옵션: "score_then_distance" / "hybrid" / "distance_only"
    interactive_loop(
        csv_path=CSV_PATH,
        user_pref=USER_PREF,
        start_latlon=START_LATLON,
        speed_kmph=SPEED_KMPH,
        top_k_max=TOP_K,
        how="hybrid",
        k_candidates=5
    )
