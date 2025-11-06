# CLIP 기반 제로샷 테마 분류 (CSV 병합 결과만 저장)
import time

start=time.perf_counter()
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageOps  # EXIF 회전 보정 추가
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import concurrent.futures as cf

from io import BytesIO #바이트 배열을 파일처럼 다루는 버퍼
import requests #http 요청으로 이미지 다운로드
import os
import time  # 재시도 백오프를 위한 time
import math  # 배치 수 계산을 위한 math

MODEL_NAME="openai/clip-vit-base-patch32" #사용할 CLIP 모델
BATCH_SIZE = 48 #한번에 처리할 이미지 수, 낮출 수 있음
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16   = torch.cuda.is_available() # GPU 환경에서 FP16 가속 사용 여부
DOWNLOAD_WORKERS = 12 

# ===== 보강 옵션(기본 동작 유지) =====
# - 안정적인 다운로드(세션/재시도/용량 상한)를 위한 설정
HTTP_TIMEOUT = 8.0
MAX_RETRIES = 3
BACKOFF_BASE = 0.9
MAX_IMAGE_BYTES = 12 * 1024 * 1024  # 12MB 상한

# - 체크포인트(중간 저장) 주기: 큰 CSV 처리 중 중도 종료 대비
CHECKPOINT_EVERY = 10  # [FIX] 배치 N개마다 저장 (0 또는 음수면 비활성) — 5 → 10으로 조정

# ===== 2) 디코딩 가속 관련 옵션 =====
# [ADD] TurboJPEG 경로에서 EXIF 회전 보정은 기본 생략(최대 속도). 필요 시 True로.
TURBOJPEG_EXIF_TRANSPOSE = False

# CSV(링크) 모드 / 컬럼명 지정
USE_CSV   = True
INPUT_CSV = r"C:\Users\HAN\Desktop\alpha\tour_raw_with_texts.csv"  # 내 CSV 파일
TITLE_COL = "title"               # 장소명 컬럼
URL_COL   = "firstimage"          # 이미지 링크 컬럼
OUT_MERGED_CSV = r"C:\Users\HAN\Desktop\alpha\merged_scores.csv"
OUT_ERRORS_CSV = r"C:\Users\HAN\Desktop\alpha\load_errors.csv" # 이미지 로딩 에러 목록

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # [ADD] 입력 크기 고정 시 커널 선택 최적화
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

##프롬포트(테마 구분) 정의

THEMES = {
    "활동적": [
        "액티비티, 레저, 체험 요소가 중심인 사진",
        "자전거, 카약, 하이킹, 암벽 등반, 서핑, 러닝 등 활동 장면",
        "놀이기구, 스포츠, 야외 체험이 핵심인 관광 사진",
        "단체로 함께 움직이거나 에너지가 느껴지는 장면",
        "격렬한 신체 활동이나 모험적인 스포츠가 보이는 사진",
        "A photo focused on activities or experiences: biking, kayaking, hiking, climbing, surfing, running, adventure sports."
    ],
    "문화/역사": [
        "역사·문화 유적 또는 전통 건축물이 중심인 사진",
        "궁궐, 성곽, 사찰, 고궁, 한옥 등 전통 양식 건축 사진",
        "문화재, 박물관, 기념비, 고고학적 유산을 담은 관광 사진",
        "종교적 의미가 있는 사찰이나 성당 내부 사진",
        "전통 의식, 행사, 공연을 담은 장면",
        "A photo of cultural or historical sites with traditional architecture: palace, temple, fortress, hanok, heritage landmarks."
    ],
    "자연/야외": [
        "자연 경관이 돋보이는 사진, 숲이나 산, 바다, 강, 호수 등",
        "푸른 하늘과 풍경 중심의 야외 사진, 국립공원, 해변, 호수",
        "인공물보다 자연 요소가 주피사체인 풍경 사진",
        "계절 변화를 보여주는 자연 사진: 단풍, 벚꽃, 눈 덮인 산",
        "인적이 드문 곳의 한적한 자연 풍경",
        "A photo of natural outdoor scenery: forest, mountain, sea, river, lake; national parks, beaches, seasonal landscapes."
    ],
    "휴식/힐링": [
        "휴식과 힐링이 중심인 사진, 스파, 온천, 카페의 아늑한 분위기",
        "정원, 잔디밭, 벤치, 한적한 산책로 등 여유로운 공간",
        "차분한 실내/야외에서 휴식을 즐기는 장면",
        "온천탕이나 마사지 체어, 카페 소파처럼 정적인 휴식 장면",
        "혼자 또는 소규모가 조용히 쉬는 장면",
        "A relaxing or healing place: spa, hot spring, cozy café, tranquil garden or walkway, calm resting activity."
    ],
}

TEMPERATURE = 0.9 # softmax 값이 낮아지면 한 테마로 쏠림
MULTILABEL_SCALING = "minmax" #배치 내 유사도를 0-1로 선형 정규화 -> 여러 테마 동시 강도를 보기 쉬움

def is_url(s:str) -> bool: #문자열이 http,https로 시작하는지 확인 
    return isinstance(s,str) and (s.startswith("http://") or s.startswith("https://"))

def l2_normalize(x, dim=-1, eps=1e-12):
    # 벡터 정규화
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def to_device(batch, device, non_blocking: bool = False):
    # [FIX] 텐서를 지정된 장치(CPU/GPU)로 이동 (비동기 전송 지원)
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}

# ===== 보강: 세션 기반 안정적 다운로드 + 재시도 + 용량 상한 + EXIF 회전 보정 =====
_session = requests.Session()
_session.headers.update({
    "User-Agent": "alpha-clip/1.0 (requests)",
    "Accept": "image/*,*/*;q=0.8",
    "Connection": "keep-alive",
})

# [ADD] (선택) HTTPAdapter로 커넥션 풀/표준 재시도 구성
try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _retry = Retry(total=MAX_RETRIES, backoff_factor=0.2,
                   status_forcelist=[429, 500, 502, 503, 504])
    _adapter = HTTPAdapter(pool_connections=DOWNLOAD_WORKERS,
                           pool_maxsize=DOWNLOAD_WORKERS,
                           max_retries=_retry)
    _session.mount("http://", _adapter)
    _session.mount("https://", _adapter)
except Exception:
    pass  # 표준 세션만으로도 동작

def _backoff_sleep(attempt:int):
    # 지수 백오프 (너무 길지 않게 가벼운 계수)
    time.sleep((BACKOFF_BASE ** attempt) * 0.5)

# [ADD] TurboJPEG 빠른 디코딩 경로 로딩
USE_TURBOJPEG = False
try:
    from turbojpeg import TurboJPEG, TJPF_RGB, TJFLAG_FASTDCT
    _jpeg = TurboJPEG()
    USE_TURBOJPEG = True
except Exception:
    USE_TURBOJPEG = False

def _is_probably_jpeg(url: str, content_type: str) -> bool:
    # [ADD] 헤더/확장자 기반 JPEG 추정
    ct = (content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct:
        return True
    lower = (url or "").lower()
    return lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".jfif")

def _open_image_from_bytes(data: bytes, content_type: str, url: str) -> Image.Image:
    """
    [ADD] JPEG이면 TurboJPEG 빠른 경로 우선, 아니면 PIL로 폴백.
    EXIF 회전 보정은 PIL 경로에서만 기본 적용.
    """
    if USE_TURBOJPEG and _is_probably_jpeg(url, content_type):
        try:
            # Ultra-fast JPEG decode (no EXIF orientation by default)
            from turbojpeg import TJPF_RGB, TJFLAG_FASTDCT
            arr = _jpeg.decode(data, pixel_format=TJPF_RGB, flags=TJFLAG_FASTDCT)
            img = Image.fromarray(arr, mode="RGB")
            if TURBOJPEG_EXIF_TRANSPOSE:
                # 속도가 중요하지 않을 때만 사용 권장
                try:
                    # EXIF Orientation만 헤더만 읽어 판단하려면 PIL을 잠깐 열어 메타만 확인
                    # (완전 디코딩까지는 가지 않음)
                    with Image.open(BytesIO(data)) as im_meta:
                        img = ImageOps.exif_transpose(img, im_meta.getexif())
                except Exception:
                    pass
            return img
        except Exception:
            # 실패 시 PIL로 폴백
            pass
    # PIL 경로 (EXIF 회전 보정 + RGB)
    img = Image.open(BytesIO(data))
    try:
        if "exif" in img.info:
            img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")

def load_image(src: str, timeout=8.0, max_retries=2) -> Image.Image:
    """
    src가 URL이면 requests로 다운로드 → BytesIO 버퍼로 감싸서 PIL/TurboJPEG로 오픈
    """
    last_err = None
    if is_url(src):
        retries = max(max_retries, MAX_RETRIES)
        for attempt in range(retries + 1):
            try:
                # stream=True로 용량 상한 직접 체크
                with _session.get(src, timeout=timeout, stream=True) as r:
                    r.raise_for_status()
                    cl = r.headers.get("Content-Length")
                    if cl and cl.isdigit() and int(cl) > MAX_IMAGE_BYTES:
                        raise ValueError(f"TooLarge: Content-Length={cl}")

                    data_io = BytesIO()
                    size = 0
                    for chunk in r.iter_content(8192):
                        if chunk:
                            size += len(chunk)
                            if size > MAX_IMAGE_BYTES:
                                raise ValueError(f"TooLarge: read>{MAX_IMAGE_BYTES}")
                            data_io.write(chunk)
                    data_io.seek(0)
                    # [MOD] 여기서 TurboJPEG 빠른 경로 우선 적용
                    raw = data_io.getvalue()
                    ctype = r.headers.get("Content-Type", "")
                    img = _open_image_from_bytes(raw, ctype, src)
                    return img
            except Exception as e:
                last_err = e
                if attempt < retries:
                    _backoff_sleep(attempt)
        raise last_err
    else:
        # URL 형식이 아니면 오류 발생 (로컬 파일 사용 안 함)
        raise ValueError(f"URL 형식이 아닙니다: {src}") 
    
def load_image_safe(item):
    try:
        img = load_image(item["src"], timeout=HTTP_TIMEOUT, max_retries=MAX_RETRIES)
        return img, (item["place"], item["src"], item["row_idx"]), None
    except Exception as e:
        return None, (item.get("place"), item.get("src"), item.get("row_idx")), str(e)

def load_images_parallel(items, max_workers=DOWNLOAD_WORKERS):
    imgs, metas, errs = [], [], []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for img, meta, err in ex.map(load_image_safe, items):
            if img is not None:
                imgs.append(img)
                metas.append(meta)
            else:
                place, src, row_idx = meta
                errs.append({"place": place, "source": src, "error": err})
    return imgs, metas, errs

def minmax_per_row(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """각 행(이미지)의 테마 유사도를 0~1로 정규화"""
    tmin = t.min(dim=1, keepdim=True).values
    tmax = t.max(dim=1, keepdim=True).values
    return (t - tmin) / (tmax - tmin + eps)

def iter_batches(items, n):
    # 리스트를 n 크기의 배치로 나누는 제너레이터
    for i in range(0, len(items), n):
        yield items[i:i+n]

processor = CLIPProcessor.from_pretrained(MODEL_NAME) #이미지 전처리+텍스트 토큰화
model     = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE) #이미지/텍스트 인코더
model.eval() #eval() 추론모드

##이미지 임베딩 추출

theme_names=list(THEMES.keys())

with torch.no_grad(): #텍스트 임베딩하기, 여러 문장으로 평균을 내서 정확도 높이기
    text_feats = []
    for theme in theme_names:
        prompts = THEMES[theme]  # 테마별 여러 문장(한글+영문)
        inputs = processor(text=prompts, images=None, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.pin_memory() for k, v in inputs.items()}  # [FIX] pinned memory
        inputs = to_device(inputs, DEVICE, non_blocking=True)        # [FIX] 비동기 전송

        # GPU면 FP16 가속 (CPU면 자동 비활성)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_FP16):
            t = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        t = l2_normalize(t, dim=-1)        # 문장별 임베딩 정규화
        t_mean = l2_normalize(t.mean(dim=0), -1)  # 앙상블 평균 → 다시 정규화
        text_feats.append(t_mean)

    TEXT_EMB   = torch.stack(text_feats, dim=0)   # [num_themes, D] (테마 임베딩)
    TEXT_EMB_T = TEXT_EMB.t().contiguous()        # 행렬곱 최적화용 전치본

## 이미지 수집 (CSV 로직만 유지)
df_out = None
sources = []

if USE_CSV:
    try:
        df_src = pd.read_csv(INPUT_CSV)
        assert TITLE_COL in df_src.columns and URL_COL in df_src.columns, \
            f"CSV에 '{TITLE_COL}', '{URL_COL}' 컬럼이 필요합니다. 현재: {list(df_src.columns)}"
        
        # 원본을 보존한 상태에서 결과 컬럼 생성 (이곳에 점수를 직접 기록)
        df_out = df_src.copy()
        # >>> 변경: 개별 score_* 컬럼 대신, 단일 문자열 컬럼 생성
        df_out["theme_scores"] = pd.Series(dtype="object")

        # 원본 인덱스를 'row_idx'로 저장하여 df_out에 점수 기록 시 사용
        df_src = df_src.dropna(subset=[TITLE_COL, URL_COL]).reset_index(drop=False).rename(columns={"index":"row_idx"})
        
        for _, row in df_src.iterrows():
            # 문자열 공백 제거
            place = str(row[TITLE_COL]).strip()   
            url   = str(row[URL_COL]).strip()
            idx   = int(row["row_idx"])
            if place and is_url(url): # 장소명이 있고 URL 형식이어야 처리
                sources.append({"place": place, "src": url, "row_idx": idx})
    except Exception as e:
        print(f"[FATAL] CSV 파일 로딩/준비 중 오류 발생: {e}")
        raise SystemExit

if not sources:
    print("[WARN] 평가할 이미지가 없습니다. (CSV 파일 확인)")
    raise SystemExit

## 배치 추론
error_rows=[] # 실패 목록 

# 보강: list(iter_batches(...))로 전체를 메모리에 올리지 않고, 직접 제너레이터를 tqdm에 넣음
total_batches = (len(sources)+BATCH_SIZE-1)//BATCH_SIZE
batches_done = 0

for batch in tqdm(iter_batches(sources, BATCH_SIZE),  # sources 리스트에서 이미지 정보를 전부 가져옴, iter_batches()로 32장씩 끊어서 묶음 
                  desc="Scoring", total=total_batches): # tqdm 각 묶음을 하나씩 처리 -> 진행률이 보임
    
    # 이미지 로딩 (병렬)
    imgs, metas, errs = load_images_parallel(batch, max_workers=DOWNLOAD_WORKERS)
    if errs:
        error_rows.extend(errs)

    if not imgs:  # 이미지 배치 로딩 실패 시 스킵
        batches_done += 1
        # 보강: 주기적 체크포인트 저장 (중도 종료 대비)
        if CHECKPOINT_EVERY > 0 and (batches_done % CHECKPOINT_EVERY == 0):
            df_out.to_csv(OUT_MERGED_CSV, index=False, encoding="utf-8-sig")
        continue

    with torch.inference_mode():
        img_inputs = processor(images=imgs, return_tensors="pt")
        if torch.cuda.is_available():
            img_inputs = {k: v.pin_memory() for k, v in img_inputs.items()}  # [FIX] pinned memory
        img_inputs = to_device(img_inputs, DEVICE, non_blocking=True)        # [FIX] 비동기 전송

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_FP16):
            img_feat = model.get_image_features(**img_inputs)

        img_feat = l2_normalize(img_feat, dim=-1)
        sim = img_feat @ TEXT_EMB_T
        
    # 스케일링 (minmax 사용)
    if MULTILABEL_SCALING.lower() == "softmax": # 유사도 점수를 최정적인 확률이나 정규화된 점수로 변환 
        probs = F.softmax(sim/TEMPERATURE, dim=1) 
    elif MULTILABEL_SCALING.lower() == "sigmoid":
        probs=torch.sigmoid(sim) 
    else:  # minmax (기본)
        probs = minmax_per_row(sim) # 유사도 점수를 0과 1사이로 선형 변환 이거 사용

    # >>> 변경: per-theme 컬럼 대신, '테마:점수'를 하나의 문자열로 합쳐 기록
    # [MOD] 행 단위 at[] 대신 배치 단위 일괄 반영으로 pandas 오버헤드 감소
    idxs, values = [], []
    for (place, src, row_idx), row_vals in zip(metas, probs.tolist()):
        theme_score_str = ", ".join(
            f"{name}:{float(v):.2f}" for name, v in zip(theme_names, row_vals)
        )
        if USE_CSV and row_idx is not None:
            idxs.append(row_idx)
            values.append(theme_score_str)
    if USE_CSV and idxs:
        df_out.loc[idxs, "theme_scores"] = values

    # 보강: 배치 단위 체크포인트 저장
    batches_done += 1
    if CHECKPOINT_EVERY > 0 and (batches_done % CHECKPOINT_EVERY == 0):
        df_out.to_csv(OUT_MERGED_CSV, index=False, encoding="utf-8-sig")

# 저장 
if USE_CSV and df_out is not None: 
    df_out.to_csv(OUT_MERGED_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] 최종 결과 저장 완료: {OUT_MERGED_CSV} (원본 CSV에 점수 병합)")
else:
    print("[WARN] CSV 모드가 활성화되지 않았거나 처리된 데이터가 없습니다.")

if error_rows:
    pd.DataFrame(error_rows).to_csv(OUT_ERRORS_CSV, index=False, encoding="utf-8-sig")
    print((f"[INFO] 로딩 실패 {len(error_rows)}건 → {OUT_ERRORS_CSV}"))

end = time.perf_counter()
print(f"실행 시간: {end - start:.6f}초")
