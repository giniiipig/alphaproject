# src/config.py
from pathlib import Path

# 프로젝트 루트 (이 파일 기준 2단계 위 디렉토리)
ROOT = Path(__file__).resolve().parents[1]

# 주요 디렉토리 경로 정의
DATA_DIR    = ROOT / "data"         # 데이터 전체 루트
RAW_DIR     = DATA_DIR / "raw"      # 원천 데이터 (TourAPI 수집 원본 등)
PROC_DIR    = DATA_DIR / "processed"  # 전처리/스코어링 결과물 저장
LABEL_DIR   = DATA_DIR / "labels"   # 학습 라벨 관련 (auto/human/merged)
MODELS_DIR  = ROOT / "models"       # 학습된 모델 저장 위치

# 주요 파일 경로
RAW_CSV             = RAW_DIR  / "tour_raw_seoul.csv"         # TourAPI 원본 데이터
RAW_WITH_TEXTS_CSV  = RAW_DIR  / "tour_raw_with_texts.csv"    # 임베딩 텍스트 포함 버전
SCORED_CSV          = PROC_DIR / "tour_scored.csv"            # 점수화된 결과물

TRAIN_LABEL_AUTO    = LABEL_DIR / "train_labels_human_input.csv"  # 학습 데이터 기본값 (사람 보정 라벨)
MODEL_PATH          = MODELS_DIR / "tour_theme_regressor.joblib"  # Sklearn 학습 모델 저장 위치
LABEL_AUTO = TRAIN_LABEL_AUTO  # 호환성 유지용 변수

# 사용자 선호 점수 샤프닝 옵션
PREF_SHARPEN_ENABLE = True   # True → 샤프닝 적용
PREF_SHARPEN_GAMMA  = 1.6    # 샤프닝 강도 (1.0이면 효과 없음)

# 학습/예측 모드 선택
USE_TORCH = False   # True → Torch 모델 사용, False → Sklearn 모델 사용

# 디렉토리 자동 생성 (없으면 새로 만듦)
for d in [DATA_DIR, RAW_DIR, PROC_DIR, LABEL_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
