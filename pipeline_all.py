import pandas as pd

from src.data_fetch.fetch_tourapi import fetch_items_to_df
from src.preprocessing.build_embedding_texts import add_embedding_texts
# from src.preprocessing.score_and_export import score_and_save
from src.preprocessing.score_and_export import score_and_save
from src.training.train_regressor import main as train_main
from src.config import RAW_CSV, RAW_WITH_TEXTS_CSV, SCORED_CSV

def run_pipeline():
    print("📌 실행 옵션을 선택하세요:")
    print("  1) TourAPI에서 새로 데이터 수집")
    print("  2) 기존 데이터에서 학습/점수화만 실행")
    while True:
        try:
            choice = int(input("👉 선택 (1 또는 2): "))
            if choice in (1, 2):
                break
            else:
                print("❌ 잘못된 입력입니다. 1 또는 2을 입력하세요.")
        except ValueError:
            print("❌ 숫자를 입력하세요 (1 또는 2).")

    check = choice

    if check == 1:
        print("📌 Step 1: TourAPI 수집")
        df_raw = fetch_items_to_df(area_code=1, max_pages=3, rows_per_page=100, save_csv=str(RAW_CSV))
        print(f"✅ 저장: {RAW_CSV}")

        print("📌 Step 2: 임베딩 텍스트 생성")
        df_texts = add_embedding_texts(df_raw, qps_delay=0.05, save_csv=str(RAW_WITH_TEXTS_CSV))
    else:
        print("📌 Step 2: 기존 임베딩 텍스트 불러오기")
        df_texts = pd.read_csv(RAW_WITH_TEXTS_CSV)
    print("📌 Step 3: 학습/재학습")
    train_main()

    print("📌 Step 4: 점수화")
    df_scored = score_and_save(df_texts, out_csv=str(SCORED_CSV))

    return df_scored

if __name__ == "__main__":
    final = run_pipeline()
    print(final.head())

