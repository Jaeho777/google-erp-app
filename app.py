import streamlit as st
from google.cloud import bigquery
import plotly.express as px
import pandas as pd

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/iseojin/Desktop/google-erp-app/keys/service-account.json"


# ✅ 서비스 계정 인증
client = bigquery.Client(location="US")

# ✅ 프로젝트/데이터셋/테이블 정보
PROJECT_ID = "noted-branch-473510-r5"
DATASET = "erp_dataset"

# 원본 매출 테이블 (스마트스토어 데이터)
SOURCE_TABLE = f"{PROJECT_ID}.{DATASET}.purchase_history1"

# 입력용 테이블 (직접 입력 저장)
INPUT_TABLE = f"{PROJECT_ID}.{DATASET}.purchase_input"

# ✅ 페이지 기본 설정
st.set_page_config(page_title="SnooCat ERP", layout="wide")
st.sidebar.title("📂 메뉴")

menu = st.sidebar.radio("이동", ["판매 입력", "대시보드", "데이터 관리"])

# ---------------- 판매 입력 ----------------
if menu == "판매 입력":
    st.header("📝 판매 데이터 입력")

    date = st.date_input("날짜")
    product = st.selectbox("상품", ["피스타치오", "다크 헤이즐넛"])
    qty = st.number_input("판매 수량", min_value=1, step=1)
    revenue = st.number_input("매출 금액", min_value=1000, step=100)

    if st.button("저장"):
        row = {
            "date": str(date),
            "product": product,
            "qty": qty,
            "revenue": revenue
        }
        # 입력용 테이블에 저장
        errors = client.insert_rows_json(INPUT_TABLE, [row])
        if not errors:
            st.success("✅ 저장 완료!")
        else:
            st.error(f"저장 실패: {errors}")

# ---------------- 대시보드 ----------------
elif menu == "대시보드":
    st.header("📊 매출 대시보드")

    # 원본 테이블 + 입력 테이블 합치기
    query = f"""
SELECT 
      DATE(SAFE.PARSE_DATETIME('%Y.%m.%d %H:%M', CAST(`구매확정일` AS STRING))) AS date,
      `상품명` AS product,
      CAST(`수량` AS INT64) AS qty,
      CAST(REGEXP_REPLACE(`최종 상품별 총 주문금액`, '[^0-9]','') AS INT64) AS revenue
    FROM `{SOURCE_TABLE}`

    UNION ALL

    SELECT 
      COALESCE(
        SAFE.PARSE_DATE('%Y-%m-%d', CAST(`date` AS STRING)),   -- 문자열 "YYYY-MM-DD"
        DATE(SAFE_CAST(`date` AS TIMESTAMP), 'Asia/Seoul'),    -- TIMESTAMP일 때 KST 기준
        SAFE_CAST(`date` AS DATE)                              -- 이미 DATE면 그대로
      ) AS date,
      product,
      CAST(qty AS INT64),
      CAST(revenue AS INT64)
    FROM `{INPUT_TABLE}`


"""




    df = client.query(query).to_dataframe()

    if df.empty:
        st.warning("⚠️ 아직 데이터가 없습니다.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.line(df, x="date", y="revenue", color="product", title="📈 날짜별 매출 추이")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(df, x="product", y="revenue", title="🛒 상품별 총 매출 비교")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📋 데이터 테이블")
        st.dataframe(df)

# ---------------- 데이터 관리 ----------------
elif menu == "데이터 관리":
    st.header("🛠 데이터 관리 (수정/삭제)")

    query = f"SELECT * FROM `{INPUT_TABLE}` ORDER BY date DESC"
    df_input = client.query(query).to_dataframe().copy(deep=True)

    if df_input.empty:
        st.info("아직 입력된 데이터가 없습니다.")
    else:
        st.dataframe(df_input)

        row_index = st.number_input("수정/삭제할 행 번호 (0부터 시작)", min_value=0, max_value=len(df_input)-1, step=1)
        row = df_input.iloc[row_index]
        st.write("선택된 행:", row.to_dict())

        # 수정하기
        with st.form("edit_form"):
            new_date = st.date_input("날짜", value=pd.to_datetime(row["date"]))
            new_product = st.selectbox("상품", ["피스타치오", "다크 헤이즐넛"], index=0 if row["product"]=="피스타치오" else 1)
            new_qty = st.number_input("수량", min_value=1, value=int(row["qty"]))
            new_rev = st.number_input("매출", min_value=1000, value=int(row["revenue"]))

            submitted = st.form_submit_button("수정 저장")
            if submitted:
                df_input.at[row_index, "date"] = str(new_date)
                df_input.at[row_index, "product"] = new_product
                df_input.at[row_index, "qty"] = new_qty
                df_input.at[row_index, "revenue"] = new_rev

                job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
                client.load_table_from_dataframe(df_input, INPUT_TABLE, job_config=job_config).result()
                st.success("✅ 수정 완료! (테이블 덮어쓰기 완료)")

        # 삭제하기
        if st.button("🗑 삭제"):
            df_input = df_input.drop(row_index).reset_index(drop=True)
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            client.load_table_from_dataframe(df_input, INPUT_TABLE, job_config=job_config).result()
            st.warning("❌ 삭제 완료! (테이블 덮어쓰기 완료)")
