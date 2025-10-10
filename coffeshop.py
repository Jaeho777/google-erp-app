# ==============================================================
# ☕ Coffee ERP Dashboard — Company Showcase Edition (Tone-Down Blue)
# ==============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------
# 0️⃣ Firebase 초기화
# ----------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("/Users/iseojin/Desktop/google-erp-app/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()
collection_name = "coffee_sales"

# ----------------------
# 1️⃣ 기본 스타일 설정
# ----------------------
st.set_page_config(page_title="☕ Coffee ERP Dashboard", layout="wide")

pio.templates.default = "plotly_white"
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Blues"

st.markdown("""
    <style>
    /* 전체 폰트 및 배경 */
    .stApp {
        background-color: #F4F6FA;
        font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
    }

    /* 상단 헤더 */
    .dashboard-header {
        display: flex;
        align-items: center;
        gap: 12px;
        background-color: #1E2A38;
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        margin-bottom: 25px;
    }

    .dashboard-header h1 {
        font-size: 1.8em;
        margin: 0;
    }

    /* 메트릭 카드 */
    .metric-card {
        background-color: white;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }
    .metric-title {
        color: #7C8DA6;
        font-size: 1em;
    }
    .metric-value {
        color: #2C3E50;
        font-size: 1.8em;
        font-weight: 600;
    }

    /* 사이드바 - 수정됨 */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;  /* 배경: 흰색 */
        color: #0D3B66 !important;             /* 글자색: 딥 블루 */
    }
    div[data-testid="stSidebarNav"] {
        color: #0D3B66 !important;
    }
    .css-1d391kg p, .css-1v3fvcr, .css-qrbaxs {
        color: #0D3B66 !important;
    }

    /* 라디오 버튼 텍스트 색상 */
    label[data-baseweb="radio"] div {
        color: #0D3B66 !important;
        font-weight: 500;
    }

    /* Plotly 배경 */
    .js-plotly-plot .plotly {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-header">
    <h1>☕ Coffee ERP Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# ----------------------
# 2️⃣ CSV 데이터 불러오기
# ----------------------
@st.cache_data(ttl=0)
def load_csv():
    df = pd.read_csv("/Users/iseojin/Desktop/google-erp-app/Coffee Shop Sales.csv")

    df = df.rename(columns={
        'transaction_id': '거래번호',
        'transaction_date': '날짜',
        'transaction_time': '시간',
        'transaction_qty': '수량',
        'store_id': '가게ID',
        'store_location': '가게위치',
        'product_id': '상품ID',
        'unit_price': '단가',
        'product_category': '상품카테고리',
        'product_type': '상품타입',
        'product_detail': '상품상세',
        'Revenue': '수익'
    })

    df['수익'] = df['수익'].astype(str).str.replace('[$,]', '', regex=True).astype(float)
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df['시'] = pd.to_datetime(df['시간'], format='%H:%M:%S', errors='coerce').dt.hour
    df['요일'] = df['날짜'].dt.day_name()
    df['월'] = df['날짜'].dt.month

    return df

df_csv = load_csv()

# ----------------------
# 3️⃣ Firebase 데이터 불러오기
# ----------------------
docs = db.collection(collection_name).stream()
firebase_data = [doc.to_dict() for doc in docs]
df_fb = pd.DataFrame(firebase_data)

if not df_fb.empty:
    df_fb['날짜'] = pd.to_datetime(df_fb['날짜'], errors='coerce')
    df_fb['수익'] = df_fb['수익'].astype(float)
    df_fb['시'] = pd.to_datetime(df_fb.get('시간', pd.Series([None]*len(df_fb))), format='%H:%M:%S', errors='coerce').dt.hour
    df_fb['요일'] = df_fb['날짜'].dt.day_name()
    df_fb['월'] = df_fb['날짜'].dt.month

df = pd.concat([df_csv, df_fb], ignore_index=True)

# ----------------------
# 4️⃣ 사이드바 메뉴
# ----------------------
menu = st.sidebar.radio(
    " 메뉴 선택",
    ["경영 현황", "매출 대시보드", "기간별 분석", "거래 추가", "거래 내역"]
)

# ----------------------
# 5️⃣ 거래 추가
# ----------------------
if menu == "거래 추가":
    st.header(" 거래 데이터 추가")

    category_options = df['상품카테고리'].dropna().unique().tolist()
    type_options = df['상품타입'].dropna().unique().tolist()
    detail_options = df['상품상세'].dropna().unique().tolist()

    with st.form("add_transaction"):
        col1, col2 = st.columns(2)
        with col1:
            날짜 = st.date_input("날짜")
            상품카테고리 = st.selectbox("상품카테고리", category_options)
            상품타입 = st.selectbox("상품타입", type_options)
        with col2:
            상품상세 = st.selectbox("상품상세", detail_options)
            수량 = st.number_input("수량", min_value=1, value=1)
            단가 = st.number_input("단가", min_value=0.0, value=1000.0, step=100.0)
        
        수익 = 수량 * 단가
        st.markdown(f"### 💰 계산된 수익: **{수익:,.0f} 원**")

        submitted = st.form_submit_button("데이터 추가")
        if submitted:
            new_doc = {
                "날짜": str(날짜),
                "상품카테고리": 상품카테고리,
                "상품타입": 상품타입,
                "상품상세": 상품상세,
                "수량": 수량,
                "단가": 단가,
                "수익": 수익
            }
            db.collection(collection_name).add(new_doc)
            st.success("✅ 거래 데이터가 Firebase에 저장되었습니다!")
            st.balloons()

# ----------------------
# 6️⃣ 경영 현황
# ----------------------
elif menu == "경영 현황":
    st.header(" 경영 현황 요약")

    col1, col2, col3 = st.columns(3)
    total_rev = df['수익'].sum()
    total_tx = df['거래번호'].nunique()
    total_qty = df['수량'].sum()

    col1.markdown(f"<div class='metric-card'><p class='metric-title'>총 매출액</p><p class='metric-value'>{total_rev:,.0f} 원</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><p class='metric-title'>총 거래 수</p><p class='metric-value'>{total_tx:,} 건</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><p class='metric-title'>총 판매 수량</p><p class='metric-value'>{total_qty:,} 개</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    top_cat = df.groupby('상품카테고리')['수익'].sum().sort_values(ascending=False).head(1)
    top_prod = df.groupby('상품타입')['수익'].sum().sort_values(ascending=False).head(1)
    st.info(f"🏆 가장 매출 높은 카테고리: **{top_cat.index[0]}** ({top_cat.iloc[0]:,.0f} 원) / 상품: **{top_prod.index[0]}**")

    col4, col5 = st.columns(2)
    with col4:
        cat = df.groupby('상품카테고리')['수익'].sum().reset_index()
        fig_cat = px.pie(cat, values='수익', names='상품카테고리', title="카테고리별 매출 비중")
        st.plotly_chart(fig_cat, use_container_width=True)
    with col5:
        daily = df.groupby('날짜')['수익'].sum().reset_index()
        fig_trend = px.line(daily, x='날짜', y='수익', title="일자별 매출 추이")
        st.plotly_chart(fig_trend, use_container_width=True)

# ----------------------
# 7️⃣ 매출 대시보드
# ----------------------
elif menu == "매출 대시보드":
    st.header(" 매출 대시보드")

    col1, col2 = st.columns(2)
    monthly = df.groupby(df['날짜'].dt.to_period("M"))['수익'].sum().reset_index()
    monthly['날짜'] = monthly['날짜'].dt.to_timestamp()

    with col1:
        fig_month = px.bar(monthly, x='날짜', y='수익', title="월별 매출")
        st.plotly_chart(fig_month, use_container_width=True)

    with col2:
        cat_sales = df.groupby('상품카테고리')['수익'].sum().reset_index()
        fig_cat2 = px.bar(cat_sales, x='상품카테고리', y='수익', title="상품 카테고리별 매출")
        st.plotly_chart(fig_cat2, use_container_width=True)

    prod_sales = df.groupby(['상품타입','상품상세'])['수익'].sum().reset_index()
    fig_sun = px.sunburst(prod_sales, path=['상품타입','상품상세'], values='수익', title="상품 구조별 매출")
    st.plotly_chart(fig_sun, use_container_width=True)

# ----------------------
# 8️⃣ 기간별 분석
# ----------------------
elif menu == "기간별 분석":
    st.header(" 기간별 매출 분석")

    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df_week = df.groupby('요일')['수익'].sum().reindex(weekday_order).reset_index()
    df_hour = df.groupby('시')['수익'].sum().reset_index()
    df_month = df.groupby('월')['수익'].sum().reset_index()

    col1, col2, col3 = st.columns(3)
    top_day = df_week.loc[df_week['수익'].idxmax()]
    top_hour = df_hour.loc[df_hour['수익'].idxmax()]
    top_month = df_month.loc[df_month['수익'].idxmax()]

    col1.markdown(f"<div class='metric-card'><p class='metric-title'>최고 매출 요일</p><p class='metric-value'>{top_day['요일']}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><p class='metric-title'>최고 매출 시간</p><p class='metric-value'>{int(top_hour['시'])}시</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><p class='metric-title'>최고 매출 달</p><p class='metric-value'>{int(top_month['월'])}월</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        fig_w = px.bar(df_week, x='요일', y='수익', title="요일별 매출")
        st.plotly_chart(fig_w, use_container_width=True)
    with colB:
        fig_h = px.line(df_hour, x='시', y='수익', title="시간대별 매출")
        st.plotly_chart(fig_h, use_container_width=True)
    fig_m = px.bar(df_month, x='월', y='수익', title="월별 매출")
    st.plotly_chart(fig_m, use_container_width=True)

# ----------------------
# 9️⃣ 거래 내역
# ----------------------
else:
    st.header(" 전체 거래 내역")
    st.dataframe(df[['날짜','상품카테고리','상품타입','상품상세','수량','수익','요일','시']])
