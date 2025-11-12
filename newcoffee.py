# =============================================================
# ☕ Coffee ERP Dashboard — Company Showcase Edition (Tone-Down Blue)
#  - 재고 관리(자동 차감/임계치 경고/자동 발주 시뮬레이션)
#  - UI 한글화(이름 매핑 + 요일 한글 표시)
#  - 원본/Firestore는 영어 저장, 화면은 한글 표시(정/역매핑)
#  - 데이터 편집(거래 수정/삭제 + 재고 일괄수정)
#  - 도움말 탭 + SKU 파라미터(리드타임/세이프티/목표일수/레시피g) + ROP 지표/권장발주
#  - NEW: 레시피(BOM) 기반 자동 차감, uom(단위) 지원, 실사/오차율, 발주 ±범위 표시
# =============================================================

import os
import json
import re
import warnings
from math import ceil
from pathlib import Path
from datetime import datetime
import time # === [AI 기능 추가] === (Mock 응답용)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

import firebase_admin
from firebase_admin import credentials, firestore

# === [AI 기능 추가] ===
# SPRINT 1 (AI 비서) 및 SPRINT 2 (수요 예측) 라이브러리
try:
    import openai
    from prophet import Prophet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
except ImportError:
    st.error("""
    AI/ML 기능을 위한 라이브러리가 부족합니다.
    터미널에서 'pip install openai prophet scikit-learn'를 실행해주세요.
    """)
    st.stop()
# === [AI 기능 추가] ===


st.set_page_config(page_title="☕ Coffee ERP Dashboard", layout="wide")


# === [AI 기능 추가] ===
# SPRINT 1: OpenAI API 키 설정
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except (KeyError, AttributeError):
    st.warning("""
    OpenAI API 키가 'secrets.toml'에 설정되지 않았습니다. 
    AI 비서 기능이 작동하지 않거나 Mock 데이터로 작동합니다.
    [.streamlit/secrets.toml] 파일에 [openai] api_key = "sk-..."를 추가하세요.
    """)
    openai.api_key = None # 키가 없어도 앱이 멈추지 않도록
# === [AI 기능 추가] ===


def init_firebase():
    try:
        if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
            cred_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            cred = credentials.Certificate(cred_info)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            return
    except json.JSONDecodeError:
        st.error("환경 변수 'GOOGLE_APPLICATION_CREDENTIALS_JSON'의 형식이 올바르지 않습니다.")
        return
    except Exception as e:
        st.warning(f"환경 변수로 Firebase 초기화 실패: {e}. 'secrets.toml'을 시도합니다.")

    # Fallback to secrets.toml if env var fails or is not present
    try:
        if "firebase" in st.secrets:
            cred_info = dict(st.secrets["firebase"])
            cred = credentials.Certificate(cred_info)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
        else:
            st.error("Firebase 인증 정보가 'secrets.toml'에 없습니다.")
    except Exception as e:
        st.error(f"Firebase 초기화 중 심각한 오류 발생: {e}")

# ... (기존 함수들: init_firebase, safe_rerun) ...
init_firebase()

def safe_rerun():
    """Reruns the script safely, handling potential Streamlit errors."""
    try:
        st.rerun()
    except st.errors.StreamlitAPIException as e:
        if "Rerun outside main thread" in str(e):
            print("Ignoring rerun error in non-main thread.")
        else:
            raise e

# ==============================================================
# 데이터 로딩 (캐시)
# ==============================================================
@st.cache_data(ttl=600)
def load_data_from_fs():
    """Firestore에서 모든 컬렉션 데이터를 로드합니다."""
    db = firestore.client()
    
    collections = ["coffee_sales", "inventory", "stock_moves", "recipes", "sku_params"]
    data = {}
    
    for coll in collections:
        try:
            docs = db.collection(coll).stream()
            data[coll] = [doc.to_dict() for doc in docs]
        except Exception as e:
            st.error(f"'{coll}' 컬렉션 로드 실패: {e}")
            data[coll] = [] # 실패해도 빈 리스트로 계속
            
    return data

@st.cache_data(ttl=600)
def process_dataframes(data):
    """로드된 데이터를 Pandas DataFrame으로 변환하고 전처리합니다."""
    
    # 1. Sales (거래)
    df = pd.DataFrame(data.get("coffee_sales", []))
    if not df.empty and 'transaction_created_at' in df.columns:
        df['날짜'] = pd.to_datetime(df['transaction_created_at'])
        df['요일'] = df['날짜'].dt.day_name()
        df['시'] = df['날짜'].dt.hour
        df['수익'] = df['unit_price'] * df['quantity']
        df.rename(columns={
            "product_detail": "상품상세", 
            "quantity": "수량", 
            "unit_price": "단가",
            "product_category": "상품카테고리",
            "product_type": "상품타입"
        }, inplace=True)
        # 요일 한글화
        day_map = {'Monday': '월', 'Tuesday': '화', 'Wednesday': '수', 'Thursday': '목', 'Friday': '금', 'Saturday': '토', 'Sunday': '일'}
        df['요일'] = df['요일'].map(day_map)
    else:
        st.warning("판매 데이터가 비어있거나 'transaction_created_at' 컬럼이 없습니다.")
        # 빈 데이터프레임이라도 기본 컬럼 정의
        df = pd.DataFrame(columns=['날짜', '상품상세', '수량', '단가', '수익', '요일', '시', '상품카테고리', '상품타입'])

    # 2. Inventory (재고)
    df_inv = pd.DataFrame(data.get("inventory", []))
    if not df_inv.empty:
        df_inv = df_inv.rename(columns={"stock": "현재재고"})

    # 3. Stock Moves (재고 이동)
    df_moves = pd.DataFrame(data.get("stock_moves", []))
    
    # 4. Recipes (레시피)
    recipes = {item['sku_en']: item['ingredients'] for item in data.get("recipes", []) if 'sku_en' in item}

    # 5. SKU Params (품목 속성)
    df_params = pd.DataFrame(data.get("sku_params", []))

    return df, df_inv, df_moves, recipes, df_params

# ==============================================================
# 헬퍼 함수
# ==============================================================
def format_krw(val):
    """숫자를 원화 형식의 문자열로 변환합니다 (예: 1,000원)."""
    if pd.isna(val) or val is None:
        return "0원"
    return f"{int(val):,}원"

def format_g(val):
    """숫자를 그램(g) 형식의 문자열로 변환합니다."""
    if pd.isna(val) or val is None:
        return "0g"
    return f"{val:,.1f}g"

def to_korean_detail(sku_en):
    """영문 SKU를 한글 상품명으로 변환 (간이 매핑)"""
    mapping = {
        "americano": "아메리카노", "latte": "라떼", "cappuccino": "카푸치노",
        "espresso": "에스프레소", "mocha": "카페모카", "cold_brew": "콜드브루",
        "coffee_bean_a": "원두 A (블렌드)", "coffee_bean_b": "원두 B (싱글)",
        "milk": "우유 (1L)", "syrup_vanilla": "바닐라 시럽", "syrup_caramel": "카라멜 시럽",
        "cup_holder": "컵 홀더", "straw": "빨대", "cup_12oz": "12oz 컵"
    }
    return mapping.get(sku_en, sku_en) # 매핑에 없으면 영문명 그대로 반환

def from_korean_detail(name_kr):
    """한글 상품명을 영문 SKU로 변환 (간이 매핑)"""
    reverse_mapping = {v: k for k, v in {
        "americano": "아메리카노", "latte": "라떼", "cappuccino": "카푸치노",
        "espresso": "에스프레소", "mocha": "카페모카", "cold_brew": "콜드브루",
        "coffee_bean_a": "원두 A (블렌드)", "coffee_bean_b": "원두 B (싱글)",
        "milk": "우유 (1L)", "syrup_vanilla": "바닐라 시럽", "syrup_caramel": "카라멜 시럽",
        "cup_holder": "컵 홀더", "straw": "빨대", "cup_12oz": "12oz 컵"
    }.items()}
    return reverse_mapping.get(name_kr, name_kr)

@st.cache_data(ttl=3600)
def load_recipe(menu_sku_en):
    """레시피 로드 (BOM)"""
    global RECIPES
    return RECIPES.get(menu_sku_en, [])

# === [AI 기능 추가] ===
@st.cache_data(ttl=3600) # 1시간 캐시
def get_item_forecast(df_all_sales: pd.DataFrame, menu_sku_en: str, days_to_forecast: int):
    """
    [SPRINT 2] Prophet을 사용하여 지정된 메뉴의 미래 판매량을 예측합니다.
    """
    
    # 1. 해당 메뉴의 일별 판매량 데이터 준비
    try:
        menu_name_kr = to_korean_detail(menu_sku_en)
        df_item = df_all_sales[
            df_all_sales['상품상세'] == menu_name_kr
        ].copy()
        
        if df_item.empty:
            return None, None # 판매 데이터 없음

        df_agg = df_item.groupby('날짜')['수량'].sum().reset_index()
        df_agg['날짜'] = pd.to_datetime(df_agg['날짜'])
        
        # 날짜가 비어있는 경우 0으로 채우기 (Prophet 성능 향상)
        if not df_agg.empty:
            date_range = pd.date_range(start=df_agg['날짜'].min(), end=df_agg['날짜'].max())
            df_agg = df_agg.set_index('날짜').reindex(date_range, fill_value=0).reset_index()
            df_agg.rename(columns={'index': '날짜'}, inplace=True)
        
        # 2. Prophet이 요구하는 'ds', 'y' 컬럼명으로 변경
        df_prophet = df_agg[['날짜', '수량']].rename(columns={"날짜": "ds", "수량": "y"})

        if len(df_prophet) < 7: # 데이터가 너무 적으면(e.g., 7일 미만) 예측 불가
            return None, None

        # 3. 모델 학습 (주간 계절성 적용)
        m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
        m.fit(df_prophet)

        # 4. 미래 예측
        future = m.make_future_dataframe(periods=days_to_forecast)
        forecast = m.predict(future)
        
        # 5. 예측된 기간(target_days)의 총 소진량 합계 반환
        # 음수 예측은 0으로 클리핑
        forecast['yhat'] = forecast['yhat'].clip(lower=0) 
        predicted_sum = forecast.iloc[-days_to_forecast:]['yhat'].sum()
        
        return max(predicted_sum, 0), forecast # 예측 차트 데이터도 반환

    except Exception as e:
        st.warning(f"Prophet 예측 중 오류 발생: {e}")
        return None, None

# === [AI 기능 추가] ===
def call_openai_api(prompt, model="gpt-3.5-turbo"):
    """
    [SPRINT 1] OpenAI API를 호출하는 래퍼 함수.
    API 키가 없으면 Mock 응답을 반환합니다.
    """
    if not openai.api_key:
        # API 키가 없을 때 Mock 응답
        time.sleep(1.5) # AI가 생각하는 것처럼 보이게
        return f"✅ **[AI Mock 응답]**\n\n'secrets.toml'에 OpenAI API 키가 설정되지 않아 Mock 응답을 반환합니다.\n\n--- (요청 프롬프트) ---\n{prompt[:200]}..."

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 카페 운영 및 마케팅 전문가입니다. 요청받은 내용을 창의적이고 전문적으로 작성해주세요."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API 호출 중 오류 발생: {e}")
        return None

# ==============================================================
# 메인 데이터 로드
# ==============================================================
try:
    data_load_state = st.info("데이터베이스에서 최신 데이터를 로드하는 중... ⏳")
    all_data = load_data_from_fs()
    df, df_inv, df_moves, RECIPES, df_params = process_dataframes(all_data)
    data_load_state.success("데이터 로드 완료! ✅")
except Exception as e:
    data_load_state.error(f"데이터 로드 실패: {e}")
    st.stop()


# ==============================================================
# 사이드바 (메뉴)
# ==============================================================
st.sidebar.title("☕ Coffee ERP (GCP Ver.)")
menu_options = ["대시보드", "재고 관리", "거래 내역", "🤖 AI 비서", "도움말"] # === [AI 기능 추가] ===
menu = st.sidebar.radio("메뉴", menu_options)

st.sidebar.markdown("---")
st.sidebar.markdown(f"© 2025 한동대학교 ERP 연구팀")


# ==============================================================
# 📈 대시보드
# ==============================================================
if menu == "대시보드":
    st.header("📈 통합 대시보드")
    
    if df.empty:
        st.info("표시할 판매 데이터가 없습니다.")
    else:
        # 날짜 필터
        min_date = df['날짜'].min().date()
        max_date = df['날짜'].max().date()
        
        date_filter = st.slider(
            "조회 기간을 선택하세요",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        
        # 필터링된 데이터
        filtered_df = df[
            (df['날짜'].dt.date >= date_filter[0]) &
            (df['날짜'].dt.date <= date_filter[1])
        ]
        
        if filtered_df.empty:
            st.warning("선택한 기간에 데이터가 없습니다.")
        else:
            # 1. 핵심 지표 (KPI)
            st.subheader("📊 핵심 지표 (KPI)")
            total_revenue = filtered_df['수익'].sum()
            total_sales_count = filtered_df.shape[0]
            avg_revenue_per_sale = total_revenue / total_sales_count if total_sales_count > 0 else 0
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric(label="총 매출", value=format_krw(total_revenue))
            kpi2.metric(label="총 판매 건수", value=f"{total_sales_count:,}건")
            kpi3.metric(label="건당 평균 매출", value=format_krw(avg_revenue_per_sale))
            
            st.markdown("---")

            # 2. 시각화
            st.subheader("visual.ly")
            c1, c2 = st.columns(2)
            
            # 일별 매출 추이
            try:
                daily_revenue = filtered_df.groupby(filtered_df['날짜'].dt.date)['수익'].sum().reset_index()
                daily_revenue['날짜'] = pd.to_datetime(daily_revenue['날짜']) # 날짜 형식 복원
                fig_daily = px.line(daily_revenue, x='날짜', y='수익', title="일별 매출 추이", markers=True)
                fig_daily.update_layout(xaxis_title="날짜", yaxis_title="매출 (원)")
                c1.plotly_chart(fig_daily, use_container_width=True)
            except Exception as e:
                c1.error(f"일별 매출 차트 로드 실패: {e}")

            # 베스트셀러 Top 5 (수익 기준)
            try:
                top_products = filtered_df.groupby('상품상세')['수익'].sum().nlargest(5).reset_index()
                fig_top_prod = px.bar(top_products, x='상품상세', y='수익', title="베스트셀러 Top 5 (매출 기준)",
                                      color='상품상세', labels={'상품상세': '상품명', '수익': '매출액'})
                c2.plotly_chart(fig_top_prod, use_container_width=True)
            except Exception as e:
                c2.error(f"베스트셀러 차트 로드 실패: {e}")

            c3, c4 = st.columns(2)

            # 요일별/시간대별 판매
            try:
                day_order = ['월', '화', '수', '목', '금', '토', '일']
                hourly_sales = filtered_df.groupby(['요일', '시'])['수량'].sum().reset_index()
                hourly_sales_pivot = hourly_sales.pivot_table(index='요일', columns='시', values='수량', fill_value=0).reindex(day_order)
                
                fig_heatmap = px.imshow(hourly_sales_pivot,
                                        title="시간대별 / 요일별 판매 히트맵 (수량 기준)",
                                        labels=dict(x="시간 (시)", y="요일", color="판매 수량"),
                                        x=[f"{i}시" for i in hourly_sales_pivot.columns],
                                        y=hourly_sales_pivot.index,
                                        color_continuous_scale="Viridis"
                                       )
                c3.plotly_chart(fig_heatmap, use_container_width=True)
            except Exception as e:
                c3.error(f"히트맵 차트 로드 실패: {e}")

            # 카테고리별 매출 비중
            try:
                cat_revenue = filtered_df.groupby('상품카테고리')['수익'].sum().reset_index()
                fig_pie = px.pie(cat_revenue, values='수익', names='상품카테고리', title='상품 카테고리별 매출 비중')
                c4.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e:
                c4.error(f"파이 차트 로드 실패: {e}")


# ==============================================================
# 📦 재고 관리 (SPRINT 2 통합)
# ==============================================================
elif menu == "재고 관리":
    st.header("📦 재고 관리 및 스마트 발주")

    # [수정] compute_ingredient_metrics_for_menu 함수를 이 안으로 이동
    # === [AI 기능 추가] === (SPRINT 2: ML 수요 예측 적용)
    def compute_ingredient_metrics_for_menu(
        menu_sku_en: str,
        df_all_sales: pd.DataFrame, # [수정] 예측을 위해 전체 판매 데이터(df)가 필요
        df_inv: pd.DataFrame,
        df_params: pd.DataFrame,
        window_days: int = 28 # [수정] 이제는 Fallback용으로 사용
    ) -> pd.DataFrame:
        
        items = load_recipe(menu_sku_en)
        if not items:
            return pd.DataFrame()
        
        menu_name_kr = to_korean_detail(menu_sku_en)
        
        # 1. 이 메뉴의 최근 판매량 집계
        # [수정] 이 부분은 이제 AI 예측을 위한 '폴백(Fallback)' 로직이 됨
        start_date = pd.Timestamp.now() - pd.Timedelta(days=window_days)
        df_menu_agg = df_all_sales[
            (df_all_sales['상품상세'] == menu_name_kr) &
            (df_all_sales['날짜'] >= start_date)
        ]
        historical_sold_sum = df_menu_agg['수량'].sum()

        # === [AI 기능 시작] ===
        # 2. AI로 미래 수요 예측
        # sku_params에서 이 메뉴의 'target_days' (재고 목표일수)를 가져옴
        try:
            target_days_forecast = int(df_params.loc[df_params['sku_en'] == menu_sku_en, 'target_days'].values[0])
        except Exception:
            target_days_forecast = 21 # 파라미터 없으면 기본 21일
        
        predicted_menu_sales, forecast_chart_data = get_item_forecast(
            df_all_sales, menu_sku_en, days_to_forecast=target_days_forecast
        )

        use_historical_fallback = False
        
        if predicted_menu_sales is None or predicted_menu_sales == 0:
            st.warning(f"🤖 AI 예측: '{menu_name_kr}'의 판매 데이터가 부족합니다. 과거 {window_days}일 평균 판매량({historical_sold_sum}개)을 기준으로 계산합니다.")
            sold_sum = historical_sold_sum
            days = window_days
            use_historical_fallback = True
        else:
            st.success(f"🤖 **AI 예측**: '{menu_name_kr}'의 향후 **{target_days_forecast}일간** 예상 판매량을 **{predicted_menu_sales:,.0f}개**로 예측했습니다.")
            sold_sum = predicted_menu_sales # 예측값으로 대체
            days = target_days_forecast # 기준일도 예측 기간으로 변경
            
            # (옵션) 예측 차트 표시
            if forecast_chart_data is not None:
                try:
                    fig = px.line(forecast_chart_data.iloc[-90:], x='ds', y='yhat', 
                                  title=f"'{menu_name_kr}' 수요 예측 (향후 {target_days_forecast}일)", 
                                  labels={'ds':'날짜', 'yhat':'예측 판매량'})
                    fig.add_scatter(x=forecast_chart_data['ds'], y=forecast_chart_data['yhat_lower'], fill='tozeroy', mode='lines', line=dict(color='rgba(0,0,0,0)'), name='불확실성')
                    fig.add_scatter(x=forecast_chart_data['ds'], y=forecast_chart_data['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'), fillcolor='rgba(231, 234, 241, 0.5)', name='')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"예측 차트 생성 오류: {e}")

        # === [AI 기능 종료] ===


        # 3. 레시피 기반 원재료 소진량 계산
        rows = []
        for item in items:
            sku_en = item['sku_en']
            g_per_unit = item['g_per_unit']
            rows.append({"sku_en": sku_en, "g_per_unit": g_per_unit})
        
        if not rows:
            return pd.DataFrame()

        use_df = pd.DataFrame(rows)
        
        # 4. 소진량 계산 (BOM * 판매량)
        use_df = use_df.merge(df_params[['sku_en', 'loss_rate']], on="sku_en", how="left").fillna(0)
        use_df['최근소진합'] = use_df['g_per_unit'] * sold_sum * (1 + use_df['loss_rate'])
        
        # 5. 재고 지표 계산
        base = use_df.groupby('sku_en')[['최근소진합']].sum()
        
        # [수정] 'days' 변수 사용 (AI 예측일 또는 과거 Window일)
        base["일평균소진"] = (base["최근소진합"] / max(days, 1)).round(3) 
        base.loc[base["일평균소진"].eq(0), "일평균소진"] = 0.01 # 0으로 나누기 방지
        
        base = base.reset_index().merge(df_inv[['sku_en', '현재재고']], on='sku_en', how='left')
        base['현재재고'] = base['현재재고'].fillna(0)
        
        base["커버일수"] = (base["현재재고"] / base["일평균소진"]).round(1)

        # 6. ROP 및 권장 발주량 계산
        base = base.merge(df_params, on="sku_en", how="left")
        
        # 파라미터가 없는 경우 기본값
        base['lead_time_days'] = base['lead_time_days'].fillna(3)
        base['safety_stock_units'] = base['safety_stock_units'].fillna(0)

        base["ROP"] = (base["일평균소진"] * base["lead_time_days"] + base["safety_stock_units"]).round(0).astype(int)
        
        # [핵심] 권장 발주량: (AI가 예측한 총 소진량) - (현재 재고)
        base["권장발주"] = (base["최근소진합"] - base["현재재고"]).apply(lambda x: max(int(ceil(x)), 0))
        
        base["상태"] = base.apply(lambda r: "🚨 발주요망" if r["현재재고"] <= r["ROP"] else "✅ 정상", axis=1)

        return base


    # --- 재고 관리 페이지 UI 시작 ---
    tab1, tab2 = st.tabs(["🎛️ 메뉴별 재고 현황", "✍️ 재고 수기 관리"])

    with tab1:
        st.subheader("🎛️ 메뉴별 재고 현황 (AI 예측 기반)")
        
        if df.empty or df_inv.empty or df_params.empty:
            st.warning("판매, 재고 또는 품목 파라미터 데이터가 부족하여 재고 현황을 계산할 수 없습니다.")
        else:
            menu_list = [to_korean_detail(sku) for sku in RECIPES.keys()]
            selected_menu_kr = st.selectbox("분석할 메뉴를 선택하세요:", menu_list, index=0)
            selected_menu_en = from_korean_detail(selected_menu_kr)
            
            st.markdown("---")
            
            try:
                # [수정] compute_ingredient_metrics_for_menu 호출 시 전체 df 전달
                report_df = compute_ingredient_metrics_for_menu(
                    selected_menu_en,
                    df, # SPRINT 2: 예측을 위해 전체 판매 데이터(df) 전달
                    df_inv,
                    df_params
                )
                
                if report_df.empty:
                    st.info(f"'{selected_menu_kr}'에 대한 레시피 정보가 없습니다.")
                else:
                    # 컬럼 순서 및 한글화
                    report_df['품목명'] = report_df['sku_en'].apply(to_korean_detail)
                    report_df['단위'] = report_df['uom']
                    
                    display_cols = [
                        '품목명', '상태', '현재재고', '단위', '권장발주', '커버일수', '일평균소진', 'ROP',
                        'lead_time_days', 'safety_stock_units'
                    ]
                    
                    # 단위 포맷팅
                    formatted_df = report_df[display_cols].copy()
                    formatted_df['현재재고'] = formatted_df.apply(lambda r: f"{r['현재재고']:,.1f} {r['단위']}", axis=1)
                    formatted_df['권장발주'] = formatted_df.apply(lambda r: f"{r['권장발주']:,.1f} {r['단위']}", axis=1)
                    formatted_df['일평균소진'] = formatted_df.apply(lambda r: f"{r['일평균소진']:,.1f} {r['단위']}", axis=1)
                    formatted_df['ROP'] = formatted_df.apply(lambda r: f"{r['ROP']:,.1f} {r['단위']}", axis=1)
                    formatted_df['커버일수'] = formatted_df['커버일수'].apply(lambda x: f"{x}일")
                    
                    st.dataframe(
                        formatted_df[['품목명', '상태', '현재재고', '권장발주', '커버일수', '일평균소진', 'ROP']],
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"재고 리포트 생성 중 오류가 발생했습니다: {e}")
                import traceback
                st.exception(traceback.format_exc())


    with tab2:
        st.subheader("✍️ 재고 수기 관리 (실사 반영)")
        st.info("실제 재고를 확인한 후, 수량을 직접 수정하고 저장하세요.")
        
        if df_inv.empty:
            st.warning("재고 정보가 없습니다.")
        else:
            # 편집용 데이터프레임 생성
            edit_df = df_inv.copy()
            edit_df['품목명'] = edit_df['sku_en'].apply(to_korean_detail)
            edit_df['현재재고'] = pd.to_numeric(edit_df['현재재고'], errors='coerce').fillna(0)
            
            # 편집기
            edited_data = st.data_editor(
                edit_df[['품목명', '현재재고', 'sku_en']],
                column_config={
                    "품목명": st.column_config.TextColumn("품목명", disabled=True),
                    "현재재고": st.column_config.NumberColumn("현재 재고", min_value=0.0, format="%.2f"),
                    "sku_en": st.column_config.TextColumn("SKU (Eng)", disabled=True),
                },
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic" # 새 품목 추가 허용 (현재는 sku_en이 없어서 저장 안됨. 추후 기능 확장)
            )
            
            if st.button("실사 재고 저장하기 💾", type="primary"):
                db = firestore.client()
                batch = db.batch()
                changed = 0
                
                # 원본과 비교
                original_inv_map = {row['sku_en']: row['현재재고'] for _, row in df_inv.iterrows()}
                
                for item in edited_data:
                    sku = item['sku_en']
                    new_stock = float(item['현재재고'])
                    
                    if sku in original_inv_map and original_inv_map[sku] != new_stock:
                        doc_ref = db.collection('inventory').document(sku)
                        batch.set(doc_ref, {'stock': new_stock, 'sku_en': sku}, merge=True)
                        changed += 1
                        
                if changed > 0:
                    batch.commit()
                    st.success(f"✅ 재고 {changed}건 저장 완료")
                    safe_rerun()
                else:
                    st.info("변경된 내용이 없습니다.")


# =============================================================
# 🤖 AI 비서 (SPRINT 1)
# =============================================================
elif menu == "🤖 AI 비서": # === [AI 기능 추가] ===
    st.header("🤖 AI 마케팅/운영 비서")
    st.markdown("현재 판매 데이터를 기반으로 AI가 마케팅 문구나 운영 보고서를 생성합니다.")

    if df.empty:
        st.info("아직 판매 데이터가 없습니다. 데이터가 쌓이면 AI 비서를 사용할 수 있습니다.")
    else:
        try:
            # 1. 현재 데이터 분석 (기존 로직 재활용)
            total_revenue = df['수익'].sum()
            total_sales_count = len(df)
            
            top_prod_series = df.groupby('상품상세')['수익'].sum().sort_values(ascending=False).head(3)
            top_prod_list = [f"{idx} ({format_krw(val)})" for idx, val in top_prod_series.items()]
            top_prod_str = ", ".join(top_prod_list)
            
            st.info(f"데이터 기준: 총 매출 {format_krw(total_revenue)}, 총 판매 {total_sales_count}건, 베스트셀러: **{top_prod_str}**")

            # 2. 프롬프트 선택
            prompt_options = {
                "인스타그램 홍보 (활기찬 톤)": f"우리는 작은 카페입니다. 이번 주 베스트셀러는 {top_prod_str} 입니다. 이 메뉴를 강조하는 인스타그램 홍보 게시물을 '매우' 친근하고 활기찬 톤으로 작성해줘. 이모지도 팍팍 넣어주고 해시태그도 5개 이상 달아줘.",
                "단골손님 감사 문자 (정중한 톤)": f"이번 주 베스트셀러({top_prod_str})를 기반으로, 단골손님에게 감사를 표하는 SMS 문자 메시지를 정중하지만 따뜻하게 작성해줘.",
                "일일 운영 보고 (매니저용)": f"오늘의 총 매출은 {format_krw(total_revenue)}, 총 판매 건수는 {total_sales_count}건이야. 베스트셀러는 {top_prod_list[0]}이고. 이 내용을 바탕으로 매니저에게 보고할 간결한 일일 운영 요약 보고서를 작성해줘. (숫자 요약 포함)"
            }
            
            selected_prompt_key = st.selectbox("AI에게 요청할 작업을 선택하세요:", list(prompt_options.keys()))
            
            custom_prompt_area = st.text_area("또는, AI에게 직접 요청할 내용을 입력하세요:", placeholder="예: 베스트셀러 메뉴 3가지를 활용한 신규 세트 메뉴 아이디어 3가지 제안해줘")
            
            if st.button("AI 생성하기 🚀", type="primary"):
                
                final_prompt = ""
                if custom_prompt_area:
                    st.info("직접 입력한 프롬프트로 요청합니다...")
                    final_prompt = custom_prompt_area
                else:
                    final_prompt = prompt_options[selected_prompt_key]

                with st.spinner("AI가 열심히 생각 중입니다... 🧠"):
                    
                    # [실제 API 호출]
                    result_text = call_openai_api(final_prompt)
                    
                    if result_text:
                        st.success("AI 생성 완료!")
                        st.text_area("결과물:", result_text, height=300)
                    else:
                        st.error("AI 응답 생성에 실패했습니다.")

        except Exception as e:
            st.error(f"데이터를 분석하는 중 오류가 발생했습니다: {e}")


# =============================================================
# 📋 거래 내역
# =============================================================
elif menu == "거래 내역":
    st.header("📋 전체 거래 내역")
    if df.empty:
        st.info("표시할 거래 데이터가 없습니다.")
    else:
        cols = ['날짜','상품카테고리','상품타입','상품상세','수량','단가','수익','요일','시']
        cols = [c for c in cols if c in df.columns]
        st.caption(f"현재 데이터 크기: {len(df)}행")
        
        # [수정] st.dataframe(df.head(1000)) -> 불필요한 중복 제거
        st.dataframe(df[cols].sort_values('날짜', ascending=False), use_container_width=True)


# =============================================================
# ❓ 도움말
# =============================================================
else:  # menu == "도움말"
    st.header("☕️ 커피 원두 재고관리 파이프라인 쉽게 이해하기")
    st.markdown("""
> **“커피 원두가 어떻게 들어오고, 얼마나 쓰이고, 언제 다시 주문돼야 하는지를 자동으로 관리하자!”** 엑셀 대신 ERP가 자동으로 계산해줍니다.

### 1. (AI) 스마트 발주 로직 (재고 관리 탭)
| 단계 | 하는 일 | 예시 |
| --- | --- | --- |
| **1. (AI) 수요 예측** | Prophet (ML)이 "아메리카노"의 **미래 21일** 판매량을 **[500잔]**으로 예측 |
| **2. 소진량 계산** | [500잔] x [레시피: 잔당 20g] = **[10,000g]** (예상 총 소진량) |
| **3. 권장 발주량** | [10,000g] - [현재 재고: 3,000g] = **[7,000g]** (권장 발주량) |
| **4. ROP (발주점)** | (일평균소진 * 리드타임) + 안전재고. 이보다 재고가 낮으면 **'🚨 발주요망'** 알림 |

### 2. (AI) 마케팅 보조 (AI 비서 탭)
| 기능 | 설명 |
| --- | --- |
| **인스타그램 생성** | 현재 베스트셀러 데이터를 기반으로 AI가 홍보 문구를 자동 생성합니다. |
| **운영 보고** | 일일 매출, 판매 건수 등을 요약하여 간결한 보고서를 생성합니다. |

### 3. 기본 데이터 흐름
| 단계 | 하는 일 | 예시 |
| --- | --- | --- |
| **1. 원두 입고** | 카페가 원두를 사와서 '재고 수기 관리' 탭에서 **[+10,000g]** 입력 |
| **2. 판매 발생** | POS에서 '아메리카노' 1잔 판매 (Firestore 'coffee_sales'에 자동 기록) |
| **3. 자동 차감** | 시스템이 '아메리카노' 레시피(BOM)를 조회하여 [원두 A: 20g] 사용 확인 |
| **4. 재고 반영** | 'inventory' DB의 '원두 A' 재고를 **[-20g]** 자동 차감 (이 기능은 현재 시뮬레이션됨) |
""")