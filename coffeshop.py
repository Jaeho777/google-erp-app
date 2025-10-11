# ==============================================================
# ☕ Coffee ERP Dashboard — Company Showcase Edition (Tone-Down Blue)
#  - 재고 관리(자동 차감/임계치 경고/자동 발주 시뮬레이션)
#  - UI 한글화(이름 매핑 + 요일 한글 표시)
#  - 원본/Firestore는 영어 저장, 화면은 한글 표시(정/역매핑)
#  - 데이터 편집(거래 수정/삭제 + 재고 일괄수정)
#  - 도움말 탭 + SKU 파라미터(리드타임/세이프티/목표일수/레시피g) + ROP 지표/권장발주
# ==============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from pathlib import Path
from math import ceil
import re

# ----------------------
# 0️⃣ 경로/상수 설정
# ----------------------
SERVICE_ACCOUNT_PATH = "/Users/jaeholee/Desktop/google-erp-app/keys/serviceAccount.json"
CSV_PATH = "/Users/jaeholee/Desktop/google-erp-app/Coffee Shop Sales.csv"
PIPELINE_IMG = "/Users/jaeholee/Desktop/google-erp-app/assets/pipeline_diagram.png"

SALES_COLLECTION = "coffee_sales"
INVENTORY_COLLECTION = "inventory"
ORDERS_COLLECTION = "orders"
SKU_PARAMS_COLLECTION = "sku_params"   # ★ 추가: SKU 파라미터 저장 컬렉션

USE_KRW_CONVERSION = False   # CSV가 USD면 True로
KRW_PER_USD = 1350

DEFAULT_INITIAL_STOCK = 100
REORDER_THRESHOLD_RATIO = 0.15  # 15%

# ----------------------
# 0-1️⃣ Firebase 초기화
# ----------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ----------------------
# 0-2️⃣ UI/스타일
# ----------------------
st.set_page_config(page_title="☕ Coffee ERP Dashboard", layout="wide")
pio.templates.default = "plotly_white"
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Blues"

st.markdown("""
    <style>
    .stApp { background-color: #F4F6FA; font-family: 'Pretendard','Noto Sans KR',sans-serif; }
    .dashboard-header { display:flex; align-items:center; gap:12px; background:#1E2A38; color:#fff;
                        padding:15px 25px; border-radius:10px; margin-bottom:25px; }
    .metric-card { background:#fff; border-radius:16px; padding:25px;
                   box-shadow:0 4px 12px rgba(0,0,0,0.06); text-align:center; transition:all .3s; }
    .metric-card:hover { transform: translateY(-3px); box-shadow:0 6px 14px rgba(0,0,0,0.12); }
    .metric-title { color:#7C8DA6; font-size:1em; }
    .metric-value { color:#2C3E50; font-size:1.8em; font-weight:600; }
    section[data-testid="stSidebar"] { background:#fff !important; color:#0D3B66 !important; }
    label[data-baseweb="radio"] div { color:#0D3B66 !important; font-weight:500; }
    .js-plotly-plot .plotly { background:transparent !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-header">
  <h1>☕ Coffee ERP Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# ----------------------
# 0-3️⃣ 한글 매핑 테이블
# ----------------------
# (1) 카테고리: EN→KO
category_map = {
    "Coffee": "커피", "Tea": "차", "Bakery": "베이커리",
    "Coffee beans": "원두", "Drinking Chocolate": "초코음료",
    "Equipment": "장비", "Extras": "기타",
    "Branded": "브랜드 상품",
    "Flavours": "플레이버(시럽)",
    "Flavors": "플레이버(시럽)",
    "Loose Tea": "잎차",
    "Loose-Leaf Tea": "잎차",
    "Packaged Chocolate": "포장 초콜릿",
}
rev_category_map = {v: k for k, v in category_map.items()}
rev_category_map.update({
    "베이커리": "Bakery",
    "원두": "Coffee beans",
    "차": "Tea",
    "초코음료": "Drinking Chocolate",
    "커피": "Coffee",
})

# (2) 타입: EN→KO
type_map = {
    "Barista Espresso": "바리스타 에스프레소",
    "Biscotti": "비스코티",
    "Brewed Black tea": "홍차(브루드)",
    "Brewed Chai tea": "차이 티(브루드)",
    "Brewed Green tea": "녹차(브루드)",
    "Brewed herbal tea": "허브티(브루드)",
    "Chai tea": "차이 티",
    "Clothing": "의류",
    "Drinking Chocolate": "초코음료",
    "Drip coffee": "드립 커피",
    "Espresso Beans": "에스프레소 원두",
    "Gourmet Beans": "고급 원두",
    "Gourmet brewed coffee": "고급 브루드 커피",
    "Green beans": "생두",
    "Herbal tea": "허브티",
    "House blend Beans": "하우스 블렌드 원두",
    "Housewares": "생활용품",
    "Organic Beans": "유기농 원두",
    "Organic Chocolate": "유기농 초콜릿",

    "Organic brewed coffee": "유기농 브루드 커피",
    "Premium brewed coffee": "프리미엄 브루드 커피",
    "Premium Beans": "프리미엄 원두",
    "Regular syrup": "일반 시럽",
    "Sugar free syrup": "무설탕 시럽",
    "Pastry": "페이스트리",
    "Scone": "스콘",
    "Hot chocolate": "핫초코",
    "Green tea": "녹차",
    "Black tea": "홍차",
    "Americano": "아메리카노",
    "Latte": "라떼",
    "Espresso": "에스프레소",
    "Cappuccino": "카푸치노",
    "Mocha": "모카",
    "Flat White": "플랫화이트",

    "Premium beans": "프리미엄 원두",
    "Regular Syrup": "일반 시럽",
    "Sugar Free Syrup": "무설탕 시럽",
    "Organic Brewed Coffee": "유기농 브루드 커피",
    "Premium Brewed Coffee": "프리미엄 브루드 커피",
}
rev_type_map = {v: k for k, v in type_map.items()}

# (3) 상세: 규칙 기반(사이즈 자동 인식) + 상세 베이스 대량 매핑
SIZE_SUFFIX_MAP = {"Lg": "라지", "Rg": "레귤러", "Sm": "스몰"}
REV_SIZE_SUFFIX_MAP = {"라지": "Lg", "레귤러": "Rg", "스몰": "Sm"}

detail_base_map = {
    "Almond Croissant": "아몬드 크루아상",
    "Brazilian": "브라질",
    "Brazilian - Organic": "브라질 유기농",
    "Cappuccino": "카푸치노",
    "Carmel syrup": "카라멜 시럽",
    "Caramel syrup": "카라멜 시럽",
    "Chili Mayan": "칠리 마야",
    "Chocolate Chip Biscotti": "초코칩 비스코티",
    "Chocolate Croissant": "초콜릿 크루아상",
    "Chocolate syrup": "초콜릿 시럽",
    "Civet Cat": "코피 루왁",
    "Columbian Medium Roast": "콜롬비아 미디엄 로스트",
    "Colombian Medium Roast": "콜롬비아 미디엄 로스트",
    "Cranberry Scone": "크랜베리 스콘",
    "Croissant": "크루아상",
    "Dark chocolate": "다크 초콜릿",
    "Earl Grey": "얼그레이",
    "English Breakfast": "잉글리시 브렉퍼스트",
    "Espresso Roast": "에스프레소 로스트",
    "Espresso shot": "에스프레소 샷",
    "Ethiopia": "에티오피아",
    "Ginger Biscotti": "진저 비스코티",
    "Ginger Scone": "진저 스콘",
    "Guatemalan Sustainably Grown": "과테말라 지속가능 재배",
    "Hazelnut Biscotti": "헤이즐넛 비스코티",
    "Hazelnut syrup": "헤이즐넛 시럽",
    "I Need My Bean! Diner mug": "I Need My Bean! 다이너 머그",
    "I Need My Bean! Latte cup": "I Need My Bean! 라떼 컵",
    "I Need My Bean! T-shirt": "I Need My Bean! 티셔츠",
    "Jamacian Coffee River": "자메이카 커피 리버",
    "Jamaican Coffee River": "자메이카 커피 리버",
    "Jumbo Savory Scone": "점보 세이보리 스콘",
    "Latte": "라떼",
    "Lemon Grass": "레몬그라스",
    "Morning Sunrise Chai": "모닝 선라이즈 차이",
    "Oatmeal Scone": "오트밀 스콘",
    "Organic Decaf Blend": "유기농 디카페인 블렌드",
    "Our Old Time Diner Blend": "아워 올드 타임 다이너 블렌드",
    "Ouro Brasileiro shot": "오우로 브라질 샷",
    "Peppermint": "페퍼민트",
    "Primo Espresso Roast": "프리모 에스프레소 로스트",
    "Scottish Cream Scone": "스코티시 크림 스콘",
    "Serenity Green Tea": "세레니티 그린 티",
    "Spicy Eye Opener Chai": "스파이시 아이 오프너 차이",
    "Sugar Free Vanilla syrup": "무설탕 바닐라 시럽",
    "Sustainably Grown Organic": "지속가능 유기농",
    "Traditional Blend Chai": "트래디셔널 블렌드 차이",
}
rev_detail_base_map = {v: k for k, v in detail_base_map.items()}

def to_korean_detail(name: str) -> str:
    s = str(name).strip()
    if re.search(r"\((라지|레귤러|스몰)\)$", s):
        return s
    m = re.search(r"\s+(Lg|Rg|Sm)$", s)
    size_en = m.group(1) if m else None
    base_en = s[: -len(size_en) - 1] if size_en else s
    base_ko = detail_base_map.get(base_en, base_en)
    if size_en:
        return f"{base_ko} ({SIZE_SUFFIX_MAP[size_en]})"
    return base_ko

def from_korean_detail(display: str) -> str:
    s = str(display).strip()
    if re.search(r"\s+(Lg|Rg|Sm)$", s):
        return s
    m = re.search(r"\((라지|레귤러|스몰)\)$", s)
    size_ko = m.group(1) if m else None
    base_ko = re.sub(r"\s*\((라지|레귤러|스몰)\)$", "", s)
    base_en = rev_detail_base_map.get(base_ko, base_ko)
    if size_ko:
        return f"{base_en} {REV_SIZE_SUFFIX_MAP[size_ko]}"
    return base_en

# 요일 한글화
weekday_map = {"Monday": "월", "Tuesday": "화", "Wednesday": "수",
               "Thursday": "목", "Friday": "금", "Saturday": "토", "Sunday": "일"}
weekday_order_kr = ["월", "화", "수", "목", "금", "토", "일"]

def map_series(s: pd.Series, mapping: dict) -> pd.Series:
    return s.apply(lambda x: mapping.get(x, x))

# ----------------------
# 1️⃣ CSV 로드
# ----------------------
@st.cache_data(ttl=0)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        'transaction_id': '거래번호', 'transaction_date': '날짜', 'transaction_time': '시간',
        'transaction_qty': '수량', 'store_id': '가게ID', 'store_location': '가게위치',
        'product_id': '상품ID', 'unit_price': '단가', 'product_category': '상품카테고리',
        'product_type': '상품타입', 'product_detail': '상품상세', 'Revenue': '수익'
    })
    df['수익'] = df['수익'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)
    df['단가'] = df['단가'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)
    if USE_KRW_CONVERSION:
        df['수익'] *= KRW_PER_USD
        df['단가'] *= KRW_PER_USD

    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    if '시간' in df.columns:
        df['시'] = pd.to_datetime(df['시간'], format='%H:%M:%S', errors='coerce').dt.hour
    else:
        df['시'] = None

    df['요일'] = df['날짜'].dt.day_name()
    df['월'] = df['날짜'].dt.month
    return df

df_csv = load_csv(CSV_PATH)

# ----------------------
# 2️⃣ Firestore(판매) 로드
# ----------------------
def load_sales_from_firestore() -> pd.DataFrame:
    docs = db.collection(SALES_COLLECTION).stream()
    data = [d.to_dict() for d in docs]
    df_fb = pd.DataFrame(data)
    if df_fb.empty:
        return df_fb

    df_fb['날짜'] = pd.to_datetime(df_fb['날짜'], errors='coerce')
    if '수익' in df_fb.columns:
        df_fb['수익'] = pd.to_numeric(df_fb['수익'], errors='coerce')
    if '단가' in df_fb.columns:
        df_fb['단가'] = pd.to_numeric(df_fb['단가'], errors='coerce')
    if '수량' in df_fb.columns:
        df_fb['수량'] = pd.to_numeric(df_fb['수량'], errors='coerce')

    if '시간' in df_fb.columns:
        df_fb['시'] = pd.to_datetime(df_fb['시간'], format='%H:%M:%S', errors='coerce').dt.hour
    else:
        df_fb['시'] = None

    df_fb['요일'] = df_fb['날짜'].dt.day_name()
    df_fb['월'] = df_fb['날짜'].dt.month
    return df_fb

df_fb = load_sales_from_firestore()

# 거래 편집용: 문서 ID 포함 로더
def load_sales_with_id():
    docs = db.collection(SALES_COLLECTION).stream()
    rows = []
    for d in docs:
        rec = d.to_dict()
        rec["_id"] = d.id
        rows.append(rec)
    df_raw = pd.DataFrame(rows)
    if df_raw.empty:
        return df_raw, df_raw

    df_raw['날짜'] = pd.to_datetime(df_raw['날짜'], errors='coerce')
    if '수익' in df_raw: df_raw['수익'] = pd.to_numeric(df_raw['수익'], errors='coerce')
    if '단가' in df_raw: df_raw['단가'] = pd.to_numeric(df_raw['단가'], errors='coerce')
    if '수량' in df_raw: df_raw['수량'] = pd.to_numeric(df_raw['수량'], errors='coerce')

    df_view = df_raw.copy()
    if '상품카테고리' in df_view: df_view['상품카테고리'] = map_series(df_view['상품카테고리'], category_map)
    if '상품타입' in df_view: df_view['상품타입'] = map_series(df_view['상품타입'], type_map)
    if '상품상세' in df_view: df_view['상품상세'] = df_view['상품상세'].apply(to_korean_detail)
    return df_raw, df_view

# ----------------------
# 3️⃣ CSV + Firebase 통합 → 화면표시용 한글화
# ----------------------
df = pd.concat([df_csv, df_fb], ignore_index=True)
if '요일' in df.columns:
    df['요일'] = map_series(df['요일'], weekday_map)
if '상품카테고리' in df.columns:
    df['상품카테고리'] = map_series(df['상품카테고리'], category_map)
if '상품타입' in df.columns:
    df['상품타입'] = map_series(df['상품타입'], type_map)
if '상품상세' in df.columns:
    df['상품상세'] = df['상품상세'].apply(to_korean_detail)

# ----------------------
# 4️⃣ 공용 유틸
# ----------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def format_krw(x: float) -> str:
    try:
        return f"{x:,.0f} 원"
    except Exception:
        return "-"

def ensure_inventory_doc(product_detail_en: str):
    ref = db.collection(INVENTORY_COLLECTION).document(product_detail_en)
    doc = ref.get()
    if not doc.exists:
        ref.set({
            "상품상세_en": product_detail_en,
            "초기재고": DEFAULT_INITIAL_STOCK,
            "현재재고": DEFAULT_INITIAL_STOCK
        })
    return ref

def deduct_stock(product_detail_en: str, qty: int):
    ref = ensure_inventory_doc(product_detail_en)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else {}
    init_stock = int(data.get("초기재고", DEFAULT_INITIAL_STOCK))
    cur_stock = int(data.get("현재재고", DEFAULT_INITIAL_STOCK))
    new_stock = max(cur_stock - int(qty), 0)
    ref.update({"현재재고": new_stock})
    return init_stock, new_stock

def load_inventory_df() -> pd.DataFrame:
    inv_docs = db.collection(INVENTORY_COLLECTION).stream()
    rows = []
    for d in inv_docs:
        doc = d.to_dict()
        en = doc.get("상품상세_en", d.id)
        ko = to_korean_detail(en)
        rows.append({
            "상품상세_en": en,
            "상품상세": ko,
            "초기재고": doc.get("초기재고", DEFAULT_INITIAL_STOCK),
            "현재재고": doc.get("현재재고", DEFAULT_INITIAL_STOCK)
        })
    return pd.DataFrame(rows)

# ---- SKU 파라미터 로드/저장 ----
def load_sku_params_df() -> pd.DataFrame:
    docs = db.collection(SKU_PARAMS_COLLECTION).stream()
    rows = []
    for d in docs:
        item = d.to_dict()
        item["_id"] = d.id
        rows.append(item)
    dfp = pd.DataFrame(rows)
    if dfp.empty:
        dfp = pd.DataFrame(columns=[
            "_id","sku_en","lead_time_days","safety_stock_units","target_days","grams_per_cup","expiry_days"
        ])
    # 기본값 보강
    for col, default in [
        ("lead_time_days", 3), ("safety_stock_units", 10),
        ("target_days", 21), ("grams_per_cup", 18.0), ("expiry_days", 28)
    ]:
        if col not in dfp.columns:
            dfp[col] = default
        else:
            dfp[col] = pd.to_numeric(dfp[col], errors="coerce").fillna(default)
    return dfp

def upsert_sku_params(dfp: pd.DataFrame):
    saved = 0
    for _, r in dfp.iterrows():
        sku_en = str(r["sku_en"]).strip()
        if not sku_en:
            continue
        doc = db.collection(SKU_PARAMS_COLLECTION).document(sku_en)
        patch = {
            "sku_en": sku_en,
            "lead_time_days": int(r.get("lead_time_days", 3)),
            "safety_stock_units": int(r.get("safety_stock_units", 10)),
            "target_days": int(r.get("target_days", 21)),
            "grams_per_cup": float(r.get("grams_per_cup", 18.0)),
            "expiry_days": int(r.get("expiry_days", 28)),
        }
        doc.set(patch)
        saved += 1
    return saved

# ---- ROP/권장발주 계산 ----
def compute_replenishment_metrics(df_all_sales: pd.DataFrame, df_inv: pd.DataFrame, df_params: pd.DataFrame, window_days: int = 28) -> pd.DataFrame:
    if df_inv.empty:
        return pd.DataFrame()

    # 판매 윈도우
    if "날짜" in df_all_sales.columns and pd.api.types.is_datetime64_any_dtype(df_all_sales["날짜"]):
        max_day = df_all_sales["날짜"].max()
        min_day = max_day - pd.Timedelta(days=window_days-1)
        df_win = df_all_sales[(df_all_sales["날짜"] >= min_day) & (df_all_sales["날짜"] <= max_day)].copy()
    else:
        df_win = df_all_sales.copy()

    # KO 표시 → EN SKU 키
    if "상품상세" in df_win.columns:
        df_win = df_win.copy()
        df_win["sku_en"] = df_win["상품상세"].apply(from_korean_detail)
    else:
        df_win["sku_en"] = ""

    # 수량 numeric
    if "수량" in df_win.columns:
        df_win["수량"] = pd.to_numeric(df_win["수량"], errors="coerce").fillna(0)
    sales_agg = df_win.groupby("sku_en")["수량"].sum().reset_index().rename(columns={"수량":"최근판매합"})

    base = df_inv.rename(columns={"상품상세_en":"sku_en"}).copy()
    base = base.merge(df_params, on="sku_en", how="left")
    base = base.merge(sales_agg, on="sku_en", how="left")
    base["최근판매합"] = pd.to_numeric(base["최근판매합"], errors="coerce").fillna(0)

    days = max(window_days, 1)
    base["일평균소진"] = (base["최근판매합"] / days).round(3)
    base["일평균소진"] = base["일평균소진"].replace([0], 0.01)  # 시연용 최소치
    base["커버일수"] = (base["현재재고"] / base["일평균소진"]).round(1)

    base["lead_time_days"] = pd.to_numeric(base.get("lead_time_days", 3), errors="coerce").fillna(3).astype(int)
    base["safety_stock_units"] = pd.to_numeric(base.get("safety_stock_units", 10), errors="coerce").fillna(10).astype(int)
    base["target_days"] = pd.to_numeric(base.get("target_days", 21), errors="coerce").fillna(21).astype(int)

    base["ROP"] = (base["일평균소진"] * base["lead_time_days"] + base["safety_stock_units"]).round(0).astype(int)
    base["권장발주"] = ((base["target_days"] * base["일평균소진"]) - base["현재재고"]).apply(lambda x: max(int(ceil(x)), 0))
    base["상태"] = base.apply(lambda r: "발주요망" if r["현재재고"] <= r["ROP"] else "정상", axis=1)

    cols = [
        "상품상세","sku_en","현재재고","초기재고",
        "최근판매합","일평균소진","커버일수","lead_time_days","safety_stock_units","target_days",
        "ROP","권장발주","상태"
    ]
    for c in cols:
        if c not in base.columns: base[c] = None
    out = base[cols].sort_values(["상태","커버일수"])
    return out

# ----------------------
# 5️⃣ 사이드바 메뉴
# ----------------------
menu = st.sidebar.radio(
    " 메뉴 선택",
    ["경영 현황", "매출 대시보드", "기간별 분석", "거래 추가", "재고 관리", "데이터 편집", "거래 내역", "도움말"]
)

# ==============================================================
# 🧾 거래 추가
# ==============================================================
if menu == "거래 추가":
    st.header(" 거래 데이터 추가")

    category_options = sorted(pd.Series(df['상품카테고리']).dropna().unique().tolist())
    type_options = sorted(pd.Series(df['상품타입']).dropna().unique().tolist())
    detail_options = sorted(pd.Series(df['상품상세']).dropna().unique().tolist())

    with st.form("add_transaction"):
        col1, col2 = st.columns(2)
        with col1:
            날짜 = st.date_input("날짜", value=datetime.now().date())
            상품카테고리_ko = st.selectbox("상품카테고리", category_options)
            상품타입_ko = st.selectbox("상품타입", type_options)
        with col2:
            상품상세_ko = st.selectbox("상품상세", detail_options)
            수량 = st.number_input("수량", min_value=1, value=1)
            단가 = st.number_input("단가(원)", min_value=0.0, value=1000.0, step=100.0)

        수익 = 수량 * 단가
        st.markdown(f"### 💰 계산된 수익: **{format_krw(수익)}**")

        submitted = st.form_submit_button("데이터 추가")
        if submitted:
            상품카테고리_en = rev_category_map.get(상품카테고리_ko, 상품카테고리_ko)
            상품타입_en = rev_type_map.get(상품타입_ko, 상품타입_ko)
            상품상세_en = from_korean_detail(상품상세_ko)

            new_doc = {
                "날짜": str(날짜),
                "시간": datetime.now().strftime("%H:%M:%S"),
                "상품카테고리": 상품카테고리_en,
                "상품타입": 상품타입_en,
                "상품상세": 상품상세_en,
                "수량": int(수량),
                "단가": float(단가),
                "수익": float(수익)
            }
            db.collection(SALES_COLLECTION).add(new_doc)

            init_stock, new_stock = deduct_stock(상품상세_en, int(수량))

            st.success(f"✅ 거래 저장 및 재고 차감 완료! (잔여: {new_stock}/{init_stock})")
            st.balloons()
            safe_rerun()

# ==============================================================
# 📈 경영 현황
# ==============================================================
elif menu == "경영 현황":
    st.header("📈 경영 현황 요약")

    if Path(PIPELINE_IMG).exists():
        st.image(PIPELINE_IMG, caption="ERP 파이프라인: 입고 → 재고 → 판매 → 발주 → 재입고")
    else:
        st.caption("💡 PIPELINE 이미지 경로를 설정하면 구조도가 표시됩니다.")

    total_rev = pd.to_numeric(df['수익'], errors='coerce').sum()
    total_tx = len(df)
    total_qty = pd.to_numeric(df['수량'], errors='coerce').sum()

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric-card'><p class='metric-title'>총 매출액</p><p class='metric-value'>{format_krw(total_rev)}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><p class='metric-title'>총 거래 수</p><p class='metric-value'>{int(total_tx):,} 건</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><p class='metric-title'>총 판매 수량</p><p class='metric-value'>{int(total_qty):,} 개</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    if not df.empty:
        try:
            top_cat = df.groupby('상품카테고리')['수익'].sum().sort_values(ascending=False).head(1)
            top_prod = df.groupby('상품타입')['수익'].sum().sort_values(ascending=False).head(1)
            st.info(f"🏆 가장 매출 높은 카테고리: **{top_cat.index[0]}** ({format_krw(top_cat.iloc[0])}) / 상품: **{top_prod.index[0]}**")
        except Exception:
            st.info("데이터가 충분하지 않아 상위 항목을 계산할 수 없습니다.")

        col4, col5 = st.columns(2)
        with col4:
            cat = df.groupby('상품카테고리')['수익'].sum().reset_index()
            fig_cat = px.pie(cat, values='수익', names='상품카테고리', title="카테고리별 매출 비중")
            st.plotly_chart(fig_cat, use_container_width=True)
        with col5:
            daily = df.groupby('날짜')['수익'].sum().reset_index()
            fig_trend = px.line(daily, x='날짜', y='수익', title="일자별 매출 추이")
            st.plotly_chart(fig_trend, use_container_width=True)

# ==============================================================
# 💹 매출 대시보드
# ==============================================================
elif menu == "매출 대시보드":
    st.header("💹 매출 대시보드")

    if df.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
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

# ==============================================================
# 📅 기간별 분석
# ==============================================================
elif menu == "기간별 분석":
    st.header("📅 기간별 매출 분석")

    if df.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        df_week = df.groupby('요일')['수익'].sum().reset_index()
        df_week['요일'] = pd.Categorical(df_week['요일'], categories=weekday_order_kr, ordered=True)
        df_week = df_week.sort_values('요일')

        df_hour = df.groupby('시')['수익'].sum().reset_index()
        df_month = df.groupby('월')['수익'].sum().reset_index()

        try:
            top_day = df_week.loc[df_week['수익'].idxmax()]
            top_hour = df_hour.loc[df_hour['수익'].idxmax()]
            top_month = df_month.loc[df_month['수익'].idxmax()]
        except Exception:
            top_day = {"요일": "-"}
            top_hour = {"시": "-"}
            top_month = {"월": "-"}

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-card'><p class='metric-title'>최고 매출 요일</p><p class='metric-value'>{top_day['요일']}</p></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><p class='metric-title'>최고 매출 시간</p><p class='metric-value'>{top_hour['시']}시</p></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><p class='metric-title'>최고 매출 달</p><p class='metric-value'>{top_month['월']}월</p></div>", unsafe_allow_html=True)

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

# ==============================================================
# 📦 재고 관리
# ==============================================================
elif menu == "재고 관리":
    st.header("📦 재고 관리 현황")

    df_inv = load_inventory_df()
    if df_inv.empty:
        st.info("현재 등록된 재고 데이터가 없습니다. '거래 추가' 시 자동 생성됩니다.")
    else:
        df_inv['재고비율'] = df_inv['현재재고'] / df_inv['초기재고']
        df_inv['상태'] = df_inv['재고비율'].apply(lambda r: "발주요망" if r <= REORDER_THRESHOLD_RATIO else "정상")
        low_stock = df_inv[df_inv['재고비율'] <= REORDER_THRESHOLD_RATIO]

        fig_stock = px.bar(
            df_inv.sort_values('재고비율'),
            x='상품상세', y='현재재고', color='재고비율',
            title="상품별 재고 현황 (현재/초기)",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_stock, use_container_width=True)

        show_cols = ['상품상세', '현재재고', '초기재고', '재고비율', '상태']
        st.dataframe(df_inv[show_cols], use_container_width=True)

        if not low_stock.empty:
            st.warning("⚠️ 일부 상품의 재고가 15% 이하입니다. 자동 발주가 권장됩니다.")
            st.dataframe(low_stock[show_cols], use_container_width=True)
            if st.button("🚚 자동 발주 생성"):
                for _, row in low_stock.iterrows():
                    need_qty = int(row['초기재고'] - row['현재재고'])
                    db.collection(ORDERS_COLLECTION).add({
                        "상품상세_en": row["상품상세_en"],
                        "발주수량": need_qty,
                        "발주일": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "기준": "15% 임계치"
                    })
                st.success("✅ 자동 발주가 생성되었습니다.")
        else:
            st.success("✅ 모든 상품의 재고가 안전 수준입니다.")

    st.markdown("---")

    # ---- SKU 파라미터 편집 ----
    st.markdown("### ⚙️ SKU 파라미터 편집 (리드타임/세이프티/목표일수/레시피g)")
    df_params = load_sku_params_df()
    if not df_inv.empty:
        missing = set(df_inv["상품상세_en"]) - set(df_params["sku_en"])
        if missing:
            add_df = pd.DataFrame({
                "sku_en": list(missing),
                "lead_time_days": 3,
                "safety_stock_units": 10,
                "target_days": 21,
                "grams_per_cup": 18.0,
                "expiry_days": 28
            })
            df_params = pd.concat([df_params, add_df], ignore_index=True)

    df_params["상품상세"] = df_params["sku_en"].apply(to_korean_detail)
    params_view = df_params[["상품상세","sku_en","lead_time_days","safety_stock_units","target_days","grams_per_cup","expiry_days"]]

    params_edit = st.data_editor(
        params_view,
        hide_index=True,
        column_config={
            "상품상세": st.column_config.Column("상품상세(표시)", disabled=True),
            "sku_en": st.column_config.Column("SKU(영문)", help="저장 키", disabled=True),
            "lead_time_days": st.column_config.NumberColumn("리드타임(일)", min_value=0, step=1),
            "safety_stock_units": st.column_config.NumberColumn("세이프티(단위)", min_value=0, step=1),
            "target_days": st.column_config.NumberColumn("목표일수", min_value=1, step=1),
            "grams_per_cup": st.column_config.NumberColumn("레시피(g/잔)", min_value=0.0, step=0.5),
            "expiry_days": st.column_config.NumberColumn("유통기한(일)", min_value=1, step=1),
        },
        use_container_width=True,
        key="sku_params_editor"
    )

    if st.button("💾 파라미터 저장"):
        saved = upsert_sku_params(params_edit.rename(columns={"sku_en":"sku_en"}))
        st.success(f"✅ {saved}건 저장 완료")
        safe_rerun()

    st.markdown("---")
    st.markdown("### 🧮 재주문점(ROP) 지표 & 권장 발주량")

    df_sales_for_calc = df.copy()
    if "상품상세" in df_sales_for_calc.columns:
        df_sales_for_calc["상품상세"] = df_sales_for_calc["상품상세"].astype(str)

    df_metrics = compute_replenishment_metrics(
        df_sales_for_calc, df_inv, params_edit.rename(columns={"sku_en":"sku_en"}), window_days=28
    )

    if df_metrics.empty:
        st.info("판매 데이터가 부족해 ROP 지표를 계산할 수 없습니다.")
    else:
        st.dataframe(df_metrics, use_container_width=True)

        low_mask = df_metrics["상태"].eq("발주요망") | (df_metrics["권장발주"] > 0)
        df_need = df_metrics[low_mask]
        if not df_need.empty:
            st.warning("⚠️ 아래 항목은 ROP 이하이거나 권장발주량이 있습니다.")
            st.dataframe(
                df_need[["상품상세","현재재고","ROP","권장발주","lead_time_days","safety_stock_units","target_days"]],
                use_container_width=True
            )

            if st.button("🧾 권장 발주 일괄 생성"):
                created = 0
                for _, r in df_need.iterrows():
                    qty = int(r["권장발주"])
                    if qty <= 0:
                        continue
                    db.collection(ORDERS_COLLECTION).add({
                        "상품상세_en": r["sku_en"],
                        "발주수량": qty,
                        "발주일": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "lead_time_days": int(r["lead_time_days"]),
                        "기준": "ROP/TargetDays"
                    })
                    created += 1
                st.success(f"✅ 발주 {created}건 생성")

    st.markdown("---")
    with st.expander("➕ 수동 입고(재고 추가)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            inv_options = sorted(df_inv['상품상세'].unique().tolist()) if not df_inv.empty else []
            sel_detail_ko = st.selectbox("상품상세(표시)", inv_options) if inv_options else None
        with c2:
            add_qty = st.number_input("입고 수량", min_value=1, value=10)
        with c3:
            submitted_in = st.button("입고 반영")
        if submitted_in and sel_detail_ko:
            sel_detail_en = from_korean_detail(sel_detail_ko)
            ref = ensure_inventory_doc(sel_detail_en)
            snap = ref.get()
            data = snap.to_dict()
            cur = int(data.get("현재재고", DEFAULT_INITIAL_STOCK))
            ref.update({"현재재고": cur + int(add_qty)})
            st.success("✅ 입고가 반영되었습니다.")
            safe_rerun()

# ==============================================================
# ✏️ 데이터 편집 (거래 수정/삭제 + 재고 일괄수정)
# ==============================================================
elif menu == "데이터 편집":
    st.header("✏️ 데이터 편집")
    tab1, tab2 = st.tabs(["거래 수정/삭제", "재고 일괄수정"])

    # ------------------ 거래 수정/삭제 ------------------
    with tab1:
        df_raw, df_view = load_sales_with_id()
        if df_view.empty:
            st.info("수정할 Firebase 거래 데이터가 없습니다. (CSV는 읽기 전용)")
        else:
            st.caption("💡 CSV 기반 행은 여기서 보이지 않습니다. (Firebase에 저장된 거래만 편집 가능)")
            edit_cols = ['_id', '날짜', '상품카테고리', '상품타입', '상품상세', '수량', '단가', '수익']
            df_edit = df_view[edit_cols].copy()

            edited = st.data_editor(
                df_edit,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "_id": st.column_config.Column("문서ID", help="읽기 전용", disabled=True),
                    "날짜": st.column_config.DateColumn("날짜"),
                    "수량": st.column_config.NumberColumn("수량", step=1, min_value=0),
                    "단가": st.column_config.NumberColumn("단가(원)", step=100.0, min_value=0.0),
                    "수익": st.column_config.NumberColumn("수익(원)", step=100.0, min_value=0.0),
                },
                use_container_width=True,
                key="trx_edit_table"
            )

            c1, c2, _ = st.columns([1,1,2])
            with c1:
                auto_rev = st.checkbox("수익 자동계산(수량×단가)", value=True)
            with c2:
                reflect_inv = st.checkbox(
                    "수정 시 재고 반영",
                    value=False,
                    help="수량 변경분만큼 재고를 증감합니다. (증가: 재고 차감, 감소: 재고 복원)"
                )

            if st.button("💾 변경 저장"):
                raw_by_id = {r['_id']: r for _, r in df_raw.iterrows()}
                changed = 0
                for _, row in edited.iterrows():
                    doc_id = row['_id']
                    if doc_id not in raw_by_id:
                        continue
                    orig = raw_by_id[doc_id]

                    cat_en = rev_category_map.get(row['상품카테고리'], row['상품카테고리'])
                    type_en = rev_type_map.get(row['상품타입'], row['상품타입'])
                    detail_en = from_korean_detail(row['상품상세'])

                    qty_new = int(row['수량']) if pd.notnull(row['수량']) else 0
                    unit_new = float(row['단가']) if pd.notnull(row['단가']) else 0.0
                    rev_new = float(qty_new * unit_new) if auto_rev else float(row['수익'] or 0)

                    patch = {}
                    try:
                        date_new_str = str(pd.to_datetime(row['날짜']).date())
                        date_old_str = str(pd.to_datetime(orig.get('날짜')).date())
                    except Exception:
                        date_new_str = str(row['날짜'])
                        date_old_str = str(orig.get('날짜'))
                    if date_new_str != date_old_str: patch['날짜'] = date_new_str
                    if cat_en != orig.get('상품카테고리'): patch['상품카테고리'] = cat_en
                    if type_en != orig.get('상품타입'): patch['상품타입'] = type_en
                    if detail_en != orig.get('상품상세'): patch['상품상세'] = detail_en
                    if qty_new != int(orig.get('수량', 0)): patch['수량'] = qty_new
                    if unit_new != float(orig.get('단가', 0)): patch['단가'] = unit_new
                    if rev_new != float(orig.get('수익', 0)): patch['수익'] = rev_new

                    if patch:
                        if reflect_inv and '수량' in patch:
                            qty_old = int(orig.get('수량', 0))
                            delta = qty_old - qty_new  # +면 재고 복원, -면 추가 차감
                            ref = ensure_inventory_doc(detail_en)
                            snap = ref.get()
                            cur = int(snap.to_dict().get("현재재고", DEFAULT_INITIAL_STOCK))
                            ref.update({"현재재고": cur + delta})

                        db.collection(SALES_COLLECTION).document(doc_id).update(patch)
                        changed += 1
                if changed:
                    st.success(f"✅ {changed}건 저장 완료")
                    safe_rerun()
                else:
                    st.info("변경된 내용이 없습니다.")

            st.markdown("---")
            del_ids = st.multiselect(
                "🗑️ 삭제할 거래 선택 (문서ID 기준)",
                options=df_view['_id'].tolist()
            )
            colx, _ = st.columns([1,3])
            with colx:
                restore_inv_on_delete = st.checkbox("삭제 시 재고 복원", value=True)
            if st.button("삭제 실행", type="primary", disabled=(len(del_ids) == 0)):
                for did in del_ids:
                    raw = df_raw[df_raw['_id'] == did].iloc[0].to_dict()
                    if restore_inv_on_delete:
                        ref = ensure_inventory_doc(raw.get('상품상세'))
                        snap = ref.get()
                        cur = int(snap.to_dict().get("현재재고", DEFAULT_INITIAL_STOCK))
                        ref.update({"현재재고": cur + int(raw.get('수량', 0))})
                    db.collection(SALES_COLLECTION).document(did).delete()
                st.success(f"✅ {len(del_ids)}건 삭제 완료")
                safe_rerun()

    # ------------------ 재고 일괄수정 ------------------
    with tab2:
        df_inv2 = load_inventory_df()
        if df_inv2.empty:
            st.info("재고 데이터가 없습니다. 판매 등록 시 자동 생성됩니다.")
        else:
            edit_cols = ['상품상세', '초기재고', '현재재고']
            inv_edited = st.data_editor(
                df_inv2[edit_cols],
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "상품상세": st.column_config.Column("상품상세(표시)", help="읽기 전용", disabled=True),
                    "초기재고": st.column_config.NumberColumn("초기재고", step=1, min_value=0),
                    "현재재고": st.column_config.NumberColumn("현재재고", step=1, min_value=0),
                },
                use_container_width=True,
                key="inv_edit_table"
            )
            if st.button("💾 재고 변경 저장"):
                changed = 0
                raw_docs = list(db.collection(INVENTORY_COLLECTION).stream())
                raw_by_en = {d.id: d.to_dict() for d in raw_docs}

                for _, row in inv_edited.iterrows():
                    detail_en = from_korean_detail(row['상품상세'])
                    orig = raw_by_en.get(detail_en, {})
                    patch = {}
                    if int(row['초기재고']) != int(orig.get('초기재고', DEFAULT_INITIAL_STOCK)):
                        patch['초기재고'] = int(row['초기재고'])
                    if int(row['현재재고']) != int(orig.get('현재재고', DEFAULT_INITIAL_STOCK)):
                        patch['현재재고'] = int(row['현재재고'])
                    if patch:
                        db.collection(INVENTORY_COLLECTION).document(detail_en).update(patch)
                        changed += 1
                if changed:
                    st.success(f"✅ 재고 {changed}건 저장 완료")
                    safe_rerun()
                else:
                    st.info("변경된 내용이 없습니다.")

# ==============================================================
# 📋 거래 내역
# ==============================================================
elif menu == "거래 내역":
    st.header("📋 전체 거래 내역")
    if df.empty:
        st.info("표시할 거래 데이터가 없습니다.")
    else:
        cols = ['날짜','상품카테고리','상품타입','상품상세','수량','단가','수익','요일','시']
        cols = [c for c in cols if c in df.columns]
        st.dataframe(df[cols].sort_values('날짜', ascending=False), use_container_width=True)

# ==============================================================
# ❓ 도움말
# ==============================================================
else:  # menu == "도움말"
    st.header("☕️ 커피 원두 재고관리 파이프라인 쉽게 이해하기")
    st.markdown("""
> **“커피 원두가 어떻게 들어오고, 얼마나 쓰이고, 언제 다시 주문돼야 하는지를 자동으로 관리하자!”**  
엑셀/감(勘) 대신 ERP가 자동으로 계산해줍니다.

### 파이프라인 한눈에 보기
| 단계 | 하는 일 | 예시 |
| --- | --- | --- |
| **1. 원두 입고** | 카페가 원두를 사와서 창고에 넣음 | “에티오피아 원두 10kg 입고” |
| **2. 재고 보관** | 원두 보관/신선도 확인 | 유통기한, 신선도 체크 |
| **3. 판매/소진** | 커피를 만들면 원두가 줄어듦 | “아메리카노 50잔 → 원두 2kg 소모” |
| **4. 재고 계산** | 남은 원두량 자동 계산 | “현재 8kg (20% 남음)” |
| **5. 발주 시점 알림** | 일정 이하로 떨어지면 알려줌 | “재고 15% 이하 → 발주 권장” |
| **6. 재주문 및 순환** | 새 원두 주문 → 다시 입고 | 순환 반복 |

### 왜 도움이 되나요?
- **데이터**로 발주 타이밍 결정
- **품절/폐기** 줄이고 **신선도** 유지
- **입고→재고→소진→발주** 전 과정이 연결됩니다.
    """)
