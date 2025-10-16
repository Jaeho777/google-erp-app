# ==============================================================
# ☕ Coffee ERP Dashboard — Company Showcase Edition (Tone-Down Blue)
#  - 재고 관리(자동 차감/임계치 경고/자동 발주 시뮬레이션)
#  - UI 한글화(이름 매핑 + 요일 한글 표시)
#  - 원본/Firestore는 영어 저장, 화면은 한글 표시(정/역매핑)
#  - 데이터 편집(거래 수정/삭제 + 재고 일괄수정)
#  - 도움말 탭 + SKU 파라미터(리드타임/세이프티/목표일수/레시피g) + ROP 지표/권장발주
#  - NEW: 레시피(BOM) 기반 자동 차감, uom(단위) 지원, 실사/오차율, 발주 ±범위 표시
# ==============================================================

import os
import re
import warnings
from math import ceil
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

import firebase_admin
from firebase_admin import credentials, firestore

# --- Pylance/static analyzer guards (no runtime effect) ---
items = []  # type: ignore
sold_qty = 0  # type: ignore
summary = []  # type: ignore

# ----------------------
# 0️⃣ 경로/상수 (팀원이 어디서 받아도 동작)
# ----------------------
BASE_DIR = Path(__file__).resolve().parent

# st.secrets 없을 때도 안전
try:
    SECRETS = dict(st.secrets)
except Exception:
    SECRETS = {}

def _resolve_path(val, default: Path) -> Path:
    """상대경로면 BASE_DIR 기준으로 절대경로로 변환"""
    if not val:
        return default
    p = Path(str(val))
    return p if p.is_absolute() else (BASE_DIR / p)

DATA_DIR   = _resolve_path(SECRETS.get("DATA_DIR")   or os.environ.get("ERP_DATA_DIR"),   BASE_DIR / "data")
ASSETS_DIR = _resolve_path(SECRETS.get("ASSETS_DIR") or os.environ.get("ERP_ASSETS_DIR"), BASE_DIR / "assets")
KEYS_DIR   = _resolve_path(SECRETS.get("KEYS_DIR")   or os.environ.get("ERP_KEYS_DIR"),   BASE_DIR / "keys")

CSV_PATH     = DATA_DIR / "Coffee Shop Sales.csv"
PIPELINE_IMG = ASSETS_DIR / "pipeline_diagram.png"
SA_FILE_PATH = KEYS_DIR / "serviceAccount.json"

SALES_COLLECTION      = "coffee_sales"
INVENTORY_COLLECTION  = "inventory"
ORDERS_COLLECTION     = "orders"
SKU_PARAMS_COLLECTION = "sku_params"

# ---- [NEW] 레시피/실사 컬렉션 ----
RECIPES_COLLECTION      = "recipes"        # 메뉴 SKU -> [ {ingredient_en, qty, uom, waste_pct} ]
STOCK_COUNTS_COLLECTION = "stock_counts"   # 실사 기록: {sku_en, count, uom, counted_at}
STOCK_MOVES_COLLECTION  = "stock_moves"    # 재고 이동 로그: 판매/시뮬/입고 등

USE_KRW_CONVERSION = False   # CSV가 USD면 True로
KRW_PER_USD = 1350

DEFAULT_INITIAL_STOCK   = 10000
REORDER_THRESHOLD_RATIO = 0.15  # 15%

# 디렉토리 준비
for p in (DATA_DIR, ASSETS_DIR, KEYS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------
# 0-1️⃣ Firebase 초기화 (Secrets → keys/ → GOOGLE_APPLICATION_CREDENTIALS)
# ----------------------
def init_firestore():
    if firebase_admin._apps:
        return firestore.client()

    # 1) st.secrets 딕셔너리(권장)
    svc_dict = SECRETS.get("firebase_service_account")
    if isinstance(svc_dict, dict) and svc_dict:
        cred = credentials.Certificate(svc_dict)
        firebase_admin.initialize_app(cred)
        return firestore.client()

    # 2) keys/serviceAccount.json
    if SA_FILE_PATH.exists():
        cred = credentials.Certificate(str(SA_FILE_PATH))
        firebase_admin.initialize_app(cred)
        return firestore.client()

    # 3) GOOGLE_APPLICATION_CREDENTIALS (파일 경로)
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and Path(gac).expanduser().exists():
        firebase_admin.initialize_app()
        return firestore.client()

    # 4) 전부 실패 → 명시적으로 에러
    st.error(
        "Firebase 자격증명을 찾을 수 없습니다.\n"
        "다음 중 하나를 설정하세요:\n"
        "• st.secrets['firebase_service_account'] 딕셔너리\n"
        "• keys/serviceAccount.json 파일\n"
        "• 환경변수 GOOGLE_APPLICATION_CREDENTIALS=자격증명파일경로"
    )
    st.stop()

db = init_firestore()

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

weekday_map = {"Monday": "월", "Tuesday": "화", "Wednesday": "수",
               "Thursday": "목", "Friday": "금", "Saturday": "토", "Sunday": "일"}
weekday_order_kr = ["월", "화", "수", "목", "금", "토", "일"]

def map_series(s: pd.Series, mapping: dict) -> pd.Series:
    return s.apply(lambda x: mapping.get(x, x))

# ----------------------
# ✅ UoM(단위) 유틸
# ----------------------
def normalize_uom(u: str | None) -> str:
    u = (u or "ea").strip().lower()
    if u in {"g", "gram", "grams", "그램", "kg", "킬로그램"}:
        return "g"
    if u in {"ml", "밀리리터", "l", "리터"}:
        return "ml"
    return "ea"

def convert_qty(qty: float, from_uom: str, to_uom: str) -> float:
    """kg↔g, l↔ml 변환. 그 외는 동일 단위로 간주.
    (입력은 g/ml/ea만 쓰는 것을 권장)
    """
    fu = normalize_uom(from_uom)
    tu = normalize_uom(to_uom)
    if fu == tu:
        return float(qty)
    # 밀도 없이 g↔ml 변환은 불가 → 단위 다르면 변환하지 않고 그대로 반환
    return float(qty)

def safe_float(x, default=0.0):
    """
    Robust float parser.
    - Returns `default` if x is None, empty, or NaN.
    - Does NOT cast `default` to float (so default can be None).
    """
    # Fast-path for explicit None
    if x is None:
        return default
    try:
        # Numbers (handle NaN)
        if isinstance(x, (int, float)):
            try:
                if pd.isna(x):
                    return default
            except Exception:
                pass
            return float(x)
        # Strings
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() in {"nan", "none"}:
                return default
            s = s.replace(",", "")
            return float(s)
        # Fallback: attempt cast
        return float(x)
    except Exception:
        return default

# ----------------------
# ✅ 날짜 파서: 명시 형식 우선 + 경고없는 폴백
# ----------------------
def parse_mixed_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    patterns = [
        (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
        (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d'),
        (r'^\d{2}/\d{2}/\d{4}$', '%m/%d/%Y'),
        (r'^\d{2}-\d{2}-\d{4}$', '%m-%d-%Y'),
        (r'^\d{4}\.\d{2}\.\d{2}$', '%Y.%m.%d'),
        (r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$', '%Y-%m-%d %H:%M:%S'),
        (r'^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}$', '%m/%d/%Y %H:%M:%S'),
    ]
    for pat, fmt in patterns:
        mask = s.str.match(pat)
        if mask.any():
            out.loc[mask] = pd.to_datetime(s.loc[mask], format=fmt, errors='coerce')
    remain = out.isna()
    if remain.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            out.loc[remain] = pd.to_datetime(s.loc[remain], errors='coerce')
    return out

# ----------------------
# 1️⃣ CSV 로드 (샘플 생성 없음)
# ----------------------
@st.cache_data(ttl=0)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"CSV를 찾을 수 없습니다. data/ 폴더에 'Coffee Shop Sales.csv'를 넣어주세요.\n(현재 찾는 경로: {path})")
        st.stop()
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

    # ✅ 경고 없는 날짜 파싱
    df['날짜'] = parse_mixed_dates(df['날짜'])

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

    # ✅ 경고 없는 날짜 파싱
    if '날짜' in df_fb.columns:
        df_fb['날짜'] = parse_mixed_dates(df_fb['날짜'])

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

    # ✅ 경고 없는 날짜 파싱
    if '날짜' in df_raw.columns:
        df_raw['날짜'] = parse_mixed_dates(df_raw['날짜'])

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

# ---- 단위 유틸 ----
VALID_UOM = {"ea","g","kg","ml","l"}
UOM_SYNONYM = {
    "piece":"ea","pcs":"ea","unit":"ea","units":"ea",
    "gram":"g","grams":"g","gms":"g",
    "kilogram":"kg","kilograms":"kg",
    "milliliter":"ml","millilitre":"ml","milliliters":"ml","millilitres":"ml",
    "liter":"l","litre":"l","liters":"l","litres":"l",
}

def normalize_uom(u: str) -> str:
    if not u:
        return "ea"
    s = str(u).strip().lower()
    s = UOM_SYNONYM.get(s, s)
    if s not in VALID_UOM:
        return s  # 알 수 없는 단위도 그대로 유지
    return s

def convert_qty(qty: float, from_uom: str, to_uom: str) -> float:
    try:
        q = float(qty)
    except Exception:
        return 0.0
    f = normalize_uom(from_uom)
    t = normalize_uom(to_uom)
    if f == t:
        return q
    # g <-> kg
    if f == "g" and t == "kg":
        return q / 1000.0
    if f == "kg" and t == "g":
        return q * 1000.0
    # ml <-> l
    if f == "ml" and t == "l":
        return q / 1000.0
    if f == "l" and t == "ml":
        return q * 1000.0
    # 상이한/비변환 단위는 그대로 반환 (상황에 따라 고도화 가능)
    return q

# (기존) 최소 보장 인벤토리 문서
# → NEW ensure_inventory_doc로 대체됨

def ensure_inventory_doc(product_detail_en: str, uom: str | None = None, is_ingredient: bool | None = None):
    """인벤토리 문서 보장 + uom/is_ingredient 관리"""
    ref = db.collection(INVENTORY_COLLECTION).document(product_detail_en)
    doc = ref.get()
    if not doc.exists:
        ref.set({
            "상품상세_en": product_detail_en,
            "초기재고": DEFAULT_INITIAL_STOCK,
            "현재재고": DEFAULT_INITIAL_STOCK,
            "uom": normalize_uom(uom or "ea"),
            "is_ingredient": bool(is_ingredient) if is_ingredient is not None else False,
        })
        return ref
    # 기존 문서 업데이트
    patch = {}
    data = doc.to_dict() or {}
    if "uom" not in data or uom:
        patch["uom"] = normalize_uom(uom or data.get("uom", "ea"))
    if is_ingredient is not None and data.get("is_ingredient") != bool(is_ingredient):
        patch["is_ingredient"] = bool(is_ingredient)
    if patch:
        ref.update(patch)
    return ref

# 재료 플래그 전용 헬퍼
def ensure_ingredient_sku(ingredient_en: str, uom: str = "ea"):
    return ensure_inventory_doc(ingredient_en, uom=uom, is_ingredient=True)
    

# (구버전) 단순 차감: 메뉴자체를 ea로 차감
def deduct_stock(product_detail_en: str, qty: int):
    ref = ensure_inventory_doc(product_detail_en)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else {}
    init_stock = int(data.get("초기재고", DEFAULT_INITIAL_STOCK))
    cur_stock = safe_float(data.get("현재재고", DEFAULT_INITIAL_STOCK))
    new_stock = max(cur_stock - int(qty), 0)
    ref.update({"현재재고": new_stock})
    return init_stock, new_stock

# ---- SKU 인벤토리 로드(uom 포함) ----
def load_inventory_df() -> pd.DataFrame:
    inv_docs = db.collection(INVENTORY_COLLECTION).stream()
    rows = []
    for d in inv_docs:
        doc = d.to_dict() or {}
        en  = doc.get("상품상세_en", d.id)
        ko  = to_korean_detail(en)
        rows.append({
            "상품상세_en": en,
            "상품상세": ko,
            "초기재고": doc.get("초기재고", DEFAULT_INITIAL_STOCK),
            "현재재고": doc.get("현재재고", DEFAULT_INITIAL_STOCK),
            "uom": normalize_uom(doc.get("uom", "ea")),
            "is_ingredient": bool(doc.get("is_ingredient", False)),
        })
    return pd.DataFrame(rows)


# ---- [NEW] 레시피 로딩/저장 ----

def get_all_recipe_ingredients() -> set:
    """레시피에 등장하는 모든 ingredient_en 집합"""
    try:
        docs = db.collection(RECIPES_COLLECTION).stream()
    except Exception:
        return set()
    S = set()
    for d in docs:
        items = (d.to_dict() or {}).get("items", []) or []
        for it in items:
            ing = str(it.get("ingredient_en", "")).strip()
            if ing:
                S.add(ing)
    return S

# ---- [NEW] 레시피 로딩/저장 ----
def load_recipe(menu_sku_en: str) -> list[dict]:
    doc = db.collection(RECIPES_COLLECTION).document(menu_sku_en).get()
    if not doc.exists:
        return []
    items = doc.to_dict().get("items", [])
    out = []
    for it in items:
        out.append({
            "ingredient_en": str(it.get("ingredient_en", "")).strip(),
            "qty": safe_float(it.get("qty", 0.0)),
            "uom": normalize_uom(it.get("uom", "ea")),
            "waste_pct": safe_float(it.get("waste_pct", 0.0))
        })
    return out

def upsert_recipe_item(menu_sku_en: str, ingredient_en: str, qty: float, uom: str = "ea", waste_pct: float = 0.0):
    """레시피 항목 1개 추가/갱신 (동일 ingredient_en 있으면 교체)"""
    ref = db.collection(RECIPES_COLLECTION).document(menu_sku_en)
    snap = ref.get()
    items = []
    if snap.exists:
        items = snap.to_dict().get("items", []) or []
        items = [it for it in items if str(it.get("ingredient_en")) != ingredient_en]
    items.append({
        "ingredient_en": ingredient_en,
        "qty": safe_float(qty),
        "uom": normalize_uom(uom),
        "waste_pct": safe_float(waste_pct),
    })
    ref.set({"menu_sku_en": menu_sku_en, "items": items})
    # 재료 플래그 보장
    ensure_ingredient_sku(ingredient_en, uom=uom)


def load_recipe(menu_sku_en: str) -> list[dict]:
    try:
        doc = db.collection(RECIPES_COLLECTION).document(menu_sku_en).get()
        if not doc.exists:
            return []
        data = doc.to_dict() or {}
        raw_items = data.get("items", []) or []
        out: list[dict] = []
        for it in raw_items:
            ing = str(it.get("ingredient_en", "")).strip()
            if not ing:
                continue
            qty = safe_float(it.get("qty"), 0.0)
            uom = normalize_uom(it.get("uom", "ea"))
            waste = safe_float(it.get("waste_pct", 0.0), 0.0)
            out.append({
                "ingredient_en": ing,
                "qty": qty,
                "uom": uom,
                "waste_pct": waste,
            })
        return out
    except Exception:
        return []


# ---- [NEW] 재고 차감(단위 인지) ----
def deduct_inventory(ingredient_en: str, qty: float, uom: str):
    """ingredient_en 인벤토리에서 qty(uom)만큼 차감"""
    ref = ensure_inventory_doc(ingredient_en, uom=uom)
    snap = ref.get()
    data = snap.to_dict() or {}
    cur = safe_float(data.get("현재재고", DEFAULT_INITIAL_STOCK))
    inv_uom = normalize_uom(data.get("uom", "ea"))
    use_qty = convert_qty(qty, from_uom=uom, to_uom=inv_uom)
    new_stock = max(cur - use_qty, 0.0)
    ref.update({"현재재고": new_stock})
    return cur, new_stock, inv_uom

# ---- [NEW] 레시피 기반 차감 ----
def apply_recipe_deduction(menu_sku_en: str, sold_qty: int, commit: bool = True) -> list[dict]:
    """
    메뉴 판매시: 레시피 있으면 재료별 차감, 없으면 메뉴 자체 차감.
    commit=False면 재고를 수정하지 않고 예상 after만 계산.
    반환: [{"ingredient_en", "used", "uom", "before", "after"}...]
    """
    items = load_recipe(menu_sku_en)
    summary: list[dict] = []

    if not items:
        # 레시피 없으면 메뉴 자체를 'ea'로 처리
        ref = ensure_inventory_doc(menu_sku_en, uom="ea")
        snap = ref.get()
        data = snap.to_dict() or {}
        before = safe_float(data.get("현재재고", DEFAULT_INITIAL_STOCK))
        inv_uom = normalize_uom(data.get("uom", "ea"))
        used = float(sold_qty)
        after = max(before - used, 0.0)
        if commit:
            ref.update({"현재재고": after})
        summary.append({"ingredient_en": menu_sku_en, "used": used, "uom": inv_uom, "before": before, "after": after})
        return summary

    for it in items:
        ing  = it["ingredient_en"]
        uom  = it["uom"]
        base = safe_float(it["qty"])
        w    = safe_float(it["waste_pct"]) / 100.0
        need = sold_qty * base * (1.0 + w)

        # 인벤토리 읽기
        ref = ensure_inventory_doc(ing, uom=uom)
        snap = ref.get()
        data = snap.to_dict() or {}
        before = safe_float(data.get("현재재고", DEFAULT_INITIAL_STOCK))
        inv_uom = normalize_uom(data.get("uom", "ea"))
        use_qty = convert_qty(need, from_uom=uom, to_uom=inv_uom)
        after = max(before - use_qty, 0.0)
        if commit:
            ref.update({"현재재고": after})
        summary.append({"ingredient_en": ing, "used": use_qty, "uom": inv_uom, "before": before, "after": after})
    return summary

def log_stock_move(menu_sku_en: str, qty: int, details: list[dict], move_type: str = "sale", note: str | None = None):
    """재고 이동 로그 기록 (상세는 ingredient 단위)."""
    try:
        db.collection(STOCK_MOVES_COLLECTION).add({
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": move_type,
            "menu_sku_en": menu_sku_en,
            "menu_sku_ko": to_korean_detail(menu_sku_en),
            "qty": int(qty),
            "details": details,
            "note": note or "",
        })
    except Exception:
        # 로깅 실패는 앱 동작에 영향 주지 않음
        pass
def adjust_inventory_by_recipe(menu_sku_en: str, diff_qty: int, move_type: str, note: str = "") -> None:
    """
    수량 증감(diff_qty)에 따라 레시피 기반으로 재고를 증/차감.
    diff_qty > 0 → 추가 차감(판매 증가), diff_qty < 0 → 복원(판매 감소/삭제)
    """
    if diff_qty == 0:
        return
    ded_summary = apply_recipe_deduction(menu_sku_en, int(diff_qty), commit=True)
    log_stock_move(menu_sku_en, int(diff_qty), ded_summary, move_type=move_type, note=note)

# ---------- SKU 파라미터 로더 (단일 정의; 메뉴 분기 시작 전) ----------
def load_sku_params_df() -> pd.DataFrame:
    """Firestore 'sku_params' 컬렉션을 DataFrame으로 로드하고 기본값/타입을 보정."""
    try:
        docs = db.collection(SKU_PARAMS_COLLECTION).stream()
    except Exception:
        docs = []

    rows = []
    for d in docs:
        item = d.to_dict() or {}
        # 문서 id도 보존
        try:
            item["_id"] = d.id
        except Exception:
            item["_id"] = item.get("_id", "")
        rows.append(item)

    dfp = pd.DataFrame(rows)
    if dfp.empty:
        dfp = pd.DataFrame(columns=[
            "_id","sku_en","lead_time_days","safety_stock_units","target_days","grams_per_cup","expiry_days"
        ])

    defaults = {
        "lead_time_days": 3,
        "safety_stock_units": 10,
        "target_days": 21,
        "grams_per_cup": 18.0,
        "expiry_days": 28,
    }
    for col, default in defaults.items():
        if col not in dfp.columns:
            dfp[col] = default
        else:
            dfp[col] = pd.to_numeric(dfp[col], errors="coerce").fillna(default)

    return dfp


# ---------- 재료 ROP/권장발주 계산 (단일 정의; 메뉴 분기 시작 전) ----------
def compute_ingredient_metrics_for_menu(
    menu_sku_en: str,
    df_all_sales: pd.DataFrame,
    df_inv: pd.DataFrame,
    df_params: pd.DataFrame,
    window_days: int = 28
) -> pd.DataFrame:
    """
    특정 메뉴의 레시피와 최근 판매량(윈도우) 기반으로 재료별
    일평균소진/커버일수/ROP/권장발주를 계산.
    반환 컬럼:
      ["상품상세","sku_en","현재재고","초기재고","uom","최근소진합","일평균소진","커버일수",
       "lead_time_days","safety_stock_units","target_days","ROP","권장발주","상태"]
    """
    items = load_recipe(menu_sku_en)
    if not items:
        return pd.DataFrame()

    # 판매 윈도우 추출
    if "날짜" in df_all_sales.columns and pd.api.types.is_datetime64_any_dtype(df_all_sales["날짜"]):
        max_day = df_all_sales["날짜"].max()
        min_day = max_day - pd.Timedelta(days=window_days - 1)
        df_win = df_all_sales[(df_all_sales["날짜"] >= min_day) & (df_all_sales["날짜"] <= max_day)].copy()
    else:
        df_win = df_all_sales.copy()

    # 메뉴 영어키 매핑
    df_win = df_win.copy()
    if "상품상세" in df_win.columns:
        df_win["sku_en"] = df_win["상품상세"].apply(from_korean_detail)
    else:
        df_win["sku_en"] = ""

    # 대상 메뉴 판매수량 합계
    df_win["수량"] = pd.to_numeric(df_win.get("수량", 0), errors="coerce").fillna(0)
    sold_sum = df_win.loc[df_win["sku_en"].eq(menu_sku_en), "수량"].sum()

    # 재료별 최근소진합(레시피×판매)
    rows = []
    for it in items:
        ing  = it.get("ingredient_en", "")
        base = safe_float(it.get("qty", 0), 0)
        w    = safe_float(it.get("waste_pct", 0), 0) / 100.0
        need = sold_sum * base * (1 + w)
        rows.append({"sku_en": ing, "최근소진합": need, "uom_src": it.get("uom", "ea")})
    use_df = pd.DataFrame(rows)

    # 인벤토리 결합 (레시피 재료만)
    base = df_inv.rename(columns={"상품상세_en": "sku_en"}).copy()
    base = base.merge(use_df, on="sku_en", how="right")

    # 단위 변환: recipe uom -> inventory uom
    base["uom"] = base["uom"].apply(normalize_uom)
    base["uom_src"] = base["uom_src"].apply(normalize_uom)
    base["최근소진합"] = base.apply(
        lambda r: convert_qty(r["최근소진합"], from_uom=r["uom_src"], to_uom=r["uom"]),
        axis=1
    )

    days = max(window_days, 1)
    base["일평균소진"] = (base["최근소진합"] / days).round(3)
    base.loc[base["일평균소진"].eq(0), "일평균소진"] = 0.01  # 0 division 방지
    base["커버일수"] = (base["현재재고"] / base["일평균소진"]).round(1)

    # 파라미터 결합 + 기본값
    base = base.merge(df_params, on="sku_en", how="left")
    base["lead_time_days"] = pd.to_numeric(base.get("lead_time_days", 3), errors="coerce").fillna(3).astype(int)
    base["safety_stock_units"] = pd.to_numeric(base.get("safety_stock_units", 10), errors="coerce").fillna(10).astype(int)
    base["target_days"] = pd.to_numeric(base.get("target_days", 21), errors="coerce").fillna(21).astype(int)

    # ROP/권장발주/상태
    base["ROP"] = (base["일평균소진"] * base["lead_time_days"] + base["safety_stock_units"]).round(0).astype(int)
    base["권장발주"] = ((base["target_days"] * base["일평균소진"]) - base["현재재고"]).apply(lambda x: max(int(ceil(x)), 0))
    base["상태"] = base.apply(lambda r: "발주요망" if r["현재재고"] <= r["ROP"] else "정상", axis=1)

    # 표시명
    base["상품상세"] = base["sku_en"].apply(to_korean_detail)

    cols = ["상품상세","sku_en","현재재고","초기재고","uom","최근소진합","일평균소진","커버일수",
            "lead_time_days","safety_stock_units","target_days","ROP","권장발주","상태"]
    for c in cols:
        if c not in base.columns:
            base[c] = None

    return base[cols].sort_values(["상태","커버일수"])


# 공통 width 설정
W = "stretch"

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

            # ✅ 레시피 자동 보장 후, 레시피 기반 차감(없으면 메뉴 자체 차감)
            try:
                # 기본 레시피 자동 보장
                _auto_defaults = {
                    "Latte": [
                        {"ingredient_en": "Espresso Roast", "qty": 18, "uom": "g", "waste_pct": 0},
                        {"ingredient_en": "Milk", "qty": 300, "uom": "ml", "waste_pct": 5},
                        {"ingredient_en": "Regular syrup", "qty": 5, "uom": "ml", "waste_pct": 0},
                    ]
                }
                doc = db.collection(RECIPES_COLLECTION).document(상품상세_en).get()
                if not doc.exists and 상품상세_en in _auto_defaults:
                    db.collection(RECIPES_COLLECTION).document(상품상세_en).set({
                        "menu_sku_en": 상품상세_en,
                        "items": _auto_defaults[상품상세_en]
                    })
                    for it in _auto_defaults[상품상세_en]:
                        ensure_inventory_doc(it["ingredient_en"], uom=it["uom"])
            except Exception:
                pass

            ded_summary = apply_recipe_deduction(상품상세_en, int(수량), commit=True)
            # 이동 로그 기록
            log_stock_move(상품상세_en, int(수량), ded_summary, move_type="sale")
            msg_lines = []
            for s in ded_summary:
                msg_lines.append(f"- {to_korean_detail(s['ingredient_en'])}: {s['used']:.2f}{s['uom']} → 잔여 {s['after']:.2f}/{s['before']:.2f}")
            st.success("✅ 거래 저장 및 재고 차감 완료!\n" + "\n".join(msg_lines))
            st.balloons()
            safe_rerun()

# ==============================================================
# 📈 경영 현황
# ==============================================================
elif menu == "경영 현황":
    st.header("📈 경영 현황 요약")

    if PIPELINE_IMG.exists():
        st.image(str(PIPELINE_IMG), caption="ERP 파이프라인: 입고 → 재고 → 판매 → 발주 → 재입고")
    else:
        st.caption("")

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
            st.plotly_chart(fig_cat, width=W)
        with col5:
            daily = df.groupby('날짜')['수익'].sum().reset_index()
            fig_trend = px.line(daily, x='날짜', y='수익', title="일자별 매출 추이")
            st.plotly_chart(fig_trend, width=W)

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
            st.plotly_chart(fig_month, width=W)

        with col2:
            cat_sales = df.groupby('상품카테고리')['수익'].sum().reset_index()
            fig_cat2 = px.bar(cat_sales, x='상품카테고리', y='수익', title="상품 카테고리별 매출")
            st.plotly_chart(fig_cat2, width=W)

        prod_sales = df.groupby(['상품타입','상품상세'])['수익'].sum().reset_index()
        fig_sun = px.sunburst(prod_sales, path=['상품타입','상품상세'], values='수익', title="상품 구조별 매출")
        st.plotly_chart(fig_sun, width=W)

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
            st.plotly_chart(fig_w, width=W)
        with colB:
            fig_h = px.line(df_hour, x='시', y='수익', title="시간대별 매출")
            st.plotly_chart(fig_h, width=W)
        fig_m = px.bar(df_month, x='월', y='수익', title="월별 매출")
        st.plotly_chart(fig_m, width=W)

elif menu == "재고 관리":

    st.header("📦 재고 관리 현황")

    # ===== 재고 초기화 =====
    with st.expander("🧹 재고 데이터 초기화 기능"):
        st.warning("⚠️ 모든 재고의 '초기재고'와 '현재재고'를 기본값(10000)으로 되돌립니다. 복구 불가하니 주의하세요.")
        if st.button("재고 데이터 초기화 실행", type="primary"):
            try:
                inv_docs = db.collection(INVENTORY_COLLECTION).stream()
                count = 0
                for d in inv_docs:
                    ref = db.collection(INVENTORY_COLLECTION).document(d.id)
                    ref.update({
                        "초기재고": DEFAULT_INITIAL_STOCK,
                        "현재재고": DEFAULT_INITIAL_STOCK
                    })
                    count += 1
                st.success(f"✅ 총 {count}개의 재고 문서를 기본값({DEFAULT_INITIAL_STOCK})으로 초기화했습니다.")
                st.balloons()
                safe_rerun()
            except Exception as e:
                st.error(f"초기화 중 오류 발생: {e}")


    # ===== 재료(Ingredient) 뷰 =====
    df_inv = load_inventory_df()
    if df_inv.empty:
        st.info("현재 등록된 재고 데이터가 없습니다. '거래 추가' 또는 아래 시드 기능을 사용하세요.")
    else:
        st.subheader("🥣 재료 재고 (레시피 연결 기반)")
        ing_set = get_all_recipe_ingredients()
        df_ing = df_inv[df_inv["is_ingredient"] | df_inv["상품상세_en"].isin(ing_set)].copy()
        if df_ing.empty:
            st.info("아직 레시피와 연결된 재료가 없습니다. 아래 '라떼 연결 마법사'를 먼저 실행해 보세요.")
        else:
            df_ing['재고비율'] = df_ing['현재재고'] / df_ing['초기재고']
            df_ing['상태'] = df_ing['재고비율'].apply(lambda r: "발주요망" if r <= REORDER_THRESHOLD_RATIO else "정상")
            low_ing = df_ing[df_ing['재고비율'] <= REORDER_THRESHOLD_RATIO]

            fig_ing = px.bar(
                df_ing.sort_values('재고비율'),
                x='상품상세', y='현재재고', color='재고비율',
                title="재료별 재고 현황",
            )
            st.plotly_chart(fig_ing, width=W)
            st.dataframe(df_ing[['상품상세','현재재고','초기재고','uom','재고비율','상태']], width=W)

            if not low_ing.empty:
                st.warning("⚠️ 일부 재료 재고가 15% 이하입니다. 발주를 고려하세요.")

    st.markdown("---")

    # ===== 라떼 연결 마법사 =====
    with st.expander("🔗 라떼 연결(한 메뉴 POC)"):
        st.caption("라떼 1잔 = Espresso Roast 18g + Milk 300ml + Regular syrup 5ml (+Milk waste 5%)")
        if st.button("라떼 레시피 생성/덮어쓰기"):
            latte_items = [
                {"ingredient_en": "Espresso Roast", "qty": 18, "uom": "g", "waste_pct": 0},
                {"ingredient_en": "Milk",           "qty": 300, "uom": "ml", "waste_pct": 5},
                {"ingredient_en": "Regular syrup",   "qty": 5,   "uom": "ml", "waste_pct": 0},
            ]
            db.collection(RECIPES_COLLECTION).document("Latte").set({
                "menu_sku_en": "Latte",
                "items": latte_items
            })
            for it in latte_items:
                ensure_ingredient_sku(it["ingredient_en"], uom=it["uom"])  # 재료 플래그 + uom 보장
            st.success("✅ 라떼 레시피가 생성되었습니다.")

        c1, c2, c3 = st.columns(3)
        with c1:
            milk_seed = st.number_input("우유 초기/현재(ml)", min_value=0, value=5000, step=100)
        with c2:
            bean_seed = st.number_input("에스프레소 로스트 초기/현재(g)", min_value=0, value=2000, step=50)
        with c3:
            syrup_seed = st.number_input("레귤러 시럽 초기/현재(ml)", min_value=0, value=1000, step=10)
        if st.button("시드 재고 반영"):
            for en, uom, qty in [
                ("Milk","ml", milk_seed),
                ("Espresso Roast","g", bean_seed),
                ("Regular syrup","ml", syrup_seed),
            ]:
                ref = ensure_ingredient_sku(en, uom=uom)
                ref.update({"초기재고": float(qty), "현재재고": float(qty)})
            st.success("✅ 시드 재고를 반영했습니다.")

    st.markdown("---")

    # ===== 재료 ROP (라떼 기준) =====
    st.markdown("### 🧮 재료 ROP (라떼 기준)")
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

    df_sales_for_calc = df.copy()
    if "상품상세" in df_sales_for_calc.columns:
        df_sales_for_calc["상품상세"] = df_sales_for_calc["상품상세"].astype(str)

    df_ing_metrics = compute_ingredient_metrics_for_menu(
        "Latte", df_sales_for_calc, df_inv, df_params, window_days=28
    )
    if df_ing_metrics.empty:
        st.info("라떼 레시피가 없거나 최근 라떼 판매가 없어 재료 ROP를 계산할 수 없습니다. 위의 마법사와 '거래 추가'를 이용해 테스트해 보세요.")
    else:
        st.dataframe(df_ing_metrics, width=W)
        need_rows = df_ing_metrics[(df_ing_metrics["상태"].eq("발주요망")) | (df_ing_metrics["권장발주"] > 0)]
        if not need_rows.empty:
            st.warning("⚠️ 아래 재료는 ROP 이하이거나 권장발주량이 존재합니다.")
            st.dataframe(need_rows[["상품상세","현재재고","uom","ROP","권장발주","lead_time_days","safety_stock_units","target_days"]], width=W)

    st.markdown("---")

    # ===== 최근 재고 이동 로그 =====
    st.markdown("### 🧾 최근 재고 이동")
    try:
        q = db.collection(STOCK_MOVES_COLLECTION).order_by("ts", direction=firestore.Query.DESCENDING).limit(50).stream()
        docs = [d.to_dict() for d in q]
    except Exception:
        docs = [d.to_dict() for d in db.collection(STOCK_MOVES_COLLECTION).stream()]
        docs.sort(key=lambda x: x.get("ts",""), reverse=True)
    move_rows = []
    for m in docs:
        base = {
            "시각": m.get("ts",""),
            "유형": m.get("type",""),
            "메뉴": to_korean_detail(m.get("menu_sku_en","")),
            "수량": m.get("qty",0),
            "비고": m.get("note",""),
        }
        for det in (m.get("details", []) or []):
            row = base | {
                "재료": to_korean_detail(det.get("ingredient_en","")),
                "사용량": round(float(det.get("used",0.0)),2),
                "단위": det.get("uom",""),
                "전": round(float(det.get("before",0.0)),2),
                "후": round(float(det.get("after",0.0)),2),
            }
            move_rows.append(row)
    if move_rows:
        kw = st.text_input("필터(메뉴/재료 포함)", "")
        df_moves = pd.DataFrame(move_rows)
        if kw:
            df_moves = df_moves[df_moves.apply(lambda r: kw in str(r.values), axis=1)]
        st.dataframe(df_moves, hide_index=True, width=W)
    else:
        st.caption("최근 이동 로그가 없습니다.")

    st.markdown("---")
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
                width=W,
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
                            diff = qty_new - int(orig.get('수량', 0))
                            adjust_inventory_by_recipe(detail_en, diff, move_type="edit_adjust", note=str(doc_id))

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
                        adjust_inventory_by_recipe(raw.get('상품상세'), -int(raw.get('수량', 0)), move_type="delete_restore", note=str(did))
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
                width=W,
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
        st.dataframe(df[cols].sort_values('날짜', ascending=False), width=W)

# ==============================================================
# ❓ 도움말
# ==============================================================
else:  # menu == "도움말"
    st.header("☕️ 커피 원두 재고관리 파이프라인 쉽게 이해하기")
    st.markdown("""
> **“커피 원두가 어떻게 들어오고, 얼마나 쓰이고, 언제 다시 주문돼야 하는지를 자동으로 관리하자!”**  
엑셀 대신 ERP가 자동으로 계산해줍니다.

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
