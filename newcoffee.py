# ==============================================================
# ☕ Coffee ERP Dashboard — Company Showcase Edition (Tone-Down Blue)
# (기존 주석 생략)
# ==============================================================

import os
import json
import re
import warnings
from math import ceil
from pathlib import Path
from datetime import datetime
import time # #[AI/ML 통합 추가] (Mock 응답용)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

import firebase_admin
from firebase_admin import credentials, firestore

# === [AI/ML 통합 추가] ===
# SPRINT 1 (AI 비서) 및 SPRINT 2 (수요 예측) 라이브러리
try:
    import openai
    import time
    from prophet import Prophet
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error
except ImportError:
    st.error("""
    AI/ML 기능을 위한 라이브러리가 부족합니다.
    터미널에서 'pip install openai prophet scikit-learn'를 실행해주세요.
    """)
    st.stop()
# === [AI/ML 통합 추가] ===
# === [빈틈 수정] 누락된 핵심 도우미 함수 (format_krw, safe_rerun) ===
def format_krw(x: float) -> str:
    """숫자를 원화 형식의 문자열로 변환합니다."""
    try:
        return f"{x:,.0f} 원"
    except Exception:
        return "-"

def safe_rerun():
    """Streamlit 버전에 맞춰 앱을 새로고침합니다."""
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception as e:
        # (새로고침 오류는 무시)
        pass
# ===================================================================


st.set_page_config(page_title="☕ Coffee ERP Dashboard", layout="wide")


# (init_firebase 함수 원본)
def init_firebase():
    try:
        if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
            cred_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            cred = credentials.Certificate(cred_info)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            return firestore.client(), "success"
        else:
            return None, "no_env"
    except Exception as e:
        return None, f"error: {e}"

# ✅ 함수 호출 후 UI 표시 분리
db, fb_status = init_firebase()

# --- Pylance/static analyzer guards (no runtime effect) ---
items = []  # type: ignore
sold_qty = 0  # type: ignore
summary = []  # type: ignore

# ----------------------
# 0️⃣ 경로/상수 (팀원이 어디서 받아도 동작)
# (원본 코드 생략)
# ----------------------
BASE_DIR = Path(__file__).resolve().parent

try:
    SECRETS = dict(st.secrets)
except Exception:
    SECRETS = {}

def _resolve_path(val, default: Path) -> Path:
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

RECIPES_COLLECTION      = "recipes"
STOCK_COUNTS_COLLECTION = "stock_counts"
STOCK_MOVES_COLLECTION  = "stock_moves"

USE_KRW_CONVERSION = False
KRW_PER_USD = 1350
DEFAULT_INITIAL_STOCK   = 10000
REORDER_THRESHOLD_RATIO = 0.15

for p in (DATA_DIR, ASSETS_DIR, KEYS_DIR):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------
# 0-1️⃣ Firebase 초기화 (Secrets → keys/ → GOOGLE_APPLICATION_CREDENTIALS)
# (원본 코드 생략)
# ----------------------
@st.cache_resource
def init_firestore():
    """Firebase 인증 및 Firestore 클라이언트 초기화 (중복 호출 방지 + 캐시 적용)"""
    if firebase_admin._apps:
        return firestore.client()
    svc_dict = SECRETS.get("firebase_service_account")
    if isinstance(svc_dict, dict) and svc_dict:
        cred = credentials.Certificate(svc_dict)
        firebase_admin.initialize_app(cred)
        return firestore.client()
    if SA_FILE_PATH.exists():
        cred = credentials.Certificate(str(SA_FILE_PATH))
        firebase_admin.initialize_app(cred)
        return firestore.client()
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and Path(gac).expanduser().exists():
        firebase_admin.initialize_app()
        return firestore.client()
    st.error(
        "Firebase 자격증명을 찾을 수 없습니다.\n"
        "다음 중 하나를 설정하세요:\n"
        "• st.secrets['firebase_service_account'] 딕셔너리\n"
        "• keys/serviceAccount.json 파일\n"
        "• 환경변수 GOOGLE_APPLICATION_CREDENTIALS=자격증명파일경로"
    )
    st.stop()


db = init_firestore()

# === [AI/ML 통합 추가] ===
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
# === [AI/ML 통합 추가] ===

# ----------------------
# 0-2️⃣ UI/스타일
# (원본 코드 생략)
# ----------------------
pio.templates.default = "plotly_white"
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Blues"

st.markdown("""
    <style>
    /* ... (기존 스타일 정의) ... */
    </style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="dashboard-header">
  <h1>☕ Coffee ERP Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# ----------------------
# 0-3️⃣ 한글 매핑 테이블
# (원본 코드 생략)
# ----------------------
category_map = {
    "Coffee": "커피", "Tea": "차", "Bakery": "베이커리",
    # ... (기존 매핑) ...
    "Packaged Chocolate": "포장 초콜릿",
}
rev_category_map = {v: k for k, v in category_map.items()}
rev_category_map.update({
    "베이커리": "Bakery",
    # ... (기존 역 매핑) ...
    "커피": "Coffee",
})

type_map = {
    "Barista Espresso": "바리스타 에스프레소",
    # ... (기존 매핑) ...
    "Premium Brewed Coffee": "프리미엄 브루드 커피",
}
rev_type_map = {v: k for k, v in type_map.items()}

SIZE_SUFFIX_MAP = {"Lg": "라지", "Rg": "레귤러", "Sm": "스몰"}
REV_SIZE_SUFFIX_MAP = {"라지": "Lg", "레귤러": "Rg", "스몰": "Sm"}

detail_base_map = {
    "Almond Croissant": "아몬드 크루아상",
    # ... (기존 매핑) ...
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
# (원본 코드 생략)
# ----------------------
def normalize_uom(u: str | None) -> str:
    u = (u or "ea").strip().lower()
    if u in {"g", "gram", "grams", "그램", "kg", "킬로그램"}:
        return "g"
    if u in {"ml", "밀리리터", "l", "리터"}:
        return "ml"
    return "ea"

def convert_qty(qty: float, from_uom: str, to_uom: str) -> float:
    fu = normalize_uom(from_uom)
    tu = normalize_uom(to_uom)
    if fu == tu:
        return float(qty)
    return float(qty)

def safe_float(x, default=0.0):
    if x is None:
        return default
    try:
        if isinstance(x, (int, float)):
            try:
                if pd.isna(x):
                    return default
            except Exception:
                pass
            return float(x)
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() in {"nan", "none"}:
                return default
            s = s.replace(",", "")
            return float(s)
        return float(x)
    except Exception:
        return default

# ----------------------
# ✅ 날짜 파서: 명시 형식 우선 + 경고없는 폴백
# (원본 코드 생략)
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
# (원본 코드 생략)
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
# (원본 코드 생략)
# ----------------------
def load_sales_from_firestore() -> pd.DataFrame:
    docs = db.collection(SALES_COLLECTION).stream()
    data = [d.to_dict() for d in docs]
    df_fb = pd.DataFrame(data)
    if df_fb.empty:
        return df_fb
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

# ==============================================================
# === [L4 마스터 데이터 로딩 블록] ===
# (순서 문제 해결: '정의'를 '호출'보다 앞으로 이동)
# ==============================================================

# --- 1. 헬퍼 함수 정의 (정의 1: Inventory) ---
@st.cache_data(ttl=60)
def load_inventory_df() -> pd.DataFrame:
    inv_docs = db.collection(INVENTORY_COLLECTION).stream()
    rows = []
    for d in inv_docs:
        doc = d.to_dict() or {}
        en = doc.get("상품상세_en", d.id)
        ko = to_korean_detail(en)
        
        # [L4] 원가 정보 로드
        cost_unit_size = safe_float(doc.get("cost_unit_size", 1.0), 1.0)
        cost_per_unit = safe_float(doc.get("cost_per_unit", 0.0), 0.0)
        
        # 1g/1ml/1ea당 원가 계산 (0으로 나누기 방지)
        unit_cost = cost_per_unit / cost_unit_size if cost_unit_size > 0 else 0.0
        
        rows.append({
            "상품상세_en": en,
            "상품상세": ko,
            "초기재고": doc.get("초기재고", DEFAULT_INITIAL_STOCK),
            "현재재고": doc.get("현재재고", DEFAULT_INITIAL_STOCK),
            "uom": normalize_uom(doc.get("uom", "ea")),
            "is_ingredient": bool(doc.get("is_ingredient", False)),
            
            # [L4] 원가 컬럼 추가
            "cost_unit_size": cost_unit_size, # 매입 단위 (e.g., 1000)
            "cost_per_unit": cost_per_unit,  # 매입가 (e.g., 30000)
            "unit_cost": unit_cost           # 1g/ml/ea당 원가 (e.g., 30)
        })
    
    # === [빈틈 수정] inventory가 비어있어도 컬럼은 유지 ===
    df = pd.DataFrame(rows, columns=[
        "상품상세_en", "상품상세", "초기재고", "현재재고", "uom", "is_ingredient",
        "cost_unit_size", "cost_per_unit", "unit_cost" # [L4]
    ])
    return df

# --- 2. 헬퍼 함수 정의 (정의 2: SKU Params) ---
@st.cache_data(ttl=60)
def load_sku_params() -> pd.DataFrame:
    try:
        docs = db.collection(SKU_PARAMS_COLLECTION).stream()
    except Exception:
        docs = []
    rows = []
    for d in docs:
        item = d.to_dict() or {}
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

# --- 3. 헬퍼 함수 정의 (정의 3: Ensure Inventory Doc) ---
def ensure_inventory_doc(product_detail_en: str, uom: str = "ea", is_ingredient: bool = False):
    ref = db.collection(INVENTORY_COLLECTION).document(product_detail_en)
    snap = ref.get()
    if snap.exists:
        data = snap.to_dict() or {}
        patch = {}
        if normalize_uom(data.get("uom")) != normalize_uom(uom):
            patch["uom"] = normalize_uom(uom)
        if bool(data.get("is_ingredient", False)) != bool(is_ingredient):
            patch["is_ingredient"] = bool(is_ingredient)
        if patch:
            ref.update(patch)
        return ref
    else:
        ref.set({
            "상품상세_en": product_detail_en,
            "초기재고": DEFAULT_INITIAL_STOCK,
            "현재재고": DEFAULT_INITIAL_STOCK,
            "uom": normalize_uom(uom),
            "is_ingredient": bool(is_ingredient),
            # [L4] 원가 기본값
            "cost_unit_size": 1.0,
            "cost_per_unit": 0.0,
            "unit_cost": 0.0,
        })
        return ref

def ensure_ingredient_sku(ingredient_en: str, uom: str = "ea"):
    return ensure_inventory_doc(ingredient_en, uom=uom, is_ingredient=True)


# --- 4. 메인 데이터 로딩 함수 (호출 1) ---
@st.cache_data(ttl=60)
def load_all_core_data():
    """
    [L4 수정] 앱 실행 시 모든 핵심 데이터를 로드합니다.
    (이제 이 함수가 호출되어도, 필요한 함수들이 '위에' 정의되어 있습니다.)
    """
    # 1. Sales (df)
    df = pd.concat([df_csv, df_fb], ignore_index=True)
    if '요일' in df.columns:
        df['요일'] = map_series(df['요일'], weekday_map)
    if '상품카테고리' in df.columns:
        df['상품카테고리'] = map_series(df['상품카테고리'], category_map)
    if '상품타입' in df.columns:
        df['상품타입'] = map_series(df['상품타입'], type_map)
    if '상품상세' in df.columns:
        df['상품상세'] = df['상품상세'].apply(to_korean_detail)
    
    # 2. Inventory (df_inv) - [L4] 원가 계산이 포함된 함수로 호출
    df_inv = load_inventory_df() 
    
    # 3. Recipes (recipes)
    recipes = {}
    try:
        recipe_docs = db.collection(RECIPES_COLLECTION).stream()
        for d in recipe_docs:
            data = d.to_dict()
            if data and "ingredients" in data:
                recipes[d.id] = data["ingredients"]
    except Exception as e:
        st.error(f"레시피 로드 실패: {e}")
        
    # 4. Params (df_params)
    df_params = load_sku_params()
    
    return df, df_inv, recipes, df_params

# --- 5. 메인 데이터 로드 '실행' ---
try:
    #data_load_state = st.info("모든 핵심 데이터(판매, 재고, 레시피) 로드 중... ⏳")
    df, df_inv, RECIPES, df_params = load_all_core_data()
    #data_load_state.success("✅ 모든 데이터 로드 완료!")
except Exception as e:
    #data_load_state.error(f"데이터 로드 실패: {e}")
    st.stop()
    

# --- 6. 원가(COGS) 계산 함수 (정의 4) ---
@st.cache_data(ttl=600)
def calculate_menu_cogs(df_inv: pd.DataFrame, recipes: dict) -> dict:
    """
    (L4) 'df_inv'의 'unit_cost'와 'recipes'를 사용해
    모든 메뉴의 COGS(매출 원가)를 계산합니다.
    """
    if 'unit_cost' not in df_inv.columns:
        st.error("calculate_menu_cogs: df_inv에 'unit_cost' 컬럼이 없습니다.")
        return {}
        
    # 1. 재료 원가 맵 생성 (sku_en -> unit_cost)
    ingredient_costs = df_inv[df_inv['is_ingredient'] == True].set_index('상품상세_en')['unit_cost'].to_dict()
    
    menu_cogs = {}
    
    # 2. 모든 레시피를 순회하며 원가 계산
    for menu_sku_en, ingredients in recipes.items():
        total_cogs = 0.0
        for item in ingredients:
            ing_sku_en = item["ingredient_en"]
            qty = safe_float(item.get("qty", 0.0))
            waste_pct = safe_float(item.get("waste_pct", 0.0))
            
            # 3. 재료 원가 가져오기
            unit_cost = safe_float(ingredient_costs.get(ing_sku_en, 0.0))
            
            # 4. 손실률(waste_pct)을 원가에 반영
            cost_with_waste = unit_cost * (1 + (waste_pct / 100.0))
            
            # 5. 이 재료의 총 원가 = (원가 * 수량)
            total_cogs += (cost_with_waste * qty)
        
        menu_cogs[menu_sku_en] = total_cogs
        
    return menu_cogs

# --- 7. 원가(COGS) '실행' 및 'df'에 통합 ---
try:
    #cogs_load_state = st.info("메뉴별 원가(COGS) 및 마진 계산 중... 💰")
    
    # 1. 메뉴별 COGS 딕셔너리 생성 (e.g., {'Americano': 600.0})
    menu_cogs_map = calculate_menu_cogs(df_inv, RECIPES)
    
    # 2. '상품상세'(한글) <-> 'menu_sku_en' 맵 생성
    cogs_map_kr = {to_korean_detail(sku_en): cogs for sku_en, cogs in menu_cogs_map.items()}

    # 3. 'df'에 '원가' 컬럼 추가
    df['원가'] = df['상품상세'].map(cogs_map_kr).fillna(0.0)
    
    # 4. '순이익' 및 '마진율' 계산
    df['수익'] = pd.to_numeric(df['수익'], errors='coerce').fillna(0)
    df['순이익'] = df['수익'] - df['원가']
    df['마진율(%)'] = (df['순이익'] / df['수익']).replace([pd.NA, float('inf'), float('-inf')], 0).fillna(0) * 100
    
    #cogs_load_state.success("✅ 원가 및 마진 계산 완료!")

except Exception as e:
    #cogs_load_state.error(f"원가 계산 중 오류: {e}")
    # 원가 없이도 앱은 계속 작동해야 함
    df['원가'] = 0.0
    df['수익'] = pd.to_numeric(df['수익'], errors='coerce').fillna(0)
    df['순이익'] = df['수익']
    df['마진율(%)'] = 0.0

# --- 8. 'load_recipe' (L4) 헬퍼 함수 정의 ---
@st.cache_data(ttl=60)
def load_recipe(menu_sku_en: str) -> list[dict]:
    """[L4 수정] DB를 매번 조회하는 대신, 전역 'RECIPES' 딕셔너리 사용"""
    global RECIPES
    return RECIPES.get(menu_sku_en, [])

# --- 9. (기존 함수) 재고 차감 함수들 (순서 변경) ---
def deduct_stock(product_detail_en: str, qty: int):
    ref = ensure_inventory_doc(product_detail_en)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else {}
    init_stock = int(data.get("초기재고", DEFAULT_INITIAL_STOCK))
    cur_stock = safe_float(data.get("현재재고", DEFAULT_INITIAL_STOCK))
    new_stock = max(cur_stock - int(qty), 0)
    ref.update({"현재재고": new_stock})
    return init_stock, new_stock

def get_all_recipe_ingredients() -> set:
    ingredients = set()
    try:
        docs = db.collection(RECIPES_COLLECTION).stream()
        for d in docs:
            items = (d.to_dict() or {}).get("ingredients", [])
            for it in items:
                ingredients.add(it["ingredient_en"])
    except Exception:
        pass
    return ingredients

def deduct_inventory(ingredient_en: str, qty: float, uom: str):
    ref = ensure_inventory_doc(ingredient_en, uom=uom)
    snap = ref.get()
    data = snap.to_dict() or {}
    cur = safe_float(data.get("현재재고", DEFAULT_INITIAL_STOCK))
    inv_uom = normalize_uom(data.get("uom", "ea"))
    use_qty = convert_qty(qty, from_uom=uom, to_uom=inv_uom)
    new_stock = max(cur - use_qty, 0.0)
    ref.update({"현재재고": new_stock})
    return cur, new_stock, inv_uom

def apply_recipe_deduction(menu_sku_en: str, sold_qty: int, commit: bool = True) -> list[dict]:
    items = load_recipe(menu_sku_en)
    summary: list[dict] = []
    if not items:
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
        ing = it["ingredient_en"]
        uom = normalize_uom(it["uom"])
        qty_per_unit = safe_float(it["qty"])
        waste_pct = safe_float(it["waste_pct"], 0)
        total_used = (qty_per_unit * sold_qty) * (1 + (waste_pct / 100.0))
        ref = ensure_inventory_doc(ing, uom=uom, is_ingredient=True)
        snap = ref.get()
        data = snap.to_dict() or {}
        before = safe_float(data.get("현재재고", DEFAULT_INITIAL_STOCK))
        inv_uom = normalize_uom(data.get("uom", "ea"))
        used_converted = convert_qty(total_used, from_uom=uom, to_uom=inv_uom)
        after = max(before - used_converted, 0.0)
        if commit:
            ref.update({"현재재고": after})
        summary.append({"ingredient_en": ing, "used": used_converted, "uom": inv_uom, "before": before, "after": after})
    return summary

def adjust_inventory_by_recipe(menu_sku_en: str,
                               qty_diff: int,
                               move_type: str = "manual_adjust",
                               note: str = ""):
    if qty_diff == 0:
        return
    details = apply_recipe_deduction(menu_sku_en, qty_diff, commit=True)
    log_doc = {
        "ts": datetime.now().isoformat(),
        "type": move_type,
        "menu_sku_en": menu_sku_en,
        "qty": qty_diff,
        "note": note,
        "details": details,
    }
    db.collection(STOCK_MOVES_COLLECTION).add(log_doc)

# === [AI/ML 통합 추가] ===
# SPRINT 1: OpenAI API 호출 헬퍼
def call_openai_api(user_prompt: str, data_context: str, model="gpt-3.5-turbo"):
    """
    [AI 수정 2] data_context(사실)와 user_prompt(요청)를 분리하여 AI가 '거짓말'을 하지 않도록 수정.
    data_context는 'system' 메시지로, user_prompt는 'user' 메시지로 전달.
    """
    
    # 1. API 키가 없는 경우
    if not openai.api_key:
        time.sleep(1.5) 
        st.error("OpenAI API 키가 'secrets.toml'에 설정되지 않았습니다.")
        return (f"⚠️ **[AI 응답 실패 (API 키 없음)]**\n\n"
                f"'secrets.toml'에 OpenAI API 키가 설정되지 않았습니다.\n\n"
                f"--- (데이터 컨텍스트) ---\n{data_context}\n\n"
                f"--- (사용자 요청) ---\n{user_prompt}")

    # 2. API 호출 시도
    try:
        # [수정] 시스템 메시지와 사용자 메시지를 명확히 분리
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "당신은 카페 운영 및 마케팅 전문가입니다. "
                    "다음은 현재 카페의 실제 데이터입니다. 이 데이터를 '사실'로 간주하고, "
                    "이 '사실'에 기반해서만 답변해야 합니다. 절대 데이터를 지어내지 마세요.\n\n"
                    f"--- [카페 실제 데이터] ---\n{data_context}\n--- [데이터 끝] ---"
                )},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
        
    # 3. [수정] 잔액 부족 또는 API 오류 발생 시
    except openai.InsufficientQuotaError as e:
        # "가짜 응답"이 아닌, 명확한 '오류'와 '시도했던 내용'을 반환
        st.error(f"❌ OpenAI API 호출 실패: 잔액(Quota)이 부족합니다. (오류: {e.message})")
        return (f"⚠️ **[AI 응답 실패 (잔액 부족)]**\n\n"
                f"OpenAI 계정의 잔액이 부족하여 응답을 생성할 수 없습니다.\n\n"
                f"--- (AI가 전달받은 데이터) ---\n{data_context}\n\n"
                f"--- (AI가 요청받은 작업) ---\n{user_prompt}")
        
    except openai.AuthenticationError as e:
        st.error("❌ OpenAI API 호출 실패: API 키가 잘못되었습니다. 'secrets.toml'을 확인하세요.")
        return None
    except Exception as e:
        st.error(f"OpenAI API 호출 중 알 수 없는 오류 발생: {e}")
        return None

# SPRINT 2: Prophet 수요 예측 헬퍼
@st.cache_data(ttl=3600) # 1시간 캐시
def get_item_forecast(df_all_sales: pd.DataFrame, menu_sku_en: str, days_to_forecast: int):
    """Prophet을 사용하여 지정된 메뉴의 미래 판매량을 예측합니다."""
    
    try:
        # === [수정] 날짜 데이터 안정화 ===
        df_all_sales = df_all_sales.copy()
        df_all_sales['날짜'] = pd.to_datetime(df_all_sales['날짜'], errors='coerce')
        df_all_sales = df_all_sales.dropna(subset=['날짜'])
        # === [수정 끝] ===

        # === [버그 수정] 이름 불일치 해결 ===
        base_sku_en = re.sub(r"\s+(Lg|Rg|Sm)$", "", menu_sku_en.strip())
        menu_name_kr_base = to_korean_detail(base_sku_en) # This should now be '아메리카노'
        
        original_menu_name_kr = to_korean_detail(menu_sku_en)
        if original_menu_name_kr != menu_name_kr_base:
            st.info(f"AI 예측: '{original_menu_name_kr}' 메뉴의 예측을 위해, 판매 데이터에서 '{menu_name_kr_base}'(으)로 조회합니다.")
        # === [버그 수정 끝] ===

        df_item = df_all_sales[
            df_all_sales['상품상세'] == menu_name_kr_base
        ].copy()
        
        if df_item.empty:
            st.warning(f"판매 데이터(df)에서 '{menu_name_kr_base}' 이름의 판매 기록을 찾을 수 없습니다. (데이터 0건)")
            return None, None # 판매 데이터 없음

        # Prophet이 날짜 데이터를 신뢰하도록 전처리
        df_agg = df_item.groupby('날짜')['수량'].sum().reset_index()
        df_agg['날짜'] = pd.to_datetime(df_agg['날짜'])
        
        if not df_agg.empty:
            date_range = pd.date_range(start=df_agg['날짜'].min(), end=df_agg['날짜'].max())
            df_agg = df_agg.set_index('날짜').reindex(date_range, fill_value=0).reset_index()
            df_agg.rename(columns={'index': '날짜'}, inplace=True)
        
        df_prophet = df_agg[['날짜', '수량']].rename(columns={"날짜": "ds", "수량": "y"})

        if len(df_prophet) < 7: # 데이터가 너무 적으면 예측 불가
            return None, None

        m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=days_to_forecast)
        forecast = m.predict(future)
        
        # === [빈틈 수정] 'y' 컬럼이 빠지는 오류 수정 ===
        # 'ds'(날짜)를 기준으로 'forecast'(예측값)와 'df_prophet'(실제값 y)를 병합합니다.
        forecast_chart_data = forecast.merge(df_prophet, on='ds', how='left')
        
        # 음수 예측은 0으로
        forecast_chart_data['yhat'] = forecast_chart_data['yhat'].clip(lower=0) 
        
        # 예측된 기간(target_days)의 총 소진량 합계 반환
        predicted_sum = forecast_chart_data.iloc[-days_to_forecast:]['yhat'].sum()
        
        # [수정] 'forecast'가 아닌, 'y' 컬럼이 합쳐진 'forecast_chart_data'를 반환
        return max(predicted_sum, 0), forecast_chart_data 

    except Exception as e:
        st.warning(f"Prophet 예측 중 오류 발생: {e}")
        return None, None
# === [AI/ML 통합 추가] ===

# ----------
# [AI/ML 통합 수정 6] 
# ( compute_ingredient_metrics_for_menu )
# SPRINT 2: ML 수요 예측 로직 수정
# - [빈틈 수정] "전체 거래 내역"이 그래프에 반영되도록 .iloc[-90:] 삭제
# - [기능 추가] '실제 판매량(y)'과 'AI 예측(yhat)'을 그래프에 동시 표시
# ----------
def compute_ingredient_metrics_for_menu(
    menu_sku_en: str,
    df_all_sales: pd.DataFrame, # 전체 판매 데이터(df)
    df_inv: pd.DataFrame,
    df_params: pd.DataFrame,
    window_days: int = 28 # [수정] 이 값은 이제 AI 실패 시에만 사용됨
) -> pd.DataFrame:
    """
    [AI 수정됨] 특정 메뉴의 레시피와 *미래 예측 판매량* 기반으로 재료별 지표 계산.
    예측 실패 시 과거 윈도우(window_days) 평균으로 대체.
    """
    items = load_recipe(menu_sku_en)
    if not items:
        return pd.DataFrame()

    # === [버그 수정] 이름 불일치 해결 (Historical) ===
    base_sku_en = re.sub(r"\s+(Lg|Rg|Sm)$", "", menu_sku_en.strip())
    menu_name_kr_base = to_korean_detail(base_sku_en) # '아메리카노'
    # === [버그 수정 끝] ===

    # === [수정] 예측 기간을 21일로 고정하여 버그 해결 ===
    target_days_forecast = 21
    window_days_fallback = 21 # AI 실패 시 사용할 과거 데이터 기간도 21일로 통일
    st.info(f"🤖 AI 수요 예측을 향후 **{target_days_forecast}일** 기준으로 실행합니다.")
    # === [수정 끝] ===

    # 1. (Fallback용) 과거 윈도우 판매량 집계
    sold_sum_historical = 0.0
    if "날짜" in df_all_sales.columns and pd.api.types.is_datetime64_any_dtype(df_all_sales["날짜"]):
        max_day = df_all_sales["날짜"].max()
        min_day = max_day - pd.Timedelta(days=window_days_fallback - 1)
        df_win = df_all_sales[(df_all_sales["날짜"] >= min_day) & (df_all_sales["날짜"] <= max_day)]
        sold_sum_historical = df_win[df_win['상품상세'] == menu_name_kr_base]['수량'].sum()
    
    # 2. [AI/ML] 미래 수요 예측
    predicted_menu_sales, forecast_chart_data = get_item_forecast(
        df_all_sales, menu_sku_en, days_to_forecast=target_days_forecast
    )

    # 3. 사용할 판매량(sold_sum) 및 기준일(days) 결정
    use_historical_fallback = False
    
    if predicted_menu_sales is None or predicted_menu_sales == 0:
        st.warning(f"🤖 AI 예측: '{to_korean_detail(menu_sku_en)}'의 판매 데이터가 부족합니다. (과거 {window_days_fallback}일 판매량: {sold_sum_historical}개)을 기준으로 계산합니다.")
        sold_sum = sold_sum_historical # 과거 데이터 사용
        days = window_days_fallback
        use_historical_fallback = True
    else:
        st.success(f"🤖 **AI 예측**: '{to_korean_detail(menu_sku_en)}'의 향후 **{target_days_forecast}일간** 예상 판매량을 **{predicted_menu_sales:,.0f}개**로 예측했습니다.")
        sold_sum = predicted_menu_sales # 예측값으로 대체
        days = target_days_forecast # 기준일도 예측 기간으로 변경
        
        # (옵션) 예측 차트 표시
        if forecast_chart_data is not None:
            try:
                # === [빈틈 수정] .iloc[-90:] 삭제! ===
                # 이제 Prophet이 학습한 '전체' 과거 데이터와 '미래' 예측을 모두 표시합니다.
                fig = px.line(forecast_chart_data, x='ds', y='yhat', 
                                title=f"'{to_korean_detail(menu_sku_en)}' 전체 기간 수요 예측", 
                                labels={'ds':'날짜', 'yhat':'예측 판매량'})
                
                # === [기능 추가] '실제' 판매량 데이터(y)를 '점'으로 추가 ===
                # (forecast_chart_data에는 Prophet이 학습한 '실제' y값이 포함됨)
                # 'y' 값이 NaN이 아닌 (즉, 과거 데이터가 있는) 부분만
                actual_data = forecast_chart_data.dropna(subset=['y'])
                fig.add_scatter(x=actual_data['ds'], y=actual_data['y'], 
                                mode='markers', 
                                name='실제 판매량', 
                                marker=dict(color='rgba(0,0,255,0.5)', size=5)) # 반투명 파란색 점
                
                # 불확실성 (기존과 동일)
                fig.add_scatter(x=forecast_chart_data['ds'], y=forecast_chart_data['yhat_lower'], fill='tozeroy', mode='lines', line=dict(color='rgba(0,0,0,0)'), name='불확실성(하한)')
                fig.add_scatter(x=forecast_chart_data['ds'], y=forecast_chart_data['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'), fillcolor='rgba(231, 234, 241, 0.5)', name='불확실성(상한)')
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"예측 차트 생성 오류: {e}")

    # 4. 레시피 기반 원재료 소진량 계산 (기존 로직 활용)
    rows = []
    for item in items:
        sku_en = item["ingredient_en"]
        qty_per_unit = safe_float(item.get("qty", 0.0))
        uom = normalize_uom(item.get("uom", "ea"))
        waste_pct = safe_float(item.get("waste_pct", 0.0))
        
        total_used = (qty_per_unit * sold_sum) * (1 + (waste_pct / 100.0))
        
        rows.append({
            "sku_en": sku_en,
            "uom_recipe": uom,
            "total_consumption": total_used # 예측/과거 기반 총 소진량
        })

    if not rows:
        return pd.DataFrame()
    
    use_df = pd.DataFrame(rows).groupby("sku_en").agg({
        "total_consumption": "sum",
        "uom_recipe": "first" 
    }).reset_index()
    
    base = use_df.rename(columns={"total_consumption": "최근소진합"})

    # 5. 재고 지표 계산 (기존 로직 활용)
    base["일평균소진"] = (base["최근소진합"] / max(days, 1)).round(3)
    base.loc[base["일평균소진"].eq(0), "일평균소진"] = 0.01

    base = base.merge(df_inv[['상품상세_en', '현재재고', '초기재고', 'uom']], left_on='sku_en', right_on='상품상세_en', how='left')
    base['현재재고'] = base['현재재고'].fillna(0)
    base['초기재고'] = base['초기재고'].fillna(DEFAULT_INITIAL_STOCK)
    base['uom'] = base['uom'].fillna('ea').apply(normalize_uom)

    base["커버일수"] = (base["현재재고"] / base["일평균소진"]).round(1)

    # 6. ROP 및 권장 발주량 계산
    base = base.merge(df_params, on="sku_en", how="left")
    
    # 파라미터가 없는 경우 기본값
    base['lead_time_days'] = base['lead_time_days'].fillna(3)
    base['safety_stock_units'] = base['safety_stock_units'].fillna(0)
    base['target_days'] = base['target_days'].fillna(21) # 재료의 목표일수

    # === [오타 수정] '일평Y균소진' -> '일평균소진' ===
    base["ROP"] = (base["일평균소진"] * base["lead_time_days"] + base["safety_stock_units"]).round(0).astype(int)
    
    # [핵심] 권장 발주량: (AI가 예측한 총 소진량) - (현재 재고)
    base["권장발주"] = (base["최근소진합"] - base["현재재고"]).apply(lambda x: max(int(ceil(x)), 0))
    
    base["상태"] = base.apply(lambda r: "🚨 발주요망" if r["현재재고"] <= r["ROP"] else "✅ 정상", axis=1)

    base["상품상세"] = base["sku_en"].apply(to_korean_detail)
    cols = ["상품상세","sku_en","현재재고","초기재고","uom","최근소진합","일평균소진","커버일수",
            "lead_time_days","safety_stock_units","target_days","ROP","권장발주","상태"]
    for c in cols:
        if c not in base.columns:
            base[c] = None
            
    # === [오타 수정] '커버일S' -> '커버일수' ===
    return base[cols].sort_values(["상태","커버일수"])

# =============================================================
# === [AI/ML 업그레이드] 프로액티브 분석 함수 (L3 + L4) ===
# =============================================================

@st.cache_data(ttl=3600) # 1시간 캐시
def find_inventory_risks(df, df_inv, df_params):
    """(AI 레벨 3) AI 예측 기반, 재고 위험 품목 상위 3개 찾기"""
    try:
        # 1. 레시피가 있는 메뉴만
        # [L4] 전역 RECIPES 사용
        menu_list_en = list(RECIPES.keys())
        if not menu_list_en:
            return "레시피가 등록되지 않아 재고 위험을 분석할 수 없습니다."
        
        all_risks = []
        
        for menu_sku_en in menu_list_en:
            # 2. 모든 메뉴에 대해 'AI 예측' 및 '재고 계산' 실행 (백그라운드)
            report_df = compute_ingredient_metrics_for_menu(
                menu_sku_en, df, df_inv, df_params, window_days=21
            )
            
            # 3. '발주요망' 상태인 재료 필터링
            risk_items = report_df[report_df['상태'] == '🚨 발주요망']
            
            if not risk_items.empty:
                for _, row in risk_items.iterrows():
                    all_risks.append(
                        f"- '{row['상품상세']}' (메뉴 '{to_korean_detail(menu_sku_en)}'용): "
                        f"현재 재고 {row['현재재고']}{row['uom']}, "
                        f"AI 예측 기반 권장 발주량 {row['권장발주']}{row['uom']}. (커버일수: {row['커버일수']}일)"
                    )
                    
        if not all_risks:
            return "AI 예측 결과, 현재 재고가 충분합니다. (위험 0건)"
        
        # 중복 제거 후 상위 3개만 반환
        return "\n".join(list(set(all_risks))[:3])

    except Exception as e:
        return f"재고 위험 분석 중 오류: {e}"

@st.cache_data(ttl=3600)
def find_slow_moving_items(df, df_inv):
    """(AI 레벨 3) 악성 재고 (30일간 5개 이하 판매) 찾기"""
    try:
        # 1. 30일간 메뉴별 판매량 집계
        min_day = df["날짜"].max() - pd.Timedelta(days=29)
        df_30d = df[df["날짜"] >= min_day]
        sales_counts = df_30d.groupby('상품상세')['수량'].sum()
        
        # 2. 30일간 5개 이하로 팔린 '비인기 메뉴'
        slow_menus_kr = sales_counts[sales_counts <= 5].index.tolist()
        if not slow_menus_kr:
            return "지난 30일간 판매가 부진한 메뉴가 없습니다."
        
        # 3. 비인기 메뉴의 레시피 -> 재료 찾기
        slow_ingredients = set()
        for menu_kr in slow_menus_kr:
            menu_en = from_korean_detail(menu_kr)
            items = load_recipe(menu_en) # [L4] 전역 RECIPES 사용
            for item in items:
                slow_ingredients.add(item['ingredient_en'])
        
        if not slow_ingredients:
            return "지난 30일간 판매가 부진한 메뉴가 있으나, 레시피가 연결되지 않았습니다."
            
        # 4. 해당 재료들의 현재 재고 확인
        df_ing_stock = df_inv[df_inv['상품상세_en'].isin(list(slow_ingredients))]
        df_ing_stock = df_ing_stock.sort_values('현재재고', ascending=False)
        
        if df_ing_stock.empty:
            return "판매 부진 메뉴와 연결된 재료 재고가 없습니다."
            
        report = []
        for _, row in df_ing_stock.head(3).iterrows(): # 재고 많은 상위 3개
            report.append(
                f"- '{row['상품상세']}' (비인기 메뉴용 재료): "
                f"현재 재고 {row['현재재고']}{row['uom']}"
            )
        return "\n".join(report)

    except Exception as e:
        return f"악성 재고 분석 중 오류: {e}"

@st.cache_data(ttl=3600)
def find_top_correlations(df):
    """(AI 레벨 3) 함께 잘 팔리는 메뉴 (상관관계) 찾기"""
    try:
        # 1. 날짜-상품별 판매량 피벗 테이블 생성
        df_pivot = df.pivot_table(
            index='날짜', 
            columns='상품상세', 
            values='수량', 
            aggfunc='sum'
        ).fillna(0)
        
        # (너무 많으면 상위 20개만)
        top_20_items = df_pivot.sum().nlargest(20).index
        df_pivot = df_pivot[top_20_items]
        
        # 2. 상관관계 매트릭스 계산
        corr_matrix = df_pivot.corr()
        
        # 3. 자기 자신(1.0)을 제외한 상위 3개 패턴 찾기
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs < 1].sort_values(ascending=False)
        
        top_3 = corr_pairs.head(3)
        if top_3.empty:
            return "유의미한 동시 판매 패턴을 찾지 못했습니다."
        
        report = []
        for (item1, item2), corr_val in top_3.items():
            report.append(f"- '{item1}' + '{item2}' (상관관계: {corr_val:.2f})")
        return "\n".join(report)
        
    except Exception as e:
        return f"판매 패턴 분석 중 오류: {e}"

@st.cache_data(ttl=3600)
def find_profit_insights(df_with_margin: pd.DataFrame):
    """(AI 레벨 4) '순이익'과 '마진율' 기반 핵심 인사이트 찾기"""
    
    if '순이익' not in df_with_margin.columns or df_with_margin['원가'].sum() == 0:
        return ("'원가' 데이터가 없습니다. '원가 & 레시피 허브' 탭에서 "
                "먼저 '재료 원가'와 '레시피'를 등록해야 '순이익' 분석이 가능합니다.")
    
    try:
        # 1. 메뉴별 집계
        df_agg = df_with_margin.groupby('상품상세').agg(
            총판매수량=('수량', 'sum'),
            총매출=('수익', 'sum'),
            총순이익=('순이익', 'sum')
        ).reset_index()
        
        # 0으로 나누기 방지
        df_agg['평균마진율(%)'] = (df_agg['총순이익'] / df_agg['총매출']).replace([pd.NA, float('inf'), float('-inf')], 0).fillna(0) * 100
        
        # 2. 효자 상품 (순이익 기여도 Top 3)
        stars = df_agg.sort_values('총순이익', ascending=False).head(3)
        star_report = "\n".join([
            f"- '{row['상품상세']}' (총 순이익: {format_krw(row['총순이익'])}, 마진율: {row['평균마진율(%)']:.1f}%)"
            for _, row in stars.iterrows()
        ])
        
        # 3. 수익성 함정 (마진율 하위 3개 - 단, 원가가 0이 아닌 메뉴 중)
        traps = df_agg[df_agg['평균마진율(%)'] > 0].sort_values('평균마진율(%)', ascending=True).head(3)
        trap_report = "\n".join([
            f"- '{row['상품상세']}' (마진율: {row['평균마진율(%)']:.1f}%)"
            for _, row in traps.iterrows()
        ])

        # 4. 손실 상품 (마진율이 0 또는 마이너스)
        loss = df_agg[df_agg['평균마진율(%)'] <= 0]
        loss_report = "손실 발생 메뉴 없음."
        if not loss.empty:
            loss_report = "\n".join([
                f"- '{row['상품상세']}' (마진율: {row['평균마진율(%)']:.1f}%)"
                for _, row in loss.iterrows()
            ])

        return f"""
[효자 상품 (순이익 Top 3)]
{star_report}

[수익성 함정 (마진율 하위 3)]
{trap_report}

[손실 발생 메뉴 (마진율 <= 0)]
{loss_report}
"""
    except Exception as e:
        return f"마진 분석 중 오류: {e}"


# ----------------------
# 5️⃣ 사이드바 메뉴
# ----------------------
# [AI/ML 통합 수정] "AI 비서" 메뉴 추가
menu = st.sidebar.radio(
    " 메뉴 선택",
    ["경영 현황", "매출 대시보드", "기간별 분석", "거래 추가", "재고 관리", "AI 비서", "데이터 편집", "거래 내역", "도움말"]
)

# ==============================================================
# 🧾 거래 추가
# (원본 코드 생략)
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
                "수량": 수량,
                "단가": 단가,
                "수익": 수익,
                "가게위치": "Firebase",
            }
            
            try:
                db.collection(SALES_COLLECTION).add(new_doc)
                st.success(f"✅ '{상품상세_ko}' {수량}건 추가 완료!")
                
                # 재고 자동 차감
                with st.spinner("재고 자동 차감 적용 중..."):
                    adjust_inventory_by_recipe(
                        상품상세_en,
                        수량,
                        move_type="sale",
                        note=f"거래 추가: {상품상세_ko} x{수량}"
                    )
                st.success("✅ 재고 차감 완료!")
                safe_rerun()
                
            except Exception as e:
                st.error(f"데이터 추가 실패: {e}")

# ==============================================================
# 📊 경영 현황
# (원본 코드 생략)
# ==============================================================
elif menu == "경영 현황":
    st.header("📊 경영 현황")
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        total_revenue = df['수익'].sum()
        total_sales_count = df.shape[0]
        avg_revenue_per_sale = total_revenue / total_sales_count if total_sales_count > 0 else 0
        
        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:20px;">
                <div class="metric-card">
                    <div class="metric-title">총 매출</div>
                    <div class="metric-value">{format_krw(total_revenue)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">총 판매 건수</div>
                    <div class="metric-value">{total_sales_count:,} 건</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">건당 평균 매출</div>
                    <div class="metric-value">{format_krw(avg_revenue_per_sale)}</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

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
# (원본 코드 생략)
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
# 📈 기간별 분석
# (원본 코드 생략)
# ==============================================================
elif menu == "기간별 분석":
    st.header("📈 기간별 분석")
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        min_date = df['날짜'].min().date()
        max_date = df['날짜'].max().date()
        
        date_filter = st.slider(
            "조회 기간",
            min_value=min_date, max_value=max_date,
            value=(min_date, max_date),
            format="YYYY/MM/DD"
        )
        
        filtered_df = df[
            (df['날짜'].dt.date >= date_filter[0]) &
            (df['날짜'].dt.date <= date_filter[1])
        ]
        
        if filtered_df.empty:
            st.warning("선택한 기간에 데이터가 없습니다.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                week_sales = filtered_df.groupby('요일')['수익'].sum().reindex(weekday_order_kr)
                fig_week = px.bar(week_sales, x=week_sales.index, y='수익', title="요일별 매출")
                st.plotly_chart(fig_week, use_container_width=True)
            with c2:
                hour_sales = filtered_df.groupby('시')['수익'].sum().reset_index()
                fig_hour = px.bar(hour_sales, x='시', y='수익', title="시간대별 매출")
                st.plotly_chart(fig_hour, use_container_width=True)

# ==============================================================
# 📦 재고 관리
# (원본 코드 생략, [AI/ML 통합 수정]이 적용된 함수를 사용)
# ==============================================================
elif menu == "재고 관리":
    st.header("📦 재고 관리 (AI 예측 기반)")
    
    # [L4] 전역 로드된 df_inv, df_params 사용 (새로 로드 X)
    # df_inv = load_inventory_df()
    # df_params = load_sku_params()
    
    # === [수정] "원가" 기능 반영 ===
    tab1, tab2 = st.tabs(["🎛️ 메뉴별 재고 현황 (AI 예측)", "🔗 원가 & 레시피 허브"])

    # ==============================================================
    # TAB 1: 메뉴별 재고 현황 (AI 예측) - (변경 없음)
    # ==============================================================
    with tab1:
        st.subheader("🎛️ 메뉴별 재고 현황 (AI 예측 기반)")
        
        # [L4] 전역 로드된 RECIPES 사용
        menu_list_en = list(RECIPES.keys())

        if not menu_list_en:
            st.warning("먼저 '원가 & 레시피 허브' 탭에서 메뉴의 레시피를 1개 이상 등록해야 합니다.")
            st.stop()

        menu_list_kr = sorted([to_korean_detail(sku) for sku in menu_list_en])
        
        selected_menu_kr = st.selectbox("분석할 메뉴를 선택하세요:", menu_list_kr, index=0)
        selected_menu_en = from_korean_detail(selected_menu_kr)
        
        st.markdown("---")
        
        try:
            # [L4] 전역 로드된 df, df_inv, df_params 전달
            report_df = compute_ingredient_metrics_for_menu(
                selected_menu_en,
                df, 
                df_inv,
                df_params
            )
            
            if report_df.empty:
                st.info(f"'{selected_menu_kr}'에 대한 레시피 정보가 없습니다.")
            else:
                display_cols = [
                    '상품상세', '상태', '현재재고', 'uom', '권장발주', '커버일수', '일평균소진', 'ROP',
                ]
                
                # (이하 포맷팅 코드 동일)
                formatted_df = report_df[display_cols].copy()
                formatted_df['현재재고'] = formatted_df.apply(lambda r: f"{r['현재재고']:,.1f} {r['uom']}", axis=1)
                formatted_df['권장발주'] = formatted_df.apply(lambda r: f"{r['권장발주']:,.1f} {r['uom']}", axis=1)
                formatted_df['일평균소진'] = formatted_df.apply(lambda r: f"{r['일평균소진']:,.1f} {r['uom']}", axis=1)
                formatted_df['ROP'] = formatted_df.apply(lambda r: f"{r['ROP']:,.1f} {r['uom']}", axis=1)
                formatted_df['커버일수'] = formatted_df['커버일수'].apply(lambda x: f"{x}일")
                
                st.dataframe(
                    formatted_df[['상품상세', '상태', '현재재고', '권장발주', '커버일수', '일평균소진', 'ROP']],
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"재고 리포트 생성 중 오류가 발생했습니다: {e}")
            import traceback
            st.exception(traceback.format_exc())

    # ==============================================================
    # === [수정] "원가 & 레시피 허브"로 업그레이드 (레벨 4) ===
    # ==============================================================
    with tab2:
        st.subheader("✨ 원가 & 레시피 허브 (L4)")
        st.caption("여기서 (1) 재료의 '원가'를 입력하고, (2) 레시피를 만들고, (3) 재고를 봅니다.")

        # === [수정] 탭 3개로 확장 ===
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["💰 1. 재료 원가 관리", "📜 2. 레시피 편집기", "📊 3. 전체 재료 현황"])

        # --- 1. 재료 원가 관리 (L4 업그레이드) ---
        with sub_tab1:
            st.info("AI가 '순이익'을 계산하도록, 재료의 **'매입 원가'**를 입력하세요.")
            st.caption("예: '원두 A'를 '1000'g에 '30000'원에 사왔다면, 그대로 입력하세요.")

            # === [빈틈 수정] inventory가 비어있을 때 (Cold Start) 처리 ===
            if df_inv.empty:
                st.warning("📦 'inventory'가 비어있습니다. (데이터 0건)")
                st.info("AI 예측과 재고 관리를 시작하려면, 'inventory'에 품목이 등록되어 있어야 합니다.")
                
                st.markdown("---")
                st.subheader("🚀 1회 자동 설정 (권장)")
                st.markdown("기존 판매 내역(CSV)에서 찾은 **모든 품목**을 'inventory'에 자동으로 등록합니다.")
                
                if st.button("모든 판매 품목을 'inventory'에 1회 등록하기"):
                    all_menu_items_kr = df['상품상세'].unique()
                    all_menu_items_en = [from_korean_detail(name_kr) for name_kr in all_menu_items_kr if name_kr]
                    
                    with st.spinner(f"{len(all_menu_items_en)}개 품목을 'inventory'로 옮기는 중..."):
                        count = 0
                        for sku_en in all_menu_items_en:
                            if sku_en:
                                ensure_inventory_doc(sku_en, uom="ea", is_ingredient=False)
                                count += 1
                    
                    st.success(f"✅ 총 {count}개의 품목을 'inventory'에 등록했습니다. 페이지를 새로고침합니다.")
                    st.balloons()
                    safe_rerun()
                
                st.stop() # 텅 빈 상태에서는 아래 편집기를 보여주지 않음
            # === [수정 완료] ===
            
            # (inventory가 비어있지 않을 때만 이 코드가 실행됨)
            df_inv_edit = df_inv.copy()
            
            # [L4] load_inventory_df가 이미 처리했지만, 안전을 위해 한번 더 체크
            if 'cost_per_unit' not in df_inv_edit.columns:
                df_inv_edit['cost_per_unit'] = 0.0 
            if 'cost_unit_size' not in df_inv_edit.columns:
                df_inv_edit['cost_unit_size'] = 1.0 
            
            df_inv_edit = df_inv_edit.sort_values('상품상세')
            
            edited_inv_df = st.data_editor(
                df_inv_edit[['상품상세', 'is_ingredient', 'uom', 'cost_unit_size', 'cost_per_unit', '상품상세_en']],
                column_config={
                    "상품상세": st.column_config.TextColumn("품목명", disabled=True),
                    "is_ingredient": st.column_config.CheckboxColumn("재료 여부 (체크)"),
                    "uom": st.column_config.TextColumn("기본 단위", disabled=True),
                    "cost_unit_size": st.column_config.NumberColumn(
                        "매입 단위(g/ml/ea)", 
                        help="원가 계산의 기준 단위입니다. (예: 1kg -> 1000, 1L -> 1000)",
                        min_value=1.0, 
                        format="%.0f"
                    ),
                    "cost_per_unit": st.column_config.NumberColumn(
                        "매입가(원)",
                        help="위 '매입 단위'에 해당하는 가격입니다. (예: 30000)",
                        min_value=0,
                        format="%d원"
                    ),
                    "상품상세_en": st.column_config.TextColumn("SKU (Eng)", disabled=True),
                },
                hide_index=True,
                use_container_width=True
            )

            if st.button("💾 '재료 및 원가' 설정 저장하기", type="primary"):
                changed = 0
                batch = db.batch()
                
                original_map = df_inv.set_index('상품상세_en').to_dict('index')

                for _, item in edited_inv_df.iterrows():
                    sku_en = item['상품상세_en']
                    orig_item = original_map.get(sku_en, {})
                    patch = {}
                    
                    is_ingr_new = bool(item['is_ingredient'])
                    if orig_item.get('is_ingredient') != is_ingr_new:
                        patch['is_ingredient'] = is_ingr_new
                    
                    cost_unit_new = safe_float(item['cost_unit_size'], 1.0)
                    if orig_item.get('cost_unit_size', 1.0) != cost_unit_new:
                        patch['cost_unit_size'] = cost_unit_new
                        
                    cost_new = safe_float(item['cost_per_unit'], 0.0)
                    if orig_item.get('cost_per_unit', 0.0) != cost_new:
                        patch['cost_per_unit'] = cost_new

                    if patch: 
                        doc_ref = db.collection(INVENTORY_COLLECTION).document(sku_en)
                        batch.update(doc_ref, patch)
                        changed += 1
                
                if changed > 0:
                    batch.commit()
                    st.success(f"✅ {changed}건의 재료/원가 설정이 업데이트되었습니다.")
                    st.balloons()
                    safe_rerun()
                else:
                    st.info("변경된 내용이 없습니다.")

        # --- 2. 레시피 편집기 (변경 없음) ---
        with sub_tab2:
            st.info("위 '1. 재료 원가 관리'에서 체크한 재료들로 레시피를 만듭니다.")
            
            try:
                df_ingredients = df_inv[df_inv['is_ingredient'] == True].copy()
                
                if df_ingredients.empty:
                    st.error("오류: '1. 재료 원가 관리' 탭에서 재료를 1개 이상 체크해야 합니다.")
                    st.caption("만약 품목이 없다면, '1. 재료 원가 관리' 탭의 '자동 설정' 버튼을 먼저 실행하세요.")
                    st.stop()
                
                ingredient_options_kr = sorted(df_ingredients['상품상세'].unique().tolist())
                ing_kr_to_en_map = dict(zip(df_ingredients['상품상세'], df_ingredients['상품상세_en']))
                ing_en_to_kr_map = dict(zip(df_ingredients['상품상세_en'], df_ingredients['상품상세']))

            except Exception as e:
                st.error(f"재료 목록 로드 실패: {e}")
                st.stop()

            all_menus_kr = sorted(df['상품상세'].unique().tolist())
            selected_menu_kr = st.selectbox(
                "레시피를 등록/수정할 메뉴를 선택하세요:",
                all_menus_kr
            )
            selected_menu_en = from_korean_detail(selected_menu_kr)
            
            st.caption(f"(Firebase 문서 ID: `{selected_menu_en}`)")
            st.markdown("---")

            current_recipe_items = load_recipe(selected_menu_en)
            recipe_df_rows = []
            if current_recipe_items:
                for item in current_recipe_items:
                    sku_en = item.get("ingredient_en")
                    recipe_df_rows.append({
                        "재료": ing_en_to_kr_map.get(sku_en, f"오류: {sku_en}?"), # 영문 -> 한글
                        "수량": safe_float(item.get("qty", 0.0)),
                        "단위": normalize_uom(item.get("uom", "g")),
                        "손실률(%)": safe_float(item.get("waste_pct", 0.0)),
                    })
            
            if not recipe_df_rows:
                recipe_df_rows = [{"재료": None, "수량": 0.0, "단위": "g", "손실률(%)": 0.0}]

            df_recipe_editor = pd.DataFrame(recipe_df_rows)
            st.subheader(f"📝 `{selected_menu_kr}` 레시피 편집")
            
            edited_df = st.data_editor(
                df_recipe_editor,
                column_config={
                    "재료": st.column_config.SelectboxColumn(
                        "재료 (필수)",
                        options=ingredient_options_kr, # [연동] 1번 탭의 결과
                        required=True,
                    ),
                    "수량": st.column_config.NumberColumn(
                        "수량", min_value=0.0, format="%.2f", required=True,
                    ),
                    "단위": st.column_config.SelectboxColumn(
                        "단위", options=["g", "ml", "ea"], required=True,
                    ),
                    "손실률(%)": st.column_config.NumberColumn(
                        "손실률(%)", min_value=0.0, max_value=100.0, format="%.1f %%", required=True,
                    ),
                },
                num_rows="dynamic", # 행 추가/삭제 가능
                use_container_width=True
            )

            if st.button(f"💾 `{selected_menu_kr}` 레시피 저장하기", type="primary"):
                final_ingredients = []
                valid = True
                
                for index, row in edited_df.iterrows():
                    재료_kr = row["재료"]
                    if not 재료_kr:
                        continue 

                    재료_en = ing_kr_to_en_map.get(재료_kr)
                    
                    if not 재료_en:
                        st.error(f"'{재료_kr}'는 유효한 재료가 아닙니다. '1. 재료 원가 관리' 탭을 확인하세요.")
                        valid = False
                        break
                    
                    final_ingredients.append({
                        "ingredient_en": 재료_en,
                        "qty": safe_float(row["수량"]),
                        "uom": normalize_uom(row["단위"]),
                        "waste_pct": safe_float(row["손실률(%)"]),
                    })

                if valid and not final_ingredients:
                    st.warning("저장할 재료가 없습니다. (모든 행이 비어있음)")
                
                elif valid and final_ingredients:
                    try:
                        db.collection(RECIPES_COLLECTION).document(selected_menu_en).set({
                            "ingredients": final_ingredients
                        })
                        st.success(f"✅ `{selected_menu_kr}` 레시피가 성공적으로 저장되었습니다!")
                        st.balloons()
                        safe_rerun()
                    except Exception as e:
                        st.error(f"Firebase 저장 실패: {e}")
        
        # === [복원] 3. 전체 재료 현황 ===
        with sub_tab3:
            st.subheader("📊 전체 재료 재고 현황")
            st.info("현재 'inventory'에 등록된 모든 재료('is_ingredient' = True)의 현황입니다.")

            df_ing = df_inv[df_inv["is_ingredient"] == True].copy()

            if df_ing.empty:
                st.warning("아직 재료로 설정된 품목이 없습니다. '1. 재료 원가 관리' 탭을 확인하세요.")
            else:
                df_ing['초기재고_calc'] = df_ing['초기재고'].replace(0, 0.01)
                df_ing['재고비율'] = df_ing['현재재고'] / df_ing['초기재고_calc']
                df_ing['상태'] = df_ing['재고비율'].apply(lambda r: "🚨 발주요망" if r <= REORDER_THRESHOLD_RATIO else "✅ 정상")

                low_ing = df_ing[df_ing['재고비율'] <= REORDER_THRESHOLD_RATIO]
                
                fig_ing = px.bar(
                    df_ing.sort_values('재고비율'),
                    x='상품상세', y='현재재고', color='재고비율', title="재료별 재고 현황 (차트)",
                    hover_data=['상품상세', '현재재고', 'uom', '상태']
                )
                st.plotly_chart(fig_ing, use_container_width=True)
                
                st.markdown("---")
                st.subheader("재료별 재고 현황 (테이블)")
                st.dataframe(df_ing[['상품상세','현재재고','초기재고','uom','재고비율','상태']], use_container_width=True)
                
                if not low_ing.empty:
                    st.warning(f"⚠️ {', '.join(low_ing['상품상세'])} 재고가 15% 이하입니다. 발주를 고려하세요.")
# =============================================================
# 🤖 AI 비서 (SPRINT 1)
# === [AI/ML 통합 수정 2] ===
# AI가 '거짓말'을 하지 않도록 데이터 컨텍스트와 사용자 요청을 분리
# =============================================================
# =============================================================
# 🤖 AI 비서 (SPRINT 1)
# === [AI/ML 통합 수정 9] ===
# "레벨 4: AI 재무/운영 분석가"로 업그레이드
# 1. 3대 분석 함수 (재고위험, "마진 인사이트", 판매패턴) 자동 실행
# 2. 분석 결과를 컨텍스트로 AI에게 전달 -> '실행 조언' 생성
# =============================================================
elif menu == "AI 비서":
    st.header("AI 재무/운영 분석가")

    # [수정] 대화 기록을 세션 상태에 저장 (이름 변경으로 강제 리셋)
    if "messages_l4" not in st.session_state:
        st.session_state.messages_l4 = [{"role": "assistant", "content": "안녕하세요, 사장님! 가게 데이터를 분석하여 '순이익'과 '재고 위험' 기반의 핵심 인사이트를 준비 중입니다..."}]
    if "analysis_done_l4" not in st.session_state:
        st.session_state.analysis_done_l4 = False

    # --- 1. (신규) 프로액티브 분석 실행 (1회만) ---
    if not st.session_state.analysis_done_l4:
        with st.spinner("AI가 전체 데이터를 스캔하여 '재고 위험', '마진 분석', '판매 패턴'을 분석 중입니다... (최대 1-2분 소요)"):
            
            # (1) 재고 위험 분석 (L3)
            risk_report = find_inventory_risks(df, df_inv, df_params)
            
            # (2) "마진 인사이트" 분석 (L4)
            profit_report = find_profit_insights(df) # (df는 이제 마진 정보 포함)
            
            # (3) 판매 패턴 분석 (L3)
            pattern_report = find_top_correlations(df)
            
            # (4) 분석 결과를 AI에게 전달할 '핵심 컨텍스트'로 조합
            st.session_state.proactive_context_l4 = f"""
            [AI 분석 리포트 1: 재고 위험 (AI 예측 기반)]
            {risk_report}
            
            [AI 분석 리포트 2: 마진 분석 (원가 기반)]
            {profit_report}
            
            [AI 분석 리포트 3: 핵심 판매 패턴]
            {pattern_report}
            """
            
            st.session_state.analysis_done_l4 = True
            
            # (5) AI를 호출하여 '실행 조언' 생성
            initial_prompt = (
                "위 3가지 분석 리포트(재고 위험, 마진 분석, 판매 패턴)를 기반으로, "
                "사장님에게 '다음 주에 즉시 실행할 액션 플랜 3가지'를 친근한 말투로 담백하게 요약/제안해 주세요. "
                "특히 '마진'과 '재고'를 연계하여 가장 중요한 것 1~2개만 뽑아서 조언해주세요."
            )
            
            initial_advice = call_openai_api(
                user_prompt=initial_prompt,
                data_context=st.session_state.proactive_context_l4
            )
            
            if initial_advice:
                st.session_state.messages_l4 = [
                    {"role": "assistant", "content": f"✅ AI 재무/운영 분석 완료! 사장님을 위한 **핵심 실행 조언**입니다.\n\n{initial_advice}"}
                ]
            else:
                st.session_state.messages_l4 = [
                    {"role": "assistant", "content": "AI 분석은 완료했으나, 조언 생성에 실패했습니다."}
                ]
            
            safe_rerun() # 분석 완료 후 화면 갱신

    # --- 2. 대화창 UI (기존과 동일) ---
    
    # 이전 대화 내용 표시
    for message in st.session_state.messages_l4:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # (디버깅용) 관리자에게만 보이는 컨텍스트
    if "proactive_context_l4" in st.session_state:
        with st.expander("AI가 현재 알고 있는 '프로액티브 분석' 데이터 📋"):
            st.text(st.session_state.proactive_context_l4)

    # [수정] st.chat_input 사용
    if prompt := st.chat_input("위 분석 내용에 대해 더 물어보시거나, 다른 것을 요청하세요..."):
        # 사용자 메시지 표시
        st.session_state.messages_l4.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("AI가 분석 내용과 사장님의 질문을 함께 생각 중입니다... 🧠"):
                
                # [수정] AI 호출 (분석 컨텍스트와 사용자 요청 분리)
                result_text = call_openai_api(
                    user_prompt=prompt,
                    data_context=st.session_state.get("proactive_context_l4", "분석 데이터 없음")
                )
                
                if result_text:
                    st.markdown(result_text)
                    st.session_state.messages_l4.append({"role": "assistant", "content": result_text})
                else:
                    st.error("AI 응답 생성에 실패했습니다.")

# ==============================================================
# ✏️ 데이터 편집
# (원본 코드 생략)
# ==============================================================
# ==============================================================
# ✏️ 데이터 편집
# === [빈틈 수정] '가게위치' 컬럼이 없는 경우(앱 추가 0건)에도 오류 없도록 수정 ===
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
            st.caption("💡 Firebase에 저장된 거래 내역만 수정/삭제할 수 있습니다. (가게위치=Firebase)")
            
            # === [빈틈 수정] ===
            # '가게위치' 컬럼이 없는 경우(앱으로 추가된 데이터가 0건)에 대한 방어 코드
            if '가게위치' in df_view.columns:
                df_view_fb = df_view[df_view['가게위치'] == 'Firebase'].copy()
            else:
                # '가게위치' 컬럼 자체가 없으면, 앱으로 추가된 데이터가 0건이라는 뜻.
                # 빈 데이터프레임을 생성하여 오류를 방지.
                df_view_fb = pd.DataFrame(columns=df_view.columns) 
            # === [수정 완료] ===
            
            if df_view_fb.empty:
                st.info("아직 앱을 통해 추가된(수정 가능한) 거래 데이터가 없습니다.")
            else:
                edited_df = st.data_editor(
                    df_view_fb[['_id','날짜','상품상세','수량','단가','수익']],
                    column_config={
                        "_id": st.column_config.TextColumn("문서ID", disabled=True),
                        "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
                    },
                    hide_index=True,
                    num_rows="dynamic"
                )
                
                reflect_inv = st.checkbox("저장 시 재고에 반영(차감/복원)", value=True)
                
                if st.button("변경된 내용 저장하기 💾"):
                    changed = 0
                    for i, new in edited_df.iterrows():
                        doc_id = new['_id']
                        orig = df_raw[df_raw['_id'] == doc_id].iloc[0]
                        patch = {}
                        
                        try:
                            new_date_str = str(pd.to_datetime(new['날짜']).date())
                        except Exception:
                            new_date_str = str(orig.get('날짜'))

                        if new_date_str != str(orig.get('날짜')):
                            patch['날짜'] = new_date_str
                        
                        detail_en = from_korean_detail(new['상품상세'])
                        if detail_en != orig.get('상품상세'):
                            patch['상품상세'] = detail_en
                        
                        qty_new = int(new['수량'])
                        if qty_new != int(orig.get('수량', 0)):
                            patch['수량'] = qty_new
                        
                        unit_new = float(new['단가'])
                        rev_new = float(new['수익'])
                        
                        if unit_new != float(orig.get('단가', 0)):
                            patch['단가'] = unit_new
                        if rev_new != float(orig.get('수익', 0)):
                            patch['수익'] = rev_new
                        
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
            
            # [수정] df_view_fb에서 ID 목록을 가져오도록 변경
            del_options = df_view_fb['_id'].tolist() if not df_view_fb.empty else []

            del_ids = st.multiselect(
                "🗑️ 삭제할 거래 선택 (문서ID 기준)",
                options=del_options
            )
            colx, _ = st.columns([1,3])
            with colx:
                restore_inv_on_delete = st.checkbox("삭제 시 재고 복원", value=True)
            
            if st.button("삭제 실행", type="primary", disabled=(len(del_ids) == 0)):
                for did in del_ids:
                    if restore_inv_on_delete:
                        try:
                            raw = df_raw[df_raw['_id'] == did].iloc[0]
                            qty_to_restore = -int(raw.get('수량', 0)) # 수량을 음수로
                            detail_en = raw.get('상품상세')
                            if qty_to_restore != 0 and detail_en:
                                adjust_inventory_by_recipe(detail_en, qty_to_restore, move_type="delete_restore", note=str(did))
                        except Exception as e:
                            st.warning(f"{did} 재고 복원 실패: {e}")
                    
                    db.collection(SALES_COLLECTION).document(did).delete()
                
                st.success(f"✅ {len(del_ids)}건 삭제 완료")
                safe_rerun()

    # ------------------ 재고 일괄수정 ------------------
    with tab2:
        st.subheader("✍️ 재고 수기 관리 (실사 반영)")
        st.info("실제 재고를 확인한 후, 수량을 직접 수정하고 저장하세요.")
        
        df_inv = load_inventory_df()
        
        if df_inv.empty:
            st.warning("재고 정보가 없습니다.")
        else:
            edited_inv = st.data_editor(
                df_inv,
                column_config={
                    "상품상세_en": st.column_config.TextColumn("SKU (Eng)", disabled=True),
                    "상품상세": st.column_config.TextColumn("품목명", disabled=True),
                    "초기재고": st.column_config.NumberColumn("초기 재고", disabled=True),
                    "현재재고": st.column_config.NumberColumn("현재 재고", min_value=0.0, format="%.2f"),
                    "uom": st.column_config.TextColumn("단위", disabled=True),
                    "is_ingredient": st.column_config.CheckboxColumn("재료 여부", disabled=True),
                },
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("실사 재고 저장하기 💾", type="primary"):
                changed = 0
                original_map = {row['상품상세_en']: row['현재재고'] for _, row in df_inv.iterrows()}
                
                batch = db.batch()
                
                for item in edited_inv:
                    sku = item['상품상세_en']
                    new_stock = safe_float(item['현재재고'])
                    
                    if sku in original_map and original_map[sku] != new_stock:
                        doc_ref = db.collection(INVENTORY_COLLECTION).document(sku)
                        batch.update(doc_ref, {'현재재고': new_stock})
                        changed += 1
                        
                if changed > 0:
                    batch.commit()
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
        st.caption(f"현재 데이터 크기: {len(df)}행")
        
        # [수정] 원본의 st.dataframe(df.head(1000)) 중복 제거
        st.dataframe(df[cols].sort_values('날짜', ascending=False), width=None, use_container_width=True)


# ==============================================================
# ❓ 도움말
# ==============================================================
else:  # menu == "도움말"
    st.header("☕️ 커피 원두 재고관리 파이프라인 쉽게 이해하기")
    
    # [AI/ML 통합 수정] 도움말 내용 업데이트
    st.markdown("""
> **“커피 원두가 어떻게 들어오고, 얼마나 쓰이고, 언제 다시 주문돼야 하는지를 자동으로 관리하자!”** 엑셀 대신 ERP가 자동으로 계산해줍니다.

### 1. (AI) 스마트 발주 로직 (재고 관리 탭)
| 단계 | 하는 일 | 예시 |
| --- | --- | --- |
| **1. (AI) 수요 예측** | Prophet (ML)이 "아메리카노"의 **미래 21일** 판매량을 **[500잔]**으로 예측 |
| **2. 소진량 계산** | [500잔] x [레시피: 잔당 20g] = **[10,000g]** (예상 총 소진량) |
| **3. 권장 발주량** | [10,000g] - [현재 재고: 3,000g] = **[7,000g]** (권장 발주량) |
| **4. ROP (발주점)** | (일평균소진 * 리드타임) + 안전재고. 이보다 재고가 낮으면 **'🚨 발주요망'** 알림 |
| **(대체)** | AI 예측 실패 시, 과거 28일 평균 판매량으로 자동 전환되어 계산됩니다. |

### 2. (AI) 마케팅 보조 (AI 비서 탭)
| 기능 | 설명 |
| --- | --- |
| **인스타그램 생성** | 현재 베스트셀러 데이터를 기반으로 AI가 홍보 문구를 자동 생성합니다. |
| **운영 보고** | 일일 매출, 판매 건수 등을 요약하여 간결한 보고서를 생성합니다. |

### 3. 기본 데이터 흐름
| 단계 | 하는 일 | 예시 |
| --- | --- | --- |
| **1. 원두 입고** | '데이터 편집' > '재고 일괄수정' 탭에서 **[+10,000g]** 수동 입력 |
| **2. 판매 발생** | '거래 추가' 탭 또는 POS에서 '아메리카노' 1잔 판매 (Firestore 'coffee_sales'에 기록) |
| **3. 자동 차감** | 시스템이 '아메리카노' 레시피(BOM)를 조회하여 [원두: 20g] 사용 확인 |
| **4. 재고 반영** | 'inventory' DB의 '원두' 재고를 **[-20g]** 자동 차감 (재고 이동 로그 기록) |
""")