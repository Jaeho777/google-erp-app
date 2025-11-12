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
    from prophet import Prophet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
except ImportError:
    st.error("""
    AI/ML 기능을 위한 라이브러리가 부족합니다.
    터미널에서 'pip install openai prophet scikit-learn'를 실행해주세요.
    """)
    st.stop()
# === [AI/ML 통합 추가] ===


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

# ----------------------
# 3️⃣ CSV + Firebase 통합 → 화면표시용 한글화
# (원본 코드 생략)
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
# (원본 코드 생략)
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
        return s
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
    if f == "g" and t == "kg":
        return q / 1000.0
    if f == "kg" and t == "g":
        return q * 1000.0
    if f == "ml" and t == "l":
        return q / 1000.0
    if f == "l" and t == "ml":
        return q * 1000.0
    return q

# ----------------------
# 4-1️⃣ [NEW] 재고관리
# (원본 코드 생략)
# ----------------------
@st.cache_data(ttl=60)
def load_recipe(menu_sku_en: str) -> list[dict]:
    try:
        ref = db.collection(RECIPES_COLLECTION).document(menu_sku_en).get()
        if ref.exists:
            data = ref.to_dict()
            return data.get("ingredients", [])
    except Exception:
        pass
    return []

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
        })
        return ref

def ensure_ingredient_sku(ingredient_en: str, uom: str = "ea"):
    return ensure_inventory_doc(ingredient_en, uom=uom, is_ingredient=True)

def deduct_stock(product_detail_en: str, qty: int):
    ref = ensure_inventory_doc(product_detail_en)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else {}
    init_stock = int(data.get("초기재고", DEFAULT_INITIAL_STOCK))
    cur_stock = safe_float(data.get("현재재고", DEFAULT_INITIAL_STOCK))
    new_stock = max(cur_stock - int(qty), 0)
    ref.update({"현재재고": new_stock})
    return init_stock, new_stock

def load_inventory_df() -> pd.DataFrame:
    inv_docs = db.collection(INVENTORY_COLLECTION).stream()
    rows = []
    for d in inv_docs:
        doc = d.to_dict() or {}
        en = doc.get("상품상세_en", d.id)
        ko = to_korean_detail(en)
        rows.append({
            "상품상세_en": en,
            "상품상세": ko,
            "초기재고": doc.get("초기재고", DEFAULT_INITIAL_STOCK),
            "현재재고": doc.get("현재재고", DEFAULT_INITIAL_STOCK),
            "uom": normalize_uom(doc.get("uom", "ea")),
            "is_ingredient": bool(doc.get("is_ingredient", False)),
        })
    return pd.DataFrame(rows)

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

# ----------------------
# 4-2️⃣ [NEW] SKU 파라미터
# (원본 코드 생략)
# ----------------------
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

# === [AI/ML 통합 추가] ===
# SPRINT 1: OpenAI API 호출 헬퍼
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
# SPRINT 2: Prophet 수요 예측 헬퍼
@st.cache_data(ttl=3600) # 1시간 캐시
def get_item_forecast(df_all_sales: pd.DataFrame, menu_sku_en: str, days_to_forecast: int):
    """Prophet을 사용하여 지정된 메뉴의 미래 판매량을 예측합니다."""
    
    try:
        # === [수정] 날짜 데이터 안정화 ===
        # 함수로 전달된 df의 날짜 컬럼을 한번 더 보정 (NaT 제거)
        df_all_sales = df_all_sales.copy()
        df_all_sales['날짜'] = pd.to_datetime(df_all_sales['날짜'], errors='coerce')
        df_all_sales = df_all_sales.dropna(subset=['날짜'])
        # === [수정 끝] ===

        # 원본 df는 '상품상세'가 한글이므로 한글명 사용
        menu_name_kr = to_korean_detail(menu_sku_en)
        
        df_item = df_all_sales[
            df_all_sales['상품상세'] == menu_name_kr
        ].copy()
        
        if df_item.empty:
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
        
        # 음수 예측은 0으로
        forecast['yhat'] = forecast['yhat'].clip(lower=0) 
        predicted_sum = forecast.iloc[-days_to_forecast:]['yhat'].sum()
        
        return max(predicted_sum, 0), forecast

    except Exception as e:
        st.warning(f"Prophet 예측 중 오류 발생: {e}")
        return None, None
    
# === [AI/ML 통합 추가] ===


# ----------
# [AI/ML 통합 수정] 
# ( compute_ingredient_metrics_for_menu )
# SPRINT 2: ML 수요 예측 기능을 기존 ROP 계산 로직에 통합
# ----------
# ----------
# [AI/ML 통합 수정 3] 
# ( compute_ingredient_metrics_for_menu )
# SPRINT 2: ML 수요 예측 로직 수정
# - 'target_days'를 가져오는 로직의 버그를 수정
# - 예측 기간을 21일로 고정하여 단순화/안정화
# ----------
# ----------
# [AI/ML 통합 수정 5] 
# ( compute_ingredient_metrics_for_menu )
# SPRINT 2: ML 수요 예측 로직 수정
# - [오타 수정] '일평Y균소진' -> '일평균소진'
# - [오타 수정] '커버일S' -> '커버일수'
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
    # 'Americano Rg' -> 'Americano' (기본 이름)으로 변경
    # '아메리카노 (레귤러)' -> '아메리카노' (기본 이름)으로 변경
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
        
        # [수정] menu_name_kr 대신 menu_name_kr_base 사용
        sold_sum_historical = df_win[df_win['상품상세'] == menu_name_kr_base]['수량'].sum()
    
    # 2. [AI/ML] 미래 수요 예측
    # (이 함수는 내부적으로 수정되었으므로, 여기서는 menu_sku_en 원본을 그대로 전달)
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
                fig = px.line(forecast_chart_data.iloc[-90:], x='ds', y='yhat', 
                                title=f"'{to_korean_detail(menu_sku_en)}' 수요 예측 (향후 {target_days_forecast}일)", 
                                labels={'ds':'날짜', 'yhat':'예측 판매량'})
                fig.add_scatter(x=forecast_chart_data['ds'], y=forecast_chart_data['yhat_lower'], fill='tozeroy', mode='lines', line=dict(color='rgba(0,0,0,0)'), name='불확실성')
                fig.add_scatter(x=forecast_chart_data['ds'], y=forecast_chart_data['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'), fillcolor='rgba(231, 234, 241, 0.5)', name='')
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
# ---------- [AI/ML 통합 수정 종료] ----------


# ----------------------
# 5️⃣ 사이드바 메뉴
# ----------------------
# [AI/ML 통합 수정] "AI 비서" 메뉴 추가
menu = st.sidebar.radio(
    " 메뉴 선택",
    ["경영 현황", "매출 대시보드", "기간별 분석", "거래 추가", "재고 관리", "🤖 AI 비서", "데이터 편집", "거래 내역", "도움말"]
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
    
    df_inv = load_inventory_df()
    df_params = load_sku_params()
    
    # === [수정] 탭 이름 변경 ===
    tab1, tab2 = st.tabs(["🎛️ 메뉴별 재고 현황", "🔗 레시피 & 재료 관리 허브"])

    # ==============================================================
    # TAB 1: 메뉴별 재고 현황 (AI 예측) - (변경 없음)
    # ==============================================================
    with tab1:
        st.subheader("🎛️ 메뉴별 재고 현황 (AI 예측 기반)")
        
        # 레시피가 등록된 메뉴만 선택지로
        try:
            recipe_docs = db.collection(RECIPES_COLLECTION).stream()
            menu_list_en = [doc.id for doc in recipe_docs if doc.id]
        except Exception:
            menu_list_en = []

        if not menu_list_en:
            st.warning("먼저 '레시피 & 재료 관리 허브' 탭에서 메뉴의 레시피를 1개 이상 등록해야 합니다.")
            st.stop()

        menu_list_kr = sorted([to_korean_detail(sku) for sku in menu_list_en])
        
        selected_menu_kr = st.selectbox("분석할 메뉴를 선택하세요:", menu_list_kr, index=0)
        selected_menu_en = from_korean_detail(selected_menu_kr)
        
        st.markdown("---")
        
        try:
            report_df = compute_ingredient_metrics_for_menu(
                selected_menu_en,
                df, # 전체 'df' 전달
                df_inv,
                df_params
            )
            
            if report_df.empty:
                st.info(f"'{selected_menu_kr}'에 대한 레시피 정보가 없습니다.")
            else:
                display_cols = [
                    '상품상세', '상태', '현재재고', 'uom', '권장발주', '커버일수', '일평균소진', 'ROP',
                ]
                
                # 단위 포맷팅
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
    # === [수정] "범용 레시피 & 재료 관리 허브"로 완전 교체 ===
    # ==============================================================
    with tab2:
        st.subheader("✨ 범용 레시피 & 재료 관리 허브")
        st.caption("여기서 (1) 재료로 쓸 품목을 지정하고, (2) 해당 재료로 레시피를 만듭니다.")

        # === [신규] 서브 탭으로 작업 흐름 분리 ===
        sub_tab1, sub_tab2 = st.tabs(["📦 1. 재료 목록 관리", "📜 2. 레시피 편집기"])

        # --- 1. 재료 목록 관리 (신규 기능) ---
        with sub_tab1:
            st.info("레시피에 사용할 **'재료'** 품목을 여기서 체크(True)하세요.")
            st.caption("('원두 A', '우유 (1L)' 등은 체크O, '아메리카노 완제품' 등은 체크X)")

            df_inv_edit = df_inv.copy()
            df_inv_edit = df_inv_edit.sort_values('상품상세')
            
            # 재료 목록 편집기
            edited_inv_df = st.data_editor(
                df_inv_edit[['상품상세_en', '상품상세', 'is_ingredient', 'uom']],
                column_config={
                    "상품상세_en": st.column_config.TextColumn("SKU (Eng)", disabled=True),
                    "상품상세": st.column_config.TextColumn("품목명", disabled=True),
                    "is_ingredient": st.column_config.CheckboxColumn("재료 여부 (체크)"),
                    "uom": st.column_config.TextColumn("기본 단위", disabled=True),
                },
                hide_index=True,
                use_container_width=True
            )

            if st.button("💾 '재료 여부' 설정 저장하기", type="primary"):
                changed = 0
                batch = db.batch()
                
                # 원본과 비교
                original_map = {row['상품상세_en']: row['is_ingredient'] for _, row in df_inv.iterrows()}

                for _, item in edited_inv_df.iterrows():
                    sku_en = item['상품상세_en']
                    is_ingr_new = bool(item['is_ingredient'])
                    
                    if sku_en in original_map and original_map[sku_en] != is_ingr_new:
                        doc_ref = db.collection(INVENTORY_COLLECTION).document(sku_en)
                        batch.update(doc_ref, {'is_ingredient': is_ingr_new})
                        changed += 1
                
                if changed > 0:
                    batch.commit()
                    st.success(f"✅ {changed}건의 재료 설정이 업데이트되었습니다.")
                    st.balloons()
                    safe_rerun()
                else:
                    st.info("변경된 내용이 없습니다.")

        # --- 2. 레시피 편집기 (기존 기능 + 업그레이드) ---
        with sub_tab2:
            st.info("위 '1. 재료 목록 관리'에서 체크한 재료들로 레시피를 만듭니다.")
            
            # --- 2-1. 재료 목록 준비 (1번 탭의 결과물) ---
            try:
                # [수정] 'is_ingredient'가 True인 품목만 재료로 간주 (df_inv 원본 사용)
                df_ingredients = df_inv[df_inv['is_ingredient'] == True].copy()
                
                if df_ingredients.empty:
                    st.error("오류: '1. 재료 목록 관리' 탭에서 재료를 1개 이상 체크해야 합니다.")
                    st.stop()
                
                # 재료 목록 (한글)
                ingredient_options_kr = sorted(df_ingredients['상품상세'].unique().tolist())
                
                # 한글 <-> 영문 변환 맵
                ing_kr_to_en_map = dict(zip(df_ingredients['상품상세'], df_ingredients['상품상세_en']))
                ing_en_to_kr_map = dict(zip(df_ingredients['상품상세_en'], df_ingredients['상품상세']))

            except Exception as e:
                st.error(f"재료 목록 로드 실패: {e}")
                st.stop()

            # --- 2-2. 메뉴 선택 (사장님이 판매하는 모든 메뉴) ---
            all_menus_kr = sorted(df['상품상세'].unique().tolist())
            selected_menu_kr = st.selectbox(
                "레시피를 등록/수정할 메뉴를 선택하세요:",
                all_menus_kr
            )
            selected_menu_en = from_korean_detail(selected_menu_kr)
            
            st.caption(f"(Firebase 문서 ID: `{selected_menu_en}`)")
            st.markdown("---")

            # --- 2-3. 현재 레시피 불러오기 & 편집기 UI ---
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

            # --- 2-4. 저장 로직 ---
            if st.button(f"💾 `{selected_menu_kr}` 레시피 저장하기", type="primary"):
                final_ingredients = []
                valid = True
                
                for index, row in edited_df.iterrows():
                    재료_kr = row["재료"]
                    if not 재료_kr:
                        continue # 빈 행은 무시

                    재료_en = ing_kr_to_en_map.get(재료_kr)
                    
                    if not 재료_en:
                        st.error(f"'{재료_kr}'는 유효한 재료가 아닙니다. '1. 재료 목록 관리' 탭을 확인하세요.")
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
                    # (선택) 레시피를 비우고 싶다면 삭제
                    # db.collection(RECIPES_COLLECTION).document(selected_menu_en).delete()
                    # st.success(f"'{selected_menu_kr}' 레시피가 삭제되었습니다.")
                
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

# =============================================================
# 🤖 AI 비서 (SPRINT 1)
# === [AI/ML 통합 추가] ===
# =============================================================
# =============================================================
# 🤖 AI 비서 (SPRINT 1)
# === [AI/ML 통합 수정 2] ===
# AI가 '거짓말'을 하지 않도록 데이터 컨텍스트와 사용자 요청을 분리
# =============================================================
elif menu == "🤖 AI 비서":
    st.header("🤖 AI 마케팅/운영 비서")
    st.markdown("현재 판매 데이터를 기반으로 AI가 마케팅 문구나 운영 보고서를 생성합니다.")

    if df.empty:
        st.info("아직 판매 데이터가 없습니다. 데이터가 쌓이면 AI 비서를 사용할 수 있습니다.")
    else:
        try:
            # 1. [수정] 데이터 컨텍스트(사실)를 명확하게 수집
            total_revenue = df['수익'].sum()
            total_sales_count = len(df)
            
            top_prod_series = df.groupby('상품상세')['수익'].sum().sort_values(ascending=False).head(3)
            top_prod_list = [f"{idx} ({format_krw(val)})" for idx, val in top_prod_series.items()]
            top_prod_str = ", ".join(top_prod_list)
            
            # AI에게 전달할 '사실' 데이터 묶음
            data_context_string = f"""
            - 총 매출: {format_krw(total_revenue)}
            - 총 판매 건수: {total_sales_count}건
            - 매출 기준 베스트셀러 Top 3: {top_prod_str}
            """
            
            st.info(f"AI에게 전달될 실제 데이터: {data_context_string.strip()}")

            # 2. [수정] 프롬프트 선택지 (사용자의 '요청' 부분만 남김)
            prompt_options = {
                "인스타그램 홍보 (활기찬 톤)": "현재 데이터를 기반으로, 베스트셀러 메뉴를 강조하는 인스타그램 홍보 게시물을 '매우' 친근하고 활기찬 톤으로 작성해줘. 이모지도 팍팍 넣어주고 해시태그도 5개 이상 달아줘.",
                "단골손님 감사 문자 (정중한 톤)": "현재 데이터를 기반으로, 단골손님에게 감사를 표하는 SMS 문자 메시지를 정중하지만 따뜻하게 작성해줘.",
                "일일 운영 보고 (매니저용)": "현재 데이터를 바탕으로 매니저에게 보고할 간결한 일일 운영 요약 보고서를 작성해줘. (숫자 요약 포함)"
            }
            
            selected_prompt_key = st.selectbox("AI에게 요청할 작업을 선택하세요:", list(prompt_options.keys()))
            
            custom_prompt_area = st.text_area("또는, AI에게 직접 요청할 내용을 입력하세요:", 
                                              placeholder="예: 현재 베스트셀러 3가지를 활용한 신규 세트 메뉴 아이디어 3가지 제안해줘")
            
            if st.button("AI 생성하기 🚀", type="primary"):
                
                # 3. [수정] 사용자 요청(user_prompt)을 확정
                user_request_prompt = ""
                if custom_prompt_area:
                    st.info("직접 입력한 프롬프트로 요청합니다...")
                    user_request_prompt = custom_prompt_area
                else:
                    user_request_prompt = prompt_options[selected_prompt_key]

                with st.spinner("AI가 실제 데이터를 기반으로 생각 중입니다... 🧠"):
                    
                    # 4. [수정] '데이터 컨텍스트'와 '사용자 요청'을 분리하여 호출
                    result_text = call_openai_api(
                        user_prompt=user_request_prompt,
                        data_context=data_context_string
                    )
                    
                    if result_text:
                        st.success("AI 생성 완료!")
                        st.text_area("결과물:", result_text, height=300)
                    else:
                        st.error("AI 응답 생성에 실패했습니다.")

        except Exception as e:
            st.error(f"데이터를 분석하는 중 오류가 발생했습니다: {e}")

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
# (원본 코드 생략)
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