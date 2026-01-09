# ==============================================================
# ☕ Coffee ERP Dashboard — Company Showcase Edition (Tone-Down Blue)
# (기존 주석 생략)
# ==============================================================

import os
import io
import json
import re
import warnings
import math
import uuid
import difflib
from math import ceil
from pathlib import Path
from datetime import datetime
import time # #[AI/ML 통합 추가] (Mock 응답용)
import base64

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime, timedelta
import plotly.graph_objects as go
import textwrap

from pricing_fetch import fetch_naver_prices, load_mapping, merge_price_rows

import firebase_admin
from firebase_admin import credentials, firestore, storage

# === [AI/ML 통합 추가] ===
# SPRINT 1 (AI 비서) 및 SPRINT 2 (수요 예측) 라이브러리
try:
    from google import genai
    from prophet import Prophet
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error
except ImportError:
    st.error("""
    AI/ML 기능을 위한 라이브러리가 부족합니다.
    터미널에서 'pip install google-genai prophet scikit-learn'를 실행해주세요.
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

# UX/비즈니스 로직에 필요한 상수/도우미
SUPPLY_MODES = ["", "거래처 도매 발주", "전화/직접 방문"]
DEFAULT_SUPPLY_MODE = SUPPLY_MODES[0]
DEFAULT_SUPPLY_LEAD_DAYS = 2
DEFAULT_GRAMS_PER_CUP = 15.0  # g 단위 재고를 잔(컵)으로 환산할 때 사용

HOLIDAYS_FIXED = {
    "01-01": "신정", "03-01": "삼일절", "05-05": "어린이날", "06-06": "현충일",
    "08-15": "광복절", "10-03": "개천절", "10-09": "한글날", "12-25": "성탄절",
}

def is_holiday_date(d) -> bool:
    try:
        return d.strftime("%m-%d") in HOLIDAYS_FIXED
    except Exception:
        return False

def format_date_with_holiday(d) -> str:
    """요일+공휴일 표시 문자열 생성."""
    try:
        weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][d.weekday()]
    except Exception:
        return str(d)
    holiday_name = HOLIDAYS_FIXED.get(d.strftime("%m-%d"))
    suffix = f" ({weekday_kr})"
    if holiday_name:
        suffix += f" 공휴일"
    return f"{d.isoformat()}{suffix}"


def parse_currency_input(raw: str) -> float:
    """쉼표/원 단위를 제거하고 숫자(float) 반환."""
    if raw is None:
        return 0.0
    s = str(raw).replace(",", "").replace("원", "").strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def render_currency_input(label: str, value: float, key: str):
    """텍스트 입력으로 통화 입력 UX 제공 (쉼표 자동 포맷)."""
    formatted_default = f"{int(value):,}" if value is not None else ""
    # Streamlit 제약: 세션 상태로 값을 넣은 위젯은 value 파라미터를 함께 주면 경고가 발생한다.
    if key in st.session_state:
        typed = st.text_input(label, key=key, help="숫자만 입력하면 자동으로 원 단위를 맞춥니다.")
    else:
        typed = st.text_input(label, value=formatted_default, key=key, help="숫자만 입력하면 자동으로 원 단위를 맞춥니다.")
    cleaned_val = parse_currency_input(typed)
    pretty = f"{int(cleaned_val):,}원" if cleaned_val else "0원"
    st.caption(f"입력값: {pretty}")
    return cleaned_val


def choose_option(label: str, options: list[str], key: str, placeholder: str | None = None):
    """옵션이 3개 이하인 경우 버튼/라디오로, 그 외에는 selectbox 사용."""
    if not options:
        return None
    if len(options) <= 3:
        return st.radio(label, options, key=key, horizontal=True, label_visibility="visible")
    return st.selectbox(label, options, key=key, index=None, placeholder=placeholder)


def get_recent_sales_entries(df_source: pd.DataFrame, limit: int = 3):
    """최근 거래 N건을 단순히 조회하는 헬퍼 (UI/재사용용)."""
    if df_source is None or df_source.empty:
        return []
    try:
        df_recent = df_source.dropna(subset=["상품상세"]).copy()
        if not pd.api.types.is_datetime64_any_dtype(df_recent.get("날짜")):
            df_recent["날짜"] = pd.to_datetime(df_recent["날짜"], errors="coerce")
        df_recent = df_recent.dropna(subset=["날짜"])
        df_recent = df_recent.sort_values("날짜", ascending=False).head(limit)
        rows = []
        for _, row in df_recent.iterrows():
            rows.append({
                "상품상세": row.get("상품상세"),
                "상품카테고리": row.get("상품카테고리"),
                "상품타입": row.get("상품타입"),
                "단가": safe_float(row.get("단가", row.get("price", 0))),
                "수량": int(safe_float(row.get("수량", 1), 1)),
                "날짜": pd.to_datetime(row.get("날짜")).date() if pd.notna(row.get("날짜")) else datetime.now().date(),
            })
        return rows
    except Exception:
        return []

st.set_page_config(page_title="☕ Coffee ERP Dashboard", layout="wide")

# === 글로벌 글자 크기 설정 ===
# 기본값을 1.1로 조정하고 단계별 배율도 함께 조정
FONT_SCALE_MAP = {"기본": 1.1, "크게": 1.2, "매우 크게": 1.35}
st.session_state.setdefault("font_scale_label", "기본")
font_scale = FONT_SCALE_MAP.get(st.session_state.get("font_scale_label", "기본"), 1.0)
st.markdown(
    f"""
    <style>
    :root {{ --base-font-scale: {font_scale}; }}
    html, body, [data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * {{
        font-size: calc(16px * var(--base-font-scale));
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
        font-size: calc(24px * var(--base-font-scale));
    }}
    /* 제목은 기본 크기 유지 */
    h1, h2, h3 {{
        font-size: revert;
    }}
    /* 대시보드 타이틀/섹션 헤더는 작게(0.9배) */
    h1, h2, h3, h4, h5 {{
        font-size: calc(1em * 0.9);
    }}
    /* 대시보드/홈 타이틀은 크게 (2.0배) */
    .dashboard-header h1 {{
        font-size: 2em !important;
    }}
    .home-title {{
        font-size: 2em !important;
        margin: 0 0 12px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_firebase():
    try:
        # 1순위: Streamlit Cloud Secrets 확인 (지금 설정하신 방법)
        if "firebase_service_account" in st.secrets:
            # Secrets 값을 딕셔너리로 가져오기
            cred_info = dict(st.secrets["firebase_service_account"])
            
            # 줄바꿈 문자(\n) 에러 방지 처리 (필수)
            if "private_key" in cred_info:
                cred_info["private_key"] = cred_info["private_key"].replace("\\n", "\n")

            cred = credentials.Certificate(cred_info)
            
            # 앱이 초기화되지 않았을 때만 초기화
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            
            return firestore.client(), "success"

        # 2순위: 환경 변수 확인 (기존 코드 유지 - 백업용)
        elif "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
            cred_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            cred = credentials.Certificate(cred_info)
            
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            
            return firestore.client(), "success"

        # 3순위: 아무것도 못 찾았을 때
        else:
            return None, "no_env_or_secrets_found"

    except Exception as e:
        return None, f"error: {e}"

# 함수 호출
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


def get_secret(key: str, default=None):
    """환경변수와 st.secrets를 모두 확인하여 값을 가져옵니다."""
    try:
        if key in st.secrets:
            return st.secrets[key]
        # naver 키가 [naver] 섹션에 있을 때 처리
        if key.startswith("NAVER_") and "naver" in st.secrets:
            sub = st.secrets["naver"]
            if key == "NAVER_CLIENT_ID" and "client_id" in sub:
                return sub["client_id"]
            if key == "NAVER_CLIENT_SECRET" and "client_secret" in sub:
                return sub["client_secret"]
    except Exception:
        pass
    return os.environ.get(key, default)


def _resolve_path(val, default: Path) -> Path:
    if not val:
        return default
    p = Path(str(val))
    return p if p.is_absolute() else (BASE_DIR / p)

DATA_DIR   = _resolve_path(SECRETS.get("DATA_DIR")   or os.environ.get("ERP_DATA_DIR"),   BASE_DIR / "data")
ASSETS_DIR = _resolve_path(SECRETS.get("ASSETS_DIR") or os.environ.get("ERP_ASSETS_DIR"), BASE_DIR / "assets")
KEYS_DIR   = _resolve_path(SECRETS.get("KEYS_DIR")   or os.environ.get("ERP_KEYS_DIR"),   BASE_DIR / "keys")
RECEIPT_DIR = _resolve_path(SECRETS.get("RECEIPT_DIR") or os.environ.get("ERP_RECEIPT_DIR"), DATA_DIR / "receipts")
UPLOAD_DIR = _resolve_path(SECRETS.get("UPLOAD_DIR") or os.environ.get("ERP_UPLOAD_DIR"), DATA_DIR / "uploads")

CSV_PATH     = DATA_DIR / "데이터 증강.csv"
CSV_AUGMENTED_PATH = DATA_DIR / "데이터 증강.csv"
CSV_PRODUCT_STATUS_PATH = DATA_DIR / "상품매출현황.csv"
CSV_HOURLY_PATH = DATA_DIR / "시간대별 매출분석.csv"
CSV_TOP5_PATH = DATA_DIR / "카피엔드_커피_Top5.csv"
PIPELINE_IMG = ASSETS_DIR / "pipeline_diagram.png"
SA_FILE_PATH = KEYS_DIR / "serviceAccount.json"

SALES_COLLECTION      = "coffee_sales"
INVENTORY_COLLECTION  = "inventory"
ORDERS_COLLECTION     = "orders"
SKU_PARAMS_COLLECTION = "sku_params"
RECEIPT_COLLECTION    = "receipt_uploads"
UPLOADS_COLLECTION    = "file_uploads"

RECIPES_COLLECTION      = "recipes"
STOCK_COUNTS_COLLECTION = "stock_counts"
STOCK_MOVES_COLLECTION  = "stock_moves"

USE_KRW_CONVERSION = True
KRW_PER_USD = 1350
DEFAULT_INITIAL_STOCK   = 10000
REORDER_THRESHOLD_RATIO = 0.15
SEED_INGREDIENTS = [
    {"ko": "에스프레소", "uom": "g"},
    {"ko": "헤이즐 시럽", "uom": "g"},
    {"ko": "물", "uom": "ml"},
    {"ko": "얼음", "uom": "g"},
    {"ko": "우유", "uom": "ml"},
    {"ko": "연유", "uom": "g"},
    {"ko": "빅트레인 바닐라 파우더", "uom": "g"},
    {"ko": "바닐라빈 시럽", "uom": "g"},
    {"ko": "설탕 시럽", "uom": "ml"},
    {"ko": "꿀", "uom": "ml"}
]
SEED_MENUS = [
    "헤이즐 아메I",
    "카페라떼I",
    "돌체라떼I",
    "바닐라빈라떼I",
    "사케라또I",
]


for p in (DATA_DIR, ASSETS_DIR, KEYS_DIR, RECEIPT_DIR, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)

def safe_doc_id(name: str) -> str:
    """Firestore 문서 ID에 사용할 수 있도록 위험 문자를 정규화합니다."""
    if not name:
        return "unknown"
    return re.sub(r"[/.#\\?\s]+", "_", str(name)).strip("_") or "unknown"


# ----------------------
# 0-1️⃣ Firebase 초기화 (Secrets → keys/ → GOOGLE_APPLICATION_CREDENTIALS)
# (원본 코드 생략)
# ----------------------
@st.cache_resource
def init_firestore():
    """Firebase 인증 및 Firestore 클라이언트 초기화 (중복 호출 방지 + 캐시 적용)"""
    if firebase_admin._apps:
        return firestore.client()
    # 1) 환경변수/파일 우선 (로컬에서 프로젝트를 명확히 지정할 때 사용)
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and Path(gac).expanduser().exists():
        firebase_admin.initialize_app()
        return firestore.client()
    # 2) secrets.toml 우선순위를 *두 번째*로 내려서, 로컬 env 설정을 덮어쓰지 않도록 함
    svc_dict = SECRETS.get("firebase_service_account")
    if isinstance(svc_dict, dict) and svc_dict:
        cred = credentials.Certificate(svc_dict)
        firebase_admin.initialize_app(cred)
        return firestore.client()
    if SA_FILE_PATH.exists():
        cred = credentials.Certificate(str(SA_FILE_PATH))
        firebase_admin.initialize_app(cred)
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

def _guess_receipt_extension(uploaded_file) -> str:
    name = getattr(uploaded_file, "name", "") or ""
    ext = Path(name).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if ext == ".jpeg" else ext
    mime_type = getattr(uploaded_file, "type", "") or ""
    if mime_type == "image/png":
        return ".png"
    if mime_type == "image/webp":
        return ".webp"
    return ".jpg"

def get_storage_bucket_name() -> str | None:
    bucket = SECRETS.get("firebase_storage_bucket") or os.environ.get("FIREBASE_STORAGE_BUCKET")
    if bucket:
        return str(bucket)
    project_id = None
    svc_dict = SECRETS.get("firebase_service_account")
    if isinstance(svc_dict, dict):
        project_id = svc_dict.get("project_id")
    project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if project_id:
        return f"{project_id}.appspot.com"
    return None

def update_receipt_metadata(receipt_id: str | None, payload: dict) -> None:
    if not receipt_id or not payload:
        return
    try:
        db.collection(RECEIPT_COLLECTION).document(receipt_id).set(payload, merge=True)
    except Exception as e:
        st.warning(f"영수증 메타데이터 저장 실패: {e}")

def save_receipt_image(uploaded_file, receipt_kind: str, receipt_id: str | None = None) -> dict | None:
    if uploaded_file is None:
        return None
    receipt_id = receipt_id or uuid.uuid4().hex
    now = datetime.now()
    date_folder = now.strftime("%Y/%m/%d")
    ext = _guess_receipt_extension(uploaded_file)
    file_name = f"{receipt_kind}_{now.strftime('%Y%m%d_%H%M%S')}_{receipt_id}{ext}"
    bytes_data = uploaded_file.getvalue()
    mime_type = getattr(uploaded_file, "type", None) or "image/jpeg"

    local_path = None
    saved_local = False
    try:
        local_dir = RECEIPT_DIR / receipt_kind / date_folder
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / file_name
        local_path.write_bytes(bytes_data)
        saved_local = True
    except Exception as e:
        st.warning(f"영수증 이미지 로컬 저장 실패: {e}")

    storage_path = f"receipts/{receipt_kind}/{date_folder}/{file_name}"
    storage_bucket = None
    storage_uri = None
    saved_storage = False
    try:
        bucket_name = get_storage_bucket_name()
        bucket = storage.bucket(bucket_name) if bucket_name else storage.bucket()
        storage_bucket = bucket.name
        blob = bucket.blob(storage_path)
        blob.metadata = {"receipt_id": receipt_id, "receipt_kind": receipt_kind}
        blob.upload_from_string(bytes_data, content_type=mime_type)
        storage_uri = f"gs://{bucket.name}/{storage_path}"
        saved_storage = True
    except Exception as e:
        st.warning(f"영수증 이미지 Storage 업로드 실패: {e}")

    if not saved_local and not saved_storage:
        return None

    meta = {
        "receipt_id": receipt_id,
        "kind": receipt_kind,
        "created_at": now.isoformat(),
        "file_name": file_name,
        "original_filename": getattr(uploaded_file, "name", None),
        "content_type": mime_type,
        "size_bytes": len(bytes_data),
        "storage_path": storage_path if saved_storage else None,
        "storage_bucket": storage_bucket,
        "storage_uri": storage_uri,
        "local_path": str(local_path) if saved_local and local_path else None,
        "saved_local": saved_local,
        "saved_storage": saved_storage,
        "source": "streamlit",
    }
    update_receipt_metadata(receipt_id, meta)
    return meta

def build_signed_url(storage_bucket: str | None, storage_path: str | None, expires_minutes: int = 60) -> str | None:
    if not storage_bucket or not storage_path:
        return None
    try:
        bucket = storage.bucket(storage_bucket)
        blob = bucket.blob(storage_path)
        return blob.generate_signed_url(expiration=timedelta(minutes=expires_minutes), method="GET")
    except Exception:
        return None

def build_storage_console_url(storage_bucket: str | None, storage_path: str | None) -> str | None:
    if not storage_bucket or not storage_path:
        return None
    return f"https://console.cloud.google.com/storage/browser/_details/{storage_bucket}/{storage_path}"

def build_signed_url_from_meta(meta: dict | None, expires_minutes: int = 60) -> str | None:
    if not meta:
        return None
    return build_signed_url(meta.get("storage_bucket"), meta.get("storage_path"), expires_minutes)

def _guess_upload_extension(uploaded_file) -> str:
    name = getattr(uploaded_file, "name", "") or ""
    ext = Path(name).suffix.lower()
    if ext:
        return ext
    mime_type = getattr(uploaded_file, "type", "") or ""
    if mime_type == "text/csv":
        return ".csv"
    if mime_type in {"application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}:
        return ".xlsx"
    return ".bin"

def update_upload_metadata(upload_id: str | None, payload: dict) -> None:
    if not upload_id or not payload:
        return
    try:
        db.collection(UPLOADS_COLLECTION).document(upload_id).set(payload, merge=True)
    except Exception as e:
        st.warning(f"업로드 메타데이터 저장 실패: {e}")

def save_data_upload_file(uploaded_file,
                          upload_kind: str,
                          category: str = "excel",
                          upload_id: str | None = None,
                          bytes_data: bytes | None = None) -> dict | None:
    if uploaded_file is None:
        return None
    upload_id = upload_id or uuid.uuid4().hex
    now = datetime.now()
    date_folder = now.strftime("%Y/%m/%d")
    ext = _guess_upload_extension(uploaded_file)
    file_name = f"{upload_kind}_{now.strftime('%Y%m%d_%H%M%S')}_{upload_id}{ext}"
    bytes_data = bytes_data or uploaded_file.getvalue()
    mime_type = getattr(uploaded_file, "type", None) or "application/octet-stream"

    local_path = None
    saved_local = False
    try:
        local_dir = UPLOAD_DIR / category / upload_kind / date_folder
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / file_name
        local_path.write_bytes(bytes_data)
        saved_local = True
    except Exception as e:
        st.warning(f"업로드 파일 로컬 저장 실패: {e}")

    storage_path = f"uploads/{category}/{upload_kind}/{date_folder}/{file_name}"
    storage_bucket = None
    storage_uri = None
    saved_storage = False
    try:
        bucket_name = get_storage_bucket_name()
        bucket = storage.bucket(bucket_name) if bucket_name else storage.bucket()
        storage_bucket = bucket.name
        blob = bucket.blob(storage_path)
        blob.metadata = {"upload_id": upload_id, "upload_kind": upload_kind, "category": category}
        blob.upload_from_string(bytes_data, content_type=mime_type)
        storage_uri = f"gs://{bucket.name}/{storage_path}"
        saved_storage = True
    except Exception as e:
        st.warning(f"업로드 파일 Storage 업로드 실패: {e}")

    if not saved_local and not saved_storage:
        return None

    meta = {
        "upload_id": upload_id,
        "kind": upload_kind,
        "category": category,
        "created_at": now.isoformat(),
        "file_name": file_name,
        "original_filename": getattr(uploaded_file, "name", None),
        "content_type": mime_type,
        "size_bytes": len(bytes_data),
        "storage_path": storage_path if saved_storage else None,
        "storage_bucket": storage_bucket,
        "storage_uri": storage_uri,
        "local_path": str(local_path) if saved_local and local_path else None,
        "saved_local": saved_local,
        "saved_storage": saved_storage,
        "source": "streamlit",
    }
    update_upload_metadata(upload_id, meta)
    return meta

def save_upload_file_once(uploaded_file,
                          session_prefix: str,
                          upload_kind: str,
                          category: str = "excel") -> dict | None:
    if uploaded_file is None:
        st.session_state.pop(f"{session_prefix}_sig", None)
        st.session_state.pop(f"{session_prefix}_meta", None)
        return None
    bytes_data = uploaded_file.getvalue()
    signature = f"{getattr(uploaded_file, 'name', '')}:{len(bytes_data)}"
    sig_key = f"{session_prefix}_sig"
    meta_key = f"{session_prefix}_meta"
    if st.session_state.get(sig_key) != signature:
        meta = save_data_upload_file(
            uploaded_file,
            upload_kind=upload_kind,
            category=category,
            bytes_data=bytes_data,
        )
        st.session_state[sig_key] = signature
        st.session_state[meta_key] = meta
    return st.session_state.get(meta_key)

def normalize_receipt_date(raw: str | None, default):
    if not raw:
        return default
    parsed = pd.to_datetime(str(raw).strip(), errors="coerce")
    if pd.isna(parsed):
        return default
    return parsed.date()

def normalize_receipt_time(raw: str | None, default: str) -> str:
    if not raw:
        return default
    s = str(raw).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).strftime("%H:%M:%S")
        except Exception:
            continue
    return default

# === [AI/ML 통합 추가] ===
# SPRINT 1: Gemini API 키 설정
# 기본값: 키가 접근 가능한 2.5 세대 모델 우선 (필요 시 하위 버전으로 폴백)
GEMINI_TEXT_MODEL = (
    SECRETS.get("gemini", {}).get("text_model")
    or os.environ.get("GEMINI_TEXT_MODEL")
    or "gemini-2.5-flash"
)
GEMINI_VISION_MODEL = (
    SECRETS.get("gemini", {}).get("vision_model")
    or os.environ.get("GEMINI_VISION_MODEL")
    or "gemini-2.5-flash"
)
GEMINI_TEXT_MODEL_CANDIDATES = [
    GEMINI_TEXT_MODEL,
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-flash-latest",
    "gemini-pro-latest",
]
GEMINI_VISION_MODEL_CANDIDATES = [
    GEMINI_VISION_MODEL,
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-flash-latest",
]
GEMINI_API_KEY = None
GEMINI_CLIENT = None

try:
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
except (KeyError, AttributeError):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Gemini API 키 설정 중 오류가 발생했습니다: {e}")
else:
    st.warning("""
    Gemini API 키가 'secrets.toml'에 설정되지 않았습니다. 
    AI 비서 기능이 작동하지 않거나 Mock 데이터로 작동합니다.
    [.streamlit/secrets.toml] 파일에 [gemini] api_key = "..."를 추가하거나 환경변수 GEMINI_API_KEY를 설정하세요.
    """)
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

NAME_MAP = {
    # 불일치 정규화: Firestore/레시피 이름 -> 증강 CSV 기준 이름
    # --- 영문 변형 (언더바/공백/슬래시) ---
    "Americano_(I_H)": "Americano (I/H)",
    "Americano (I H)": "Americano (I/H)",
    "Caffè_Latte_(I_H)": "Caffè Latte (I/H)",
    "Latte": "Caffè Latte (I/H)",
    "Hazelnut_Americano_(Iced)": "Hazelnut Americano (Iced)",
    "Honey_Americano_(Iced)": "Honey Americano (Iced)",
    "Shakerato_(Iced)": "Shakerato (Iced)",
    "Vanilla_Bean_Latte_(Iced)": "Vanilla Bean Latte (Iced)",
    "Dolce_Latte_(Iced)": "Dolce Latte (Iced)",
    # --- 한글/기존 레거시 변형 ---
    "카페라떼I": "Caffè Latte (I/H)",
    "카페라떼": "Caffè Latte (I/H)",
    "헤이즐 아메I": "Hazelnut Americano (Iced)",
    "헤이즐_아메I": "Hazelnut Americano (Iced)",
    "헤이즐 아메": "Hazelnut Americano (Iced)",
    "꿀 아메": "Honey Americano (Iced)",
    "사케라또I": "Shakerato (Iced)",
    "사케라또": "Shakerato (Iced)",
    "바닐라빈라떼I": "Vanilla Bean Latte (Iced)",
    "바닐라빈라떼": "Vanilla Bean Latte (Iced)",
    "돌체라떼I": "Dolce Latte (Iced)",
    "돌체라떼": "Dolce Latte (Iced)",
    # CSV에 없는 메뉴는 매핑하지 않음 (수동 삭제/재입력 대상)
}

def apply_name_map(name: str | None) -> str:
    """증강 CSV 기준 메뉴 이름으로 정규화."""
    if name is None:
        return ""
    raw = str(name).strip()
    # 1) 직접 매핑 키
    if raw in NAME_MAP:
        return NAME_MAP[raw]
    # 2) 언더바/슬래시/공백 변형 시도
    variants = {
        raw,
        raw.replace("_", " "),
        re.sub(r"\(([^)]+)_([^)]+)\)", r"(\1/\2)", raw),
        raw.replace("(I H)", "(I/H)"),
        raw.replace("(I/H)", "(I H)"),
    }
    for v in variants:
        if v in NAME_MAP:
            return NAME_MAP[v]
    return raw

def build_menu_candidates(name: str) -> set[str]:
    """메뉴 이름의 다양한 변형(언더바, 슬래시, 사이즈 접미사 제거)을 모두 포함한 후보 집합 생성."""
    raw = str(name or "").strip()
    cleaned = raw.replace("_", " ")
    slash = re.sub(r"\(([^)]+)_([^)]+)\)", r"(\1/\2)", raw)
    base = re.sub(r"\s+(Lg|Rg|Sm)$", "", cleaned)
    variants = {
        raw,
        cleaned,
        slash,
        base,
        base.replace("(I H)", "(I/H)"),
        base.replace("(I/H)", "(I H)"),
    }
    # 소문자/대문자 혼용 방지
    variants |= {v.lower() for v in variants}
    # 한글 변환 후보 추가
    variants |= {to_korean_detail(v) for v in list(variants)}
    return variants

def normalize_menu_key(value: str) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\b(iced|ice|hot|cold|warm)\b", " ", s)
    s = re.sub(r"\b(lg|rg|sm|large|medium|small|regular)\b", " ", s)
    s = re.sub(r"(아이스|핫|차가운|뜨거운|라지|레귤러|스몰|대|중|소)", " ", s)
    s = re.sub(r"i\s*/\s*h", " ", s)
    s = re.sub(r"[(){}\[\]/\\_,.\-]+", " ", s)
    s = re.sub(r"\s+", "", s)
    return s

def build_menu_fuzzy_index(menu_names: list[str]) -> list[tuple[str, str]]:
    norm_map: dict[str, set[str]] = {}
    for name in sorted(set(menu_names)):
        if not name:
            continue
        for cand in {name, to_korean_detail(name), from_korean_detail(name)}:
            norm = normalize_menu_key(cand)
            if norm:
                norm_map.setdefault(norm, set()).add(name)
    index: list[tuple[str, str]] = []
    for norm, names in norm_map.items():
        if len(names) == 1:
            index.append((norm, next(iter(names))))
    return index

def build_menu_lookup(menu_names: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in menu_names:
        for cand in build_menu_candidates(name):
            lookup.setdefault(cand, name)
        compact = re.sub(r"\s+", "", str(name)).lower()
        if compact:
            lookup.setdefault(compact, name)
    return lookup

def match_menu_name(raw_name: str,
                    menu_lookup: dict[str, str],
                    fuzzy_index: list[tuple[str, str]] | None = None) -> tuple[str, bool]:
    raw = str(raw_name or "").strip()
    if not raw:
        return "", False
    candidates = build_menu_candidates(raw)
    mapped = apply_name_map(raw)
    candidates |= build_menu_candidates(mapped)
    candidates.add(re.sub(r"\s+", "", raw).lower())
    for cand in candidates:
        if cand in menu_lookup:
            return menu_lookup[cand], True
    if fuzzy_index:
        raw_norms = {normalize_menu_key(raw), normalize_menu_key(mapped)}
        best_name = ""
        best_score = 0.0
        second_score = 0.0
        for raw_norm in raw_norms:
            if not raw_norm:
                continue
            for cand_norm, menu_name in fuzzy_index:
                score = difflib.SequenceMatcher(None, raw_norm, cand_norm).ratio()
                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_name = menu_name
                elif score > second_score:
                    second_score = score
        if best_name and best_score >= 0.86 and (best_score - second_score) >= 0.05:
            return best_name, True
    return raw, False

def normalize_column_key(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").strip().lower())

def pick_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    col_lookup = {normalize_column_key(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_column_key(alias)
        if key in col_lookup:
            return col_lookup[key]
    return None

def read_tabular_upload(uploaded_file, bytes_data: bytes | None = None) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    name = str(getattr(uploaded_file, "name", "")).lower()
    try:
        if bytes_data is None:
            bytes_data = uploaded_file.getvalue()
        buf = io.BytesIO(bytes_data)
        if name.endswith(".csv"):
            return pd.read_csv(buf)
        return pd.read_excel(buf)
    except Exception as e:
        st.error(f"엑셀/CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def normalize_cell_str(value, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    s = str(value).strip()
    return s if s else default

def parse_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value or "").strip().lower()
    if s in {"y", "yes", "true", "1", "t", "o"}:
        return True
    if s in {"n", "no", "false", "0", "f", "x"}:
        return False
    return False

def normalize_inventory_key(value: str) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[(){}\[\]/\\_,.\-]+", "", s)
    s = re.sub(r"\d+(?:\.\d+)?\s*(ml|l|kg|g|ea|개|팩|box|봉|병|pcs|pc)", "", s, flags=re.I)
    return s

def build_inventory_candidates(name: str) -> set[str]:
    raw = str(name or "").strip()
    if not raw:
        return set()
    base = {raw, apply_name_map(raw)}
    candidates: set[str] = set()
    for v in base:
        if not v:
            continue
        candidates.add(v)
        candidates.add(v.lower())
        candidates.add(re.sub(r"\s+", "", v).lower())
        candidates.add(re.sub(r"[(){}\[\]/\\_,.\-]+", "", v).lower())
        no_unit = re.sub(r"\b\d+(?:\.\d+)?\s*(ml|l|kg|g|ea|개|팩|box|봉|병|pcs|pc)\b", "", v, flags=re.I).strip()
        if no_unit:
            candidates.add(no_unit)
            candidates.add(re.sub(r"\s+", "", no_unit).lower())
        candidates.add(to_korean_detail(v))
        candidates.add(from_korean_detail(v))
    candidates |= {normalize_inventory_key(v) for v in list(candidates) if v}
    return {c for c in candidates if c}

def build_inventory_fuzzy_index(df_inv: pd.DataFrame) -> list[tuple[str, str]]:
    index: list[tuple[str, str]] = []
    if df_inv is None or df_inv.empty:
        return index
    for _, row in df_inv.iterrows():
        sku_en = str(row.get("상품상세_en") or row.get("상품상세") or "").strip()
        if not sku_en:
            continue
        for name in (row.get("상품상세"), row.get("상품상세_en")):
            norm = normalize_inventory_key(name)
            if norm:
                index.append((norm, sku_en))
    return index

def build_inventory_lookup(df_inv: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if df_inv is None or df_inv.empty:
        return lookup
    for _, row in df_inv.iterrows():
        ko = str(row.get("상품상세", "")).strip()
        en = str(row.get("상품상세_en", "")).strip()
        target = en or from_korean_detail(ko) or ko
        keys = build_inventory_candidates(ko) | build_inventory_candidates(en)
        for key in keys:
            if key:
                lookup.setdefault(key, target)
    return lookup

def match_inventory_name(raw_name: str,
                         inv_lookup: dict[str, str],
                         fuzzy_index: list[tuple[str, str]] | None = None) -> tuple[str, bool]:
    raw = str(raw_name or "").strip()
    if not raw:
        return "", False
    candidates = build_inventory_candidates(raw)
    for key in candidates:
        if key in inv_lookup:
            return inv_lookup[key], True
    if fuzzy_index:
        raw_norm = normalize_inventory_key(raw)
        if raw_norm:
            best_en = ""
            best_score = 0.0
            second_score = 0.0
            for cand_norm, sku_en in fuzzy_index:
                score = difflib.SequenceMatcher(None, raw_norm, cand_norm).ratio()
                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_en = sku_en
                elif score > second_score:
                    second_score = score
            if best_en and best_score >= 0.86 and (best_score - second_score) >= 0.05:
                return best_en, True
    return apply_name_map(from_korean_detail(raw)), False

SALES_UPLOAD_ALIASES = {
    "name": ["상품상세", "상품명", "메뉴", "품목", "item", "name", "menu", "product"],
    "qty": ["수량", "qty", "quantity", "count"],
    "price": ["단가", "price", "unitprice", "unit_price", "가격"],
    "total": ["총액", "total", "amount", "금액", "매출"],
    "date": ["날짜", "date", "거래일", "거래날짜"],
    "time": ["시간", "time", "거래시간"],
    "category": ["상품카테고리", "카테고리", "category"],
    "type": ["상품타입", "타입", "type"],
    "channel": ["채널", "channel", "구분"],
}

INVENTORY_UPLOAD_ALIASES = {
    "name": ["상품상세", "품목", "품명", "품목명", "제품명", "재료", "item", "name", "item_name", "material"],
    "sku": ["상품상세_en", "상품코드", "sku", "sku_en", "code", "item_code"],
    "qty": ["수량", "qty", "quantity", "입고수량", "입고량", "입고", "재고", "재고수량", "stock"],
    "uom": ["단위", "uom", "unit", "규격"],
    "cost_unit_size": ["매입단위", "단위수량", "cost_unit_size", "unit_size"],
    "cost_per_unit": ["매입가", "cost_per_unit", "매입가격", "price", "cost", "unit_cost"],
    "is_ingredient": ["재료여부", "is_ingredient", "ingredient"],
}

from typing import Union


# ----------------------
# ✅ UoM(단위) 유틸
# (원본 코드 생략)
# ----------------------
def normalize_uom(u: Union[str, None]) -> str:
    u = (u or "ea").strip().lower()
    if u in {"g", "gram", "grams", "그램", "kg", "킬로그램"}:
        return "g"
    if u in {"ml", "밀리리터", "l", "리터"}:
        return "ml"
    return "ea"

def _uom_base_and_factor(u: Union[str, None]) -> tuple[str, float]:
    s = (u or "").strip().lower()
    if s in {"kg", "킬로그램"}:
        return "g", 1000.0
    if s in {"g", "gram", "grams", "그램"}:
        return "g", 1.0
    if s in {"l", "리터"}:
        return "ml", 1000.0
    if s in {"ml", "밀리리터"}:
        return "ml", 1.0
    if s in {"ea", "개", "pcs", "pc", "piece", "pieces", "팩", "봉", "병", "box"}:
        return "ea", 1.0
    if not s:
        return "ea", 1.0
    return normalize_uom(s), 1.0

def convert_qty(qty: float, from_uom: str, to_uom: str) -> float:
    fu, f_factor = _uom_base_and_factor(from_uom)
    tu, t_factor = _uom_base_and_factor(to_uom)
    if fu != tu:
        return float(qty)
    try:
        return float(qty) * f_factor / t_factor
    except Exception:
        return float(qty)

def convert_stock_to_cups(qty: float, uom: str, grams_per_cup: float = DEFAULT_GRAMS_PER_CUP) -> float:
    """g 단위를 잔(컵) 기준으로 환산."""
    if normalize_uom(uom) != "g":
        return float(qty)
    if grams_per_cup <= 0:
        return float(qty)
    return float(qty) / grams_per_cup

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


def estimate_ingredient_daily_usage(df_sales: pd.DataFrame, recipes: dict, days: int = 30) -> dict:
    """최근 N일간 판매+레시피 기반 재료별 일평균 소진량 계산."""
    if df_sales is None or df_sales.empty or not recipes:
        return {}
    df_use = df_sales.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_use.get("날짜")):
        try:
            df_use["날짜"] = pd.to_datetime(df_use["날짜"], errors="coerce")
        except Exception:
            return {}
    df_use = df_use.dropna(subset=["날짜"])
    if df_use.empty:
        return {}
    cutoff = df_use["날짜"].max() - pd.Timedelta(days=days - 1)
    df_use = df_use[df_use["날짜"] >= cutoff]
    if df_use.empty:
        return {}

    usage_map: dict[str, float] = {}
    for _, row in df_use.iterrows():
        try:
            qty = safe_float(row.get("수량", 1), 1)
            menu_en = from_korean_detail(row.get("상품상세"))
            ingredients = recipes.get(menu_en, [])
            for item in ingredients:
                base_qty = safe_float(item.get("qty", 0.0), 0.0)
                waste_pct = safe_float(item.get("waste_pct", 0.0), 0.0)
                total_used = (base_qty * qty) * (1 + (waste_pct / 100.0))
                sku = item.get("ingredient_en")
                if not sku:
                    continue
                usage_map[sku] = usage_map.get(sku, 0.0) + total_used
        except Exception:
            continue

    days_span = max((df_use["날짜"].max() - df_use["날짜"].min()).days + 1, 1)
    return {k: v / days_span for k, v in usage_map.items()}

# ----------------------
# 1️⃣ CSV 로드 (샘플 생성 없음)
# (원본 코드 생략)
# ----------------------
@st.cache_data(ttl=3600) 
def load_csv_FINAL(path: Path): # [Pylance 오류] 타입 힌트 제거
    """
    증강/현장 CSV(`데이터 증강.csv`)를 로드하고 수익 컬럼을 계산합니다.
    예상 스키마: timestamp, menu_item, price, day_of_week, hour, day_type (+optional quantity, category)
    """
    if not path.exists():
        st.error(f"CSV를 찾을 수 없습니다. (경로: {path})")
        st.stop()
    
    start_time = time.time()
    df_raw = pd.read_csv(path)

    required_cols = {'timestamp', 'menu_item', 'price'}
    if not required_cols.issubset(df_raw.columns):
        st.error(f"CSV 스키마가 예상과 다릅니다. 필수 컬럼: {required_cols}")
        st.stop()

    df = df_raw.copy()
    df['menu_item'] = df['menu_item'].apply(apply_name_map)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['날짜'] = df['timestamp'].dt.normalize()
    df['시간'] = df['timestamp'].dt.strftime('%H:%M:%S')

    hour_series = df_raw['hour'] if 'hour' in df_raw.columns else df['timestamp'].dt.hour
    df['시'] = pd.to_numeric(hour_series, errors='coerce')

    qty_series = df_raw['quantity'] if 'quantity' in df_raw.columns else pd.Series(1, index=df.index)
    df['수량'] = pd.to_numeric(qty_series, errors='coerce').fillna(1)
    df['단가'] = pd.to_numeric(df_raw['price'], errors='coerce')
    df['수익'] = df['수량'] * df['단가']

    menu_series = df_raw['menu_item'] if 'menu_item' in df_raw.columns else pd.Series("미확인 메뉴", index=df.index)
    menu_series = menu_series.apply(apply_name_map)
    df['상품상세'] = menu_series.fillna("미확인 메뉴")

    type_series = df_raw['menu_item'] if 'menu_item' in df_raw.columns else pd.Series("기타", index=df.index)
    df['상품타입'] = type_series.fillna("기타")

    category_series = df_raw['menu_item'] if 'menu_item' in df_raw.columns else pd.Series("기타", index=df.index)
    df['상품카테고리'] = category_series.fillna("기타")

    dow_series = df_raw['day_of_week'] if 'day_of_week' in df_raw.columns else df['날짜'].dt.day_name()
    df['요일'] = pd.Series(dow_series).fillna(df['날짜'].dt.day_name())
    df['월'] = df['날짜'].dt.month

    df['거래번호'] = df.index + 1
    df['가게ID'] = "LOCAL"
    df['가게위치'] = "증강데이터"

    df = df.dropna(subset=['날짜', '수익'])
    
    end_time = time.time()
    load_time = end_time - start_time
    row_count_final = len(df)
    
    return df, load_time, row_count_final

df_csv, load_time, row_count = load_csv_FINAL(CSV_PATH)
# 증강 CSV 기준 7개 메뉴를 고정 리스트로 사용 (캐시/데이터 소스와 무관하게 동일하게 노출)
MENU_MASTER_EN = [
    "Americano (I/H)",
    "Caffè Latte (I/H)",
    "Dolce Latte (Iced)",
    "Hazelnut Americano (Iced)",
    "Honey Americano (Iced)",
    "Shakerato (Iced)",
    "Vanilla Bean Latte (Iced)",
]
MENU_MASTER_KR = [to_korean_detail(m) for m in MENU_MASTER_EN]


@st.cache_data(ttl=600)
def load_augmented_sales(path: Path = CSV_AUGMENTED_PATH):
    """데이터 증강 CSV를 로드해 간단한 변환 컬럼을 추가합니다."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['hour'] = pd.to_numeric(df.get('hour'), errors='coerce')
    df['date'] = df['timestamp'].dt.date
    return df


@st.cache_data(ttl=600)
def load_product_status(path: Path = CSV_PRODUCT_STATUS_PATH):
    """상품매출현황 CSV를 로드해 숫자 필드를 정리합니다."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={
        '상 품 명': '상품명',
        '상 품 코 드': '상품코드',
        '수    량': '수량'
    })
    num_cols = ['수량', '점유율(수량)', '판매금액', '점유율(금액)']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.replace('%', ''),
                errors='coerce'
            )
    return df


@st.cache_data(ttl=600)
def load_hourly_sales(path: Path = CSV_HOURLY_PATH):
    """시간대별 매출분석 CSV를 로드해 숫자 필드를 정리합니다."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={'시간': 'hour'})
    for col in df.columns:
        if col == 'hour':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_top5_recipe(path: Path = CSV_TOP5_PATH):
    """카피엔드 Top5 레시피 CSV 로드."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    num_cols = ['단가(원)', '수량', '개별가', '사용량', '사용 단가', '사용단가 합계', '판매가격', '원가율']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    return df


def build_recipes_from_top5(df_top: pd.DataFrame) -> dict:
    """카피엔드 Top5 CSV를 레시피 딕셔너리로 변환."""
    if df_top is None or df_top.empty:
        return {}
    df = df_top.copy()
    df['메 뉴'] = df['메 뉴'].ffill()
    df['품 목'] = df['품 목'].ffill()
    df = df.dropna(subset=['메 뉴', '품 목'])
    df['사용량'] = pd.to_numeric(df.get('사용량'), errors='coerce')
    recipe_map: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        menu_kr = str(row.get('메 뉴', '')).strip()
        ing_kr = str(row.get('품 목', '')).strip()
        qty = safe_float(row.get('사용량', 0))
        uom_raw = str(row.get('단위', '')).strip()
        # 단위 문자열에서 알파벳 부분만 추출 (예: "40g" -> "g")
        uom_match = re.search(r'([a-zA-Z]+)', uom_raw)
        uom = uom_match.group(1) if uom_match else 'ea'
        if not menu_kr or not ing_kr or qty <= 0:
            continue
        menu_en = from_korean_detail(menu_kr)
        ing_en = from_korean_detail(ing_kr)
        recipe_map.setdefault(menu_en, []).append({
            "ingredient_en": ing_en,
            "qty": qty,
            "uom": normalize_uom(uom),
            "waste_pct": 0.0,
        })
    return recipe_map


def build_top5_cost_map(df_top: pd.DataFrame) -> dict:
    """Top5 CSV에서 재료별 단위 원가 fallback 맵을 생성."""
    if df_top is None or df_top.empty:
        return {}
    df = df_top.copy()
    df['품 목'] = df['품 목'].ffill()
    df = df.dropna(subset=['품 목'])
    df['사용량'] = pd.to_numeric(df.get('사용량'), errors='coerce')
    df['사용 단가'] = pd.to_numeric(df.get('사용 단가'), errors='coerce')
    cost_map = {}
    for _, row in df.iterrows():
        ing_kr = str(row.get('품 목', '')).strip()
        qty = safe_float(row.get('사용량', 0))
        use_cost = safe_float(row.get('사용 단가', 0))
        uom_raw = str(row.get('단위', '')).strip()
        uom_match = re.search(r'([a-zA-Z]+)', uom_raw)
        uom = uom_match.group(1) if uom_match else 'ea'
        if not ing_kr or qty <= 0 or use_cost <= 0:
            continue
        ing_en = from_korean_detail(ing_kr)
        # 사용 단가를 사용량으로 나눠 단위당 원가 추정
        unit_cost_est = use_cost / qty if qty else 0
        cost_map[ing_en] = {
            "unit_cost": unit_cost_est,
            "uom": normalize_uom(uom),
            "use_qty": qty,
            "use_cost": use_cost,
        }
    return cost_map


@st.cache_data(ttl=3600)
def run_prophet_backtesting(df_input, test_days=30): # [Pylance 오류] 타입 힌트 제거
    """
    '예측'이 아닌 '연구 검증'을 수행합니다.
    """
    
    if df_input is None or df_input.empty:
        return None, None, "오류: 입력 데이터가 없습니다."
        
    # 1. 데이터 전처리 (Prophet 형식: ds, y)
    if '수익' not in df_input.columns or '날짜' not in df_input.columns:
        st.error(f"치명적 오류: 백테스팅에 필요한 '날짜' 또는 '수익' 컬럼이 df에 없습니다.")
        return None, None, "데이터 컬럼명 오류"
        
    df_prophet = df_input[['날짜', '수익']].copy()
    
    df_prophet = df_prophet.rename(columns={'날짜': 'ds', '수익': 'y'})
    df_prophet = df_prophet.groupby('ds').sum().reset_index()
    total_points = len(df_prophet)

    if total_points < 5:
        return None, None, "데이터 포인트가 너무 적습니다."

    max_test_days = max(1, total_points - 10)
    if max_test_days < 3:
        return None, None, "검증할 만큼 데이터가 부족합니다."

    effective_test_days = min(test_days, max_test_days)

    # 2. 훈련/테스트 데이터 분리
    split_date = df_prophet['ds'].max() - pd.to_timedelta(effective_test_days, 'D')
    train_data = df_prophet[df_prophet['ds'] <= split_date]
    test_data = df_prophet[df_prophet['ds'] > split_date]

    if len(train_data) < 10:
        return None, None, "오류: 훈련 데이터가 너무 적습니다."

    # 3. 모델 훈련 (데이터 기간이 짧으므로 yearly_seasonality=False)
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
    m.fit(train_data)

    # 4. 예측
    future_frame = m.make_future_dataframe(periods=test_days, freq='D')
    forecast = m.predict(future_frame)
    
    # 5. 예측 결과와 실제 테스트 데이터 병합
    comparison_df = pd.merge(test_data[['ds', 'y']], forecast[['ds', 'yhat']], on='ds')

    # 6. MAPE 계산
    comparison_df = comparison_df[comparison_df['y'] > 0] # 0으로 나누기 방지
    if comparison_df.empty:
        return None, None, "오류: MAPE 계산을 위한 유효한 비교 데이터가 없습니다. ('수익' 컬럼이 0 또는 NaN일 수 있습니다)"
        
    mape = mean_absolute_percentage_error(comparison_df['y'], comparison_df['yhat']) * 100
    
    # 7. 시각화
    fig = m.plot(forecast)
    ax = fig.gca()
    ax.plot(test_data['ds'], test_data['y'], 'r.', label='Actual Test Data (실제값)')
    ax.legend()

    return mape, fig, f"모델 검증 완료 (테스트 기간: {test_days}일)"


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
    if '상품상세' in df_fb.columns:
        df_fb['상품상세'] = df_fb['상품상세'].apply(apply_name_map)
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
        ko = doc.get("상품상세") or to_korean_detail(en)
        
        # [L4] 원가 정보 로드
        cost_unit_size = safe_float(doc.get("cost_unit_size", 1.0), 1.0)
        cost_per_unit = safe_float(doc.get("cost_per_unit", 0.0), 0.0)
        
        # 1g/1ml/1ea당 원가 계산 (0으로 나누기 방지)
        calculated_unit_cost = cost_per_unit / cost_unit_size if cost_unit_size > 0 else 0.0
        
        # [Fix] 저장된 unit_cost가 있으면 우선 사용, 없으면 계산값 사용
        stored_unit_cost = safe_float(doc.get("unit_cost", 0.0), 0.0)
        unit_cost = stored_unit_cost if stored_unit_cost > 0 else calculated_unit_cost
        
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
            "unit_cost": unit_cost,           # 1g/ml/ea당 원가 (e.g., 30)

            # [New] 판매가 (메뉴용)
            "sale_price": safe_float(doc.get("sale_price", 0.0), 0.0),

            # 공급 방식/리드타임 (UX 개선)
            "supply_mode": doc.get("supply_mode", DEFAULT_SUPPLY_MODE),
            "supply_lead_days": safe_float(doc.get("supply_lead_days", DEFAULT_SUPPLY_LEAD_DAYS)),
        })
    
    # === [빈틈 수정] inventory가 비어있어도 컬럼은 유지 ===
    df = pd.DataFrame(rows, columns=[
        "상품상세_en", "상품상세", "초기재고", "현재재고", "uom", "is_ingredient",
        "cost_unit_size", "cost_per_unit", "unit_cost", "sale_price",
        "supply_mode", "supply_lead_days" # 공급 정보
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
            "_id","sku_en","lead_time_days","safety_stock_units","target_days","grams_per_cup","expiry_days",
            "supply_mode","supply_lead_days"
        ])
    defaults = {
        "lead_time_days": 3,
        "safety_stock_units": 10,
        "target_days": 21,
        "grams_per_cup": 18.0,
        "expiry_days": 28,
        "supply_mode": DEFAULT_SUPPLY_MODE,
        "supply_lead_days": DEFAULT_SUPPLY_LEAD_DAYS,
    }
    for col, default in defaults.items():
        if col not in dfp.columns:
            dfp[col] = default
        else:
            if isinstance(default, str):
                dfp[col] = dfp[col].fillna(default)
            else:
                dfp[col] = pd.to_numeric(dfp[col], errors="coerce").fillna(default)
    return dfp

# --- 캐시 안전 초기화 헬퍼 ---
def clear_cache_safe(*funcs):
    """Streamlit cache 함수의 clear()를 안전하게 호출합니다."""
    for fn in funcs:
        clear_fn = getattr(fn, "clear", None)
        if callable(clear_fn):
            try:
                clear_fn()
            except Exception:
                pass

# --- 3. 헬퍼 함수 정의 (정의 3: Ensure Inventory Doc) ---
def ensure_inventory_doc(product_detail_en: str, uom: str = "ea", is_ingredient: bool = False):
    ref = db.collection(INVENTORY_COLLECTION).document(safe_doc_id(product_detail_en))
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
            "supply_mode": DEFAULT_SUPPLY_MODE,
            "supply_lead_days": DEFAULT_SUPPLY_LEAD_DAYS,
        })
        return ref

def ensure_ingredient_sku(ingredient_en: str, uom: str = "ea"):
    return ensure_inventory_doc(ingredient_en, uom=uom, is_ingredient=True)

def ensure_inventory_ingredient(ingredient_name: str, uom: str = "g") -> str:
    name = str(ingredient_name or "").strip()
    if not name:
        return ""
    ingredient_en = from_korean_detail(name)
    ref = db.collection(INVENTORY_COLLECTION).document(safe_doc_id(ingredient_en))
    snap = ref.get()
    if snap.exists:
        data = snap.to_dict() or {}
        patch = {}
        if not data.get("상품상세") and name:
            patch["상품상세"] = name
        if not bool(data.get("is_ingredient", False)):
            patch["is_ingredient"] = True
            if normalize_uom(data.get("uom")) != normalize_uom(uom):
                patch["uom"] = normalize_uom(uom)
        elif not data.get("uom"):
            patch["uom"] = normalize_uom(uom)
        if patch:
            ref.set(patch, merge=True)
    else:
        ref.set({
            "상품상세_en": ingredient_en,
            "상품상세": name,
            "is_ingredient": True,
            "uom": normalize_uom(uom),
            "초기재고": 0.0,
            "현재재고": 0.0,
            "cost_unit_size": 1.0,
            "cost_per_unit": 0.0,
            "unit_cost": 0.0,
            "supply_mode": DEFAULT_SUPPLY_MODE,
            "supply_lead_days": DEFAULT_SUPPLY_LEAD_DAYS,
        })
    return ingredient_en


def ensure_seed_ingredients():
    """Top5 레시피 핵심 재료를 inventory에 기본 등록합니다."""
    for item in SEED_INGREDIENTS:
        ko = item["ko"]
        uom = item["uom"]
        en = from_korean_detail(ko)
        ensure_inventory_doc(en, uom=uom, is_ingredient=True)

def ensure_seed_menus():
    """Top5 메뉴를 inventory에 '완제품'으로 기본 등록합니다."""
    for menu_ko in SEED_MENUS:
        menu_en = from_korean_detail(menu_ko)
        ref = db.collection(INVENTORY_COLLECTION).document(safe_doc_id(menu_en))
        snap = ref.get()
        if snap.exists:
            continue
        ref.set({
            "상품상세_en": menu_en,
            "상품상세": menu_ko,
            "is_ingredient": False,
            "uom": "ea",
            "초기재고": 0.0,
            "현재재고": 0.0,
            "cost_unit_size": 1.0,
            "cost_per_unit": 0.0,
            "unit_cost": 0.0,
        })


def reset_inventory_to_seed():
    """모든 inventory 문서를 삭제 후 시드 재료만 다시 채웁니다."""
    try:
        docs = list(db.collection(INVENTORY_COLLECTION).stream())
        for d in docs:
            db.collection(INVENTORY_COLLECTION).document(d.id).delete()
        ensure_seed_ingredients()
        ensure_seed_menus()
        return len(docs)
    except Exception as e:
        st.error(f"인벤토리 초기화 실패: {e}")
        return None


# --- 4. 메인 데이터 로딩 함수 (호출 1) ---
@st.cache_data(ttl=60)
def load_all_core_data():
    """
    [L4 수정] 앱 실행 시 모든 핵심 데이터를 로드합니다.
    (이제 이 함수가 호출되어도, 필요한 함수들이 '위에' 정의되어 있습니다.)
    """
    # 0. 핵심 재료가 inventory에 없으면 선등록
    ensure_seed_ingredients()
    ensure_seed_menus()

    # 1. Sales (df)
    df = pd.concat([df_csv, df_fb], ignore_index=True)
    if '상품상세' in df.columns:
        df['상품상세'] = df['상품상세'].apply(apply_name_map)
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
            if not data or "ingredients" not in data:
                continue
            menu_en = data.get("menu_sku_en") or apply_name_map(d.id)
            menu_en = apply_name_map(menu_en)
            if not menu_en:
                continue
            recipes[menu_en] = data["ingredients"]
    except Exception as e:
        st.error(f"레시피 로드 실패: {e}")

    # 3-1. Top5 CSV를 레시피/원가로 병합 (Firestore 없을 때 기본 레시피/원가)
    df_top_local = load_top5_recipe()
    top5_cost_map = build_top5_cost_map(df_top_local) if df_top_local is not None else {}
    csv_recipes = build_recipes_from_top5(df_top_local) if df_top_local is not None else {}
    for menu_en, ing_list in csv_recipes.items():
        if menu_en not in recipes or not recipes[menu_en]:
            recipes[menu_en] = ing_list
        
    # 4. Params (df_params)
    df_params = load_sku_params()
    
    return df, df_inv, recipes, df_params, top5_cost_map

# --- 5. 메인 데이터 로드 '실행' ---
try:
    #data_load_state = st.info("모든 핵심 데이터(판매, 재고, 레시피) 로드 중... ⏳")
    df, df_inv, RECIPES, df_params, TOP5_COST_MAP = load_all_core_data()
    #data_load_state.success("✅ 모든 데이터 로드 완료!")
except Exception as e:
    #data_load_state.error(f"데이터 로드 실패: {e}")
    st.stop()
    
# --- 공통 메뉴 옵션 헬퍼 ---
def get_menu_options(df_sales: pd.DataFrame, df_inventory: pd.DataFrame) -> list[str]:
    """판매 기록과 재고(완제품) 기준으로 노출할 메뉴 옵션을 동적으로 만든다."""
    options = set()
    try:
        if "상품상세" in df_sales.columns:
            options |= {to_korean_detail(str(x)) for x in df_sales["상품상세"].dropna().unique()}
    except Exception:
        pass
    try:
        if not df_inventory.empty:
            menu_rows = df_inventory[df_inventory["is_ingredient"] == False]
            options |= {to_korean_detail(str(x)) for x in menu_rows["상품상세"].dropna().unique()}
    except Exception:
        pass
    if not options:
        options |= set(MENU_MASTER_KR) | set(SEED_MENUS)
    return sorted(options)

def render_menu_management(prefix: str, title: str = "🍽️ 메뉴 구성 관리"):
    if title:
        st.subheader(title)
    st.caption("메뉴를 추가하면서 레시피를 함께 등록할 수 있습니다. (초기 재고는 0으로 생성)")

    try:
        df_ingredients = df_inv[df_inv["is_ingredient"] == True].copy()
        ingredient_options_kr = sorted(df_ingredients["상품상세"].dropna().unique().tolist())
        ing_kr_to_en_map = dict(zip(df_ingredients["상품상세"], df_ingredients["상품상세_en"]))
    except Exception:
        ingredient_options_kr = []
        ing_kr_to_en_map = {}

    menu_df_edit = df_inv[df_inv["is_ingredient"] == False].copy()
    menu_cols = ["상품상세", "uom", "현재재고"]
    menu_cols = [c for c in menu_cols if c in menu_df_edit.columns]

    if menu_df_edit.empty:
        st.info("등록된 메뉴가 없습니다. 아래에서 신규 메뉴를 추가해주세요.")
    else:
        st.dataframe(
            menu_df_edit[menu_cols].sort_values("상품상세"),
            hide_index=True,
            use_container_width=True,
        )

    with st.form(f"{prefix}_menu_add_form"):
        menu_label = st.text_input(
            "메뉴명 (표시용)",
            placeholder="예: 헤이즐넛 아메리카노 I",
            key=f"{prefix}_menu_label",
        )
        menu_uom = st.selectbox(
            "단위",
            ["ea", "g", "ml"],
            index=0,
            help="잔 단위는 ea, 재료형 메뉴는 g/ml",
            key=f"{prefix}_menu_uom",
        )
        with st.expander("레시피 입력 (선택)", expanded=False):
            if ingredient_options_kr:
                preview = ", ".join(ingredient_options_kr[:6])
                suffix = " ..." if len(ingredient_options_kr) > 6 else ""
                st.caption(f"기존 재료 예시: {preview}{suffix}")
            recipe_rows = [{"재료": None, "수량": 0.0, "단위": "g", "손실률(%)": 0.0}]
            edited_recipe_df = st.data_editor(
                pd.DataFrame(recipe_rows),
                column_config={
                    "재료": st.column_config.TextColumn("재료 (직접 입력)", required=False),
                    "수량": st.column_config.NumberColumn("수량", min_value=0.0, format="%.2f", required=False),
                    "단위": st.column_config.SelectboxColumn("단위", options=["g", "ml", "ea"], required=False),
                    "손실률(%)": st.column_config.NumberColumn("손실률(%)", min_value=0.0, max_value=100.0, format="%.1f %%", required=False),
                },
                num_rows="dynamic",
                use_container_width=True,
                key=f"{prefix}_menu_recipe_editor",
            )
        submit_menu = st.form_submit_button("메뉴 추가/업데이트")

    if submit_menu:
        clean_name = menu_label.strip()
        if not clean_name:
            st.warning("메뉴명을 입력해주세요.")
        else:
            menu_en = from_korean_detail(clean_name)
            doc_id = safe_doc_id(menu_en)
            ref = db.collection(INVENTORY_COLLECTION).document(doc_id)
            snap = ref.get()

            base_fields = {
                "상품상세_en": menu_en,
                "상품상세": clean_name,
                "is_ingredient": False,
                "uom": normalize_uom(menu_uom),
                "supply_mode": DEFAULT_SUPPLY_MODE,
                "supply_lead_days": DEFAULT_SUPPLY_LEAD_DAYS,
            }
            if snap.exists:
                ref.update(base_fields)
                st.success(f"✅ '{clean_name}' 메뉴 정보를 업데이트했습니다. (재고는 유지)")
            else:
                ref.set({
                    **base_fields,
                    "초기재고": 0.0,
                    "현재재고": 0.0,
                    "cost_unit_size": 1.0,
                    "cost_per_unit": 0.0,
                    "unit_cost": 0.0,
                })
                st.success(f"✅ '{clean_name}' 메뉴를 추가했습니다.")

            final_ingredients = []
            recipe_errors = []
            if edited_recipe_df is not None and not edited_recipe_df.empty:
                for _, row in edited_recipe_df.iterrows():
                    ing_kr = str(row.get("재료") or "").strip()
                    qty = safe_float(row.get("수량", 0.0), 0.0)
                    if not ing_kr or qty <= 0:
                        continue
                    uom_val = normalize_uom(row.get("단위") or "g")
                    ing_en = ing_kr_to_en_map.get(ing_kr)
                    if not ing_en:
                        ing_en = ensure_inventory_ingredient(ing_kr, uom=uom_val)
                        if ing_en:
                            ing_kr_to_en_map[ing_kr] = ing_en
                    if not ing_en:
                        recipe_errors.append(ing_kr)
                        continue
                    final_ingredients.append({
                        "ingredient_en": ing_en,
                        "qty": qty,
                        "uom": uom_val,
                        "waste_pct": safe_float(row.get("손실률(%)", 0.0), 0.0),
                    })

            if final_ingredients:
                try:
                    db.collection(RECIPES_COLLECTION).document(safe_doc_id(menu_en)).set({
                        "ingredients": final_ingredients,
                        "menu_sku_en": menu_en,
                        "menu_name_ko": clean_name,
                    }, merge=True)
                    st.success("✅ 레시피가 함께 저장되었습니다.")
                except Exception as e:
                    st.warning(f"레시피 저장 실패: {e}")
            elif edited_recipe_df is not None and edited_recipe_df.size > 0:
                st.info("레시피 입력이 비어 있어 레시피 저장을 건너뜁니다.")
            if recipe_errors:
                st.warning(f"레시피 매칭 실패 재료: {', '.join(sorted(set(recipe_errors)))}")

            clear_cache_safe(load_all_core_data, load_inventory_df, load_recipe)
            safe_rerun()

    if not menu_df_edit.empty:
        st.markdown("---")
        del_targets = st.multiselect(
            "삭제할 메뉴 선택",
            menu_df_edit["상품상세"].tolist(),
            key=f"{prefix}_menu_delete_select",
        )
        if st.button(
            "선택 메뉴 삭제",
            use_container_width=True,
            disabled=not del_targets,
            key=f"{prefix}_menu_delete_btn",
        ):
            removed = 0
            for name in del_targets:
                try:
                    doc_id = safe_doc_id(from_korean_detail(name))
                    db.collection(INVENTORY_COLLECTION).document(doc_id).delete()
                    removed += 1
                except Exception as e:
                    st.warning(f"'{name}' 삭제 실패: {e}")
            if removed:
                st.success(f"🗑️ {removed}개 메뉴를 삭제했습니다. (기존 판매 데이터는 유지)")
                clear_cache_safe(load_all_core_data, load_inventory_df)
                safe_rerun()
            else:
                st.info("삭제된 메뉴가 없습니다.")

# --- 6. 원가(COGS) 계산 함수 (정의 4) ---
@st.cache_data(ttl=600)
def calculate_menu_cogs(df_inv: pd.DataFrame, recipes: dict, cost_override: Union[dict, None] = None) -> dict:
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
            if unit_cost == 0 and cost_override:
                fallback = cost_override.get(ing_sku_en)
                if fallback:
                    # 사용 단가 / 사용량으로 추정한 단위 원가
                    unit_cost = safe_float(fallback.get("unit_cost", 0.0))
            
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
    menu_cogs_map = calculate_menu_cogs(df_inv, RECIPES, cost_override=TOP5_COST_MAP)
    
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

def update_inventory_qty(ingredient_en: str,
                         qty: float,
                         uom: str = "ea",
                         is_ingredient: bool = True,
                         mode: str = "add",
                         cost_unit_size: float | None = None,
                         cost_per_unit: float | None = None,
                         move_type: str = "restock",
                         note: str = "") -> tuple[float, float, str] | None:
    if qty == 0:
        return None
    ref = db.collection(INVENTORY_COLLECTION).document(safe_doc_id(ingredient_en))
    snap = ref.get()
    if snap.exists:
        data = snap.to_dict() or {}
        cur = safe_float(data.get("현재재고", 0.0), 0.0)
        inv_uom = normalize_uom(data.get("uom", uom))
    else:
        cur = 0.0
        inv_uom = normalize_uom(uom)
        ref.set({
            "상품상세_en": ingredient_en,
            "상품상세": to_korean_detail(ingredient_en),
            "초기재고": 0.0,
            "현재재고": 0.0,
            "uom": inv_uom,
            "is_ingredient": bool(is_ingredient),
            "cost_unit_size": 1.0,
            "cost_per_unit": 0.0,
            "unit_cost": 0.0,
            "supply_mode": DEFAULT_SUPPLY_MODE,
            "supply_lead_days": DEFAULT_SUPPLY_LEAD_DAYS,
        })

    qty_converted = convert_qty(qty, from_uom=uom, to_uom=inv_uom)
    if mode == "set":
        new_stock = max(qty_converted, 0.0)
    else:
        new_stock = max(cur + qty_converted, 0.0)

    patch = {
        "현재재고": new_stock,
        "uom": inv_uom,
        "is_ingredient": bool(is_ingredient),
    }
    if not snap.exists:
        patch["초기재고"] = new_stock
    if cost_unit_size is not None and cost_unit_size > 0:
        patch["cost_unit_size"] = cost_unit_size
    if cost_per_unit is not None and cost_per_unit >= 0:
        patch["cost_per_unit"] = cost_per_unit
    if patch:
        ref.set(patch, merge=True)

    log_doc = {
        "ts": datetime.now().isoformat(),
        "type": move_type,
        "ingredient_en": ingredient_en,
        "qty": qty_converted,
        "uom": inv_uom,
        "mode": mode,
        "note": note,
        "before": cur,
        "after": new_stock,
    }
    db.collection(STOCK_MOVES_COLLECTION).add(log_doc)
    return cur, new_stock, inv_uom

# === [AI/ML 통합 추가] ===
# SPRINT 1: Gemini API 호출 헬퍼
def call_gemini_api(user_prompt: str, data_context: str, model: str = GEMINI_TEXT_MODEL):
    """
    [AI 수정 2] data_context(사실)와 user_prompt(요청)를 분리하여 AI가 '거짓말'을 하지 않도록 수정.
    data_context는 시스템 지시로 전달합니다.
    """

    # 1. API 키가 없는 경우
    if not GEMINI_API_KEY:
        time.sleep(1.5)
        st.error("Gemini API 키가 'secrets.toml'에 설정되지 않았습니다.")
        return (
            "⚠️ **[AI 응답 실패 (API 키 없음)]**\n\n"
            "'secrets.toml'에 Gemini API 키가 설정되지 않았습니다.\n\n"
            f"--- (데이터 컨텍스트) ---\n{data_context}\n\n"
            f"--- (사용자 요청) ---\n{user_prompt}"
        )

    if not GEMINI_CLIENT:
        st.error("Gemini 클라이언트를 초기화하지 못했습니다.")
        return None

    system_instruction = (
        "당신은 카페 운영 및 마케팅 전문가입니다. "
        "다음은 현재 카페의 실제 데이터입니다. 이 데이터를 '사실'로 간주하고, "
        "이 '사실'에 기반해서만 답변해야 합니다. 절대 데이터를 지어내지 마세요.\n\n"
        f"--- [카페 실제 데이터] ---\n{data_context}\n--- [데이터 끝] ---"
    )
    prompt_text = f"{system_instruction}\n\n[사용자 요청]\n{user_prompt}"

    # 2. API 호출 (3.5 우선, 실패 시 하위 버전 폴백)
    candidates = GEMINI_TEXT_MODEL_CANDIDATES
    last_error = None
    for m in candidates:
        try:
            response = GEMINI_CLIENT.models.generate_content(
                model=m,
                contents=[{"role": "user", "parts": [{"text": prompt_text}]}],
                config={"response_mime_type": "text/plain"},
            )
            if getattr(response, "text", None):
                if m != candidates[0]:
                    st.info(f"⚠️ 기본 모델 실패로 {m} 으로 폴백했습니다.")
                return response.text
        except Exception as e:
            last_error = e
            continue
    st.error(f"Gemini API 호출 중 오류 발생: {last_error}")
    return None
    
# ==========================================
# [AI/ML 통합 추가] 영수증 이미지 분석 헬퍼 함수
# ==========================================
def analyze_receipt_image(uploaded_file):
    """
    업로드된 영수증 이미지를 Gemini에게 보내서 상호명, 날짜, 시간, 품목 리스트, 총액을 JSON으로 추출합니다.
    """
    if not GEMINI_API_KEY:
        st.error("Gemini API 키가 설정되지 않았습니다.")
        return None
    if not GEMINI_CLIENT:
        st.error("Gemini 클라이언트를 초기화하지 못했습니다.")
        return None

    # 1. 업로드된 이미지를 바이너리 형태로 준비
    bytes_data = uploaded_file.getvalue()
    mime_type = getattr(uploaded_file, "type", None) or "image/jpeg"
    encoded_image = base64.b64encode(bytes_data).decode("utf-8")

    # 2. 프롬프트 설정 (JSON 형식 강제)
    system_prompt = """
    You are a receipt/POS OCR assistant. The image can be a printed receipt or a POS screen capture.
    Analyze the image and extract the following information in JSON format:
    {
        "store_name": "Store Name",
        "date": "YYYY-MM-DD",
        "time": "HH:MM",
        "items": [
            {"name": "Item Name 1", "qty": 1, "price": 1000, "total": 1000},
            {"name": "Item Name 2", "qty": 2, "price": 2000, "total": 4000}
        ],
        "total_amount": 5000
    }
    If date/time is missing, use null. Prices should be numbers (remove currency symbols).
    If qty is missing but price and total exist, infer qty = total / price.
    """

    # 3. API 호출 (3.5 우선, 실패 시 하위 버전 폴백)
    candidates = GEMINI_VISION_MODEL_CANDIDATES
    last_error = None
    for m in candidates:
        try:
            response = GEMINI_CLIENT.models.generate_content(
                model=m,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": system_prompt},
                            {"inline_data": {"mime_type": mime_type, "data": encoded_image}},
                        ],
                    }
                ],
                config={"response_mime_type": "application/json"},
            )

            result_text = getattr(response, "text", "") or ""
            if result_text:
                if m != candidates[0]:
                    st.info(f"⚠️ 기본 비전 모델 실패로 {m} 으로 폴백했습니다.")
                return json.loads(result_text)  # 딕셔너리로 변환하여 반환
        except Exception as e:
            last_error = e
            continue

    st.error(f"이미지 분석 중 오류 발생: {last_error}")
    return None

# SPRINT 2: Prophet 수요 예측 헬퍼
@st.cache_data(ttl=3600) # 1시간 캐시
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
        menu_candidates = build_menu_candidates(menu_sku_en)
        menu_name_kr_base = to_korean_detail(re.sub(r"\s+(Lg|Rg|Sm)$", "", str(menu_sku_en).strip()))
        
        original_menu_name_kr = to_korean_detail(menu_sku_en)
        if original_menu_name_kr != menu_name_kr_base:
            st.info(f"AI 예측: '{original_menu_name_kr}' 메뉴의 예측을 위해, 판매 데이터에서 '{menu_name_kr_base}'(으)로 조회합니다.")
        # === [버그 수정 끝] ===

        df_item = df_all_sales[
            df_all_sales['상품상세'].isin(menu_candidates)
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
        
        # [수정] freq='D'를 추가하여 '일(Daily)' 단위 예측임을 명시
        future = m.make_future_dataframe(periods=days_to_forecast, freq='D')
        forecast = m.predict(future)
        
        # === [빈틈 수정] 'y' 컬럼이 빠지는 오류 수정 ===
        forecast_chart_data = forecast.merge(df_prophet, on='ds', how='left')
        
        # 음수 예측은 0으로
        forecast_chart_data['yhat'] = forecast_chart_data['yhat'].clip(lower=0) 
        
        # 예측된 기간(target_days)의 총 소진량 합계 반환
        predicted_sum = forecast_chart_data.iloc[-days_to_forecast:]['yhat'].sum()
        
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
    menu_candidates = build_menu_candidates(menu_sku_en)
    menu_name_kr_base = to_korean_detail(re.sub(r"\s+(Lg|Rg|Sm)$", "", str(menu_sku_en).strip()))
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
        sold_sum_historical = df_win[df_win['상품상세'].isin(menu_candidates)]['수량'].sum()
    
    # 2. [AI/ML] 미래 수요 예측
    predicted_menu_sales, forecast_chart_data = get_item_forecast(
        df_all_sales, menu_sku_en, days_to_forecast=target_days_forecast
    )

    # 3. 사용할 판매량(sold_sum) 및 기준일(days) 결정
    fallback_chart_data = None
    use_historical_fallback = False
    
    if predicted_menu_sales is None or predicted_menu_sales == 0:
        st.warning(f"🤖 AI 예측: '{to_korean_detail(menu_sku_en)}'의 판매 데이터가 부족합니다. (과거 {window_days_fallback}일 판매량: {sold_sum_historical}개)을 기준으로 계산합니다.")
        sold_sum = sold_sum_historical # 과거 데이터 사용
        days = window_days_fallback
        use_historical_fallback = True
        # 과거 판매량 차트도 함께 표시
        try:
            df_hist = df_all_sales.copy()
            if "날짜" in df_hist.columns and not pd.api.types.is_datetime64_any_dtype(df_hist["날짜"]):
                df_hist["날짜"] = pd.to_datetime(df_hist["날짜"], errors="coerce")
            max_day = df_hist["날짜"].max()
            min_day = max_day - pd.Timedelta(days=window_days_fallback - 1)
            df_hist = df_hist[
                (df_hist["날짜"] >= min_day) & (df_hist["날짜"] <= max_day) &
                (df_hist["상품상세"].isin(menu_candidates))
            ]
            df_hist = df_hist.groupby(df_hist["날짜"].dt.date)["수량"].sum().reset_index()
            if not df_hist.empty:
                fallback_chart_data = df_hist.rename(columns={"날짜": "ds", "수량": "y"})
        except Exception:
            fallback_chart_data = None
    else:
        st.success(f"🤖 **AI 예측**: '{to_korean_detail(menu_sku_en)}'의 향후 **{target_days_forecast}일간** 예상 판매량을 **{predicted_menu_sales:,.0f}개**로 예측했습니다.")
        sold_sum = predicted_menu_sales # 예측값으로 대체
        days = target_days_forecast # 기준일도 예측 기간으로 변경

    # === 그래프 렌더링 (예측 성공/실패 모두 여기서 처리, Prophet 스타일 유지) ===
    # 실제 판매 히스토리(전체 거래 내역) 집계
    actual_history = df_all_sales[df_all_sales["상품상세"].isin(menu_candidates)].copy()
    if not actual_history.empty:
        if not pd.api.types.is_datetime64_any_dtype(actual_history["날짜"]):
            actual_history["날짜"] = pd.to_datetime(actual_history["날짜"], errors="coerce")
        actual_history = actual_history.dropna(subset=["날짜"])
        actual_history = actual_history.groupby(actual_history["날짜"].dt.date)["수량"].sum().reset_index()
        # 날짜 연속 구간으로 리샘플
        if not actual_history.empty:
            date_range_hist = pd.date_range(start=actual_history["날짜"].min(), end=actual_history["날짜"].max(), freq="D")
            actual_history = actual_history.set_index("날짜").reindex(date_range_hist, fill_value=0).reset_index().rename(columns={"index": "날짜"})
        actual_history.rename(columns={"날짜": "ds", "수량": "y"}, inplace=True)

    try:
        if forecast_chart_data is not None:
            fig = px.line(
                forecast_chart_data,
                x="ds",
                y="yhat",
                title=f"'{to_korean_detail(menu_sku_en)}' 전체 기간 수요 예측",
                labels={"ds": "날짜", "yhat": "예측 판매량"},
            )
            # 전체 거래 히스토리 라인
            if not actual_history.empty:
                fig.add_scatter(
                    x=actual_history["ds"],
                    y=actual_history["y"],
                    mode="lines+markers",
                    name="실제 판매량(전체)",
                    line=dict(color="rgba(0,0,0,0.5)", width=2),
                    marker=dict(color="rgba(0,0,0,0.6)", size=4),
                )
            # 예측에 포함된 과거 구간의 실제 y 점 표시
            actual_data = forecast_chart_data.dropna(subset=["y"])
            if not actual_data.empty:
                fig.add_scatter(
                    x=actual_data["ds"],
                    y=actual_data["y"],
                    mode="markers",
                    name="실제 판매량(예측 구간)",
                    marker=dict(color="rgba(0,0,255,0.5)", size=5),
                )
            fig.add_scatter(
                x=forecast_chart_data["ds"],
                y=forecast_chart_data["yhat_lower"],
                fill="tozeroy",
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                name="불확실성(하한)",
            )
            fig.add_scatter(
                x=forecast_chart_data["ds"],
                y=forecast_chart_data["yhat_upper"],
                fill="tonexty",
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor="rgba(231, 234, 241, 0.5)",
                name="불확실성(상한)",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"ai-forecast-{menu_sku_en}-{uuid.uuid4()}")
        elif fallback_chart_data is not None and not getattr(fallback_chart_data, "empty", True):
            fig_line = px.line(
                fallback_chart_data,
                x="ds",
                y="y",
                markers=True,
                title=f"'{to_korean_detail(menu_sku_en)}' 최근 {window_days_fallback}일 판매량",
                labels={"ds": "날짜", "y": "판매량"},
            )
            st.plotly_chart(fig_line, use_container_width=True, key=f"ai-fallback-{menu_sku_en}-{uuid.uuid4()}")
        else:
            st.info(f"'{to_korean_detail(menu_sku_en)}' 그래프를 표시할 데이터가 없습니다.")
    except Exception as e:
        st.error(f"예측/판매 차트 생성 오류: {e}")

    # 4. 레시피 기반 원재료 소진량 계산 (기존 로직 활용)
    rows = []
    for item in items:
        # ... (이하 모든 계산 로직은 원본과 동일) ...
        sku_en = item["ingredient_en"]
        qty_per_unit = safe_float(item.get("qty", 0.0))
        uom = normalize_uom(item.get("uom", "ea"))
        waste_pct = safe_float(item.get("waste_pct", 0.0))
        
        total_used = (qty_per_unit * sold_sum) * (1 + (waste_pct / 100.0))
        
        rows.append({
            "sku_en": sku_en,
            "uom_recipe": uom,
            "total_consumption": total_used
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
    
    base['lead_time_days'] = base['lead_time_days'].fillna(3)
    base['safety_stock_units'] = base['safety_stock_units'].fillna(0)
    base['target_days'] = base['target_days'].fillna(21)

    base["ROP"] = (base["일평균소진"] * base["lead_time_days"] + base["safety_stock_units"]).round(0).astype(int)
    target_need = base["일평균소진"] * base["target_days"]
    base["권장발주"] = (target_need + base["safety_stock_units"] - base["현재재고"]).apply(lambda x: max(int(ceil(x)), 0))
    # 재고=0이고 소진이 있는 경우 최소 1개 주문하도록 보정
    base.loc[(base["권장발주"] == 0) & (base["일평균소진"] > 0), "권장발주"] = 1
    base["상태"] = base.apply(lambda r: "🚨 발주요망" if r["현재재고"] <= r["ROP"] else "✅ 정상", axis=1)

    base["상품상세"] = base["sku_en"].apply(to_korean_detail)
    cols = ["상품상세","sku_en","현재재고","초기재고","uom","최근소진합","일평균소진","커버일수",
            "lead_time_days","safety_stock_units","target_days","ROP","권장발주","상태"]
    for c in cols:
        if c not in base.columns:
            base[c] = None
            
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
        
        # 3. 자기 자신(1.0)을 제외하고, 0.5 이상만 취득
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[(corr_pairs < 0.99) & (corr_pairs >= 0.5)]
        corr_pairs = corr_pairs.sort_values(ascending=False)
        
        if corr_pairs.empty:
            return "유의미한 동시 판매 패턴을 찾지 못했습니다."
        
        top_3 = corr_pairs.head(3)
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
# 1. 모든 메뉴 옵션 정의
# 1. 모든 메뉴 옵션 정의
MENU_OPTIONS = [
    "홈", "경영 현황", "기간별 분석", "거래 추가", 
    "재고 관리", "AI 비서", "데이터 편집", "거래 내역", "연구 검증", "도움말"
]

# 2. 세션 상태 초기화 (앱 실행 시 '홈'으로 설정)
if "current_page" not in st.session_state:
    st.session_state.current_page = "홈"

# 3. 페이지 변경 헬퍼 함수 (버튼 클릭 시 사용)
def set_page(page_name):
    st.session_state.current_page = page_name

# 4. 사이드바 제거됨
# st.sidebar.radio(...) 관련 코드가 제거되었습니다.

# 5. 현재 페이지를 세션 상태에서 가져옴
menu = st.session_state.current_page


# ==============================================================
# 🏠 홈 (메인 화면)
# ==============================================================
if menu == "홈":
    st.markdown("<h1 class='home-title'>🏠 비즈니스 관리 시스템</h1>", unsafe_allow_html=True)
    st.write("원하시는 메뉴를 선택해주세요.")
    
    # CSS 스타일 (버튼 높이 및 텍스트) - (기존 코드와 동일)
    st.markdown("""
    <style>
    /* 'border=True' 컨테이너의 패딩을 조절 */
    div[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        padding-top: 10px;
    }
    /* 버튼 높이, 폰트 크기 조절 */
    div[data-testid="stButton"] > button {
        height: 70px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stButton p { /* 버튼 내의 텍스트(이모지 포함) */
        font-size: 1.1rem;
        font-weight: 600;
    }
    /* 부제목 텍스트 스타일 */
    .home-desc {
        text-align: center; 
        font-size: 0.9rem; 
        color: #555;
        margin-top: -10px; 
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 메뉴 아이템 정의 (우선순위 그룹 + 보조 기능)
    top_menus = {
        "경영 현황": ("📈", "전체 경영 현황 확인"),
        # "매출 대시보드": ("📊", "매출 데이터 분석"),
        "기간별 분석": ("📅", "기간별 데이터 분석"),
        "거래 추가": ("➕", "반복 거래를 빠르게 등록"),
        "재고 관리": ("📦", "재고/발주 핵심 정보"),
        "AI 비서": ("🤖", "AI 기반 브리핑"),
    }
    secondary_menus = {
        "매출 대시보드": ("📊", "세부 매출 분석"),
        "거래 내역": ("🧾", "거래 이력 조회"),
        "데이터 편집": ("✏️", "데이터 수정 및 관리"),
        "연구 검증": ("🔬", "데이터 검증 및 연구"),
        "도움말": ("❓", "사용 가이드 및 지원"),
    }

    # 상단 5개 메뉴 (한 줄)
    cols = st.columns(len(top_menus))
    for idx, (key, (icon, desc)) in enumerate(top_menus.items()):
        with cols[idx].container(border=True):
            st.button(
                label=f"{icon} {key}",
                on_click=set_page,
                args=(key,),
                use_container_width=True,
            )
            st.markdown(f"<div class='home-desc'>{desc}</div>", unsafe_allow_html=True)

    # 보조 메뉴는 접어서 노출
    with st.expander("🔧 추가 기능", expanded=False):
        cols2 = st.columns(len(secondary_menus))
        for idx, (key, (icon, desc)) in enumerate(secondary_menus.items()):
            with cols2[idx].container(border=True):
                st.button(
                    label=f"{icon} {key}",
                    on_click=set_page,
                    args=(key,),
                    use_container_width=True,
                )
                st.markdown(f"<div class='home-desc'>{desc}</div>", unsafe_allow_html=True)

    # 추가 기능 아래 별도 드롭다운으로 보기 설정 노출
    with st.expander("👀 보기 설정", expanded=False):
        st.radio("글자 크기", list(FONT_SCALE_MAP.keys()), horizontal=True, key="font_scale_label")

# ==============================================================
# 🧾 거래 추가 (버튼 가시성 향상을 위해 수정된 예시)
# ==============================================================

elif menu == "경영 현황":
    # 1. st.columns()의 반환값을 언패킹합니다. (헤더가 필요 없으므로, 첫 번째 컬럼을 무시하기 위해 _ 사용)
    _, col_button = st.columns([0.8, 0.2])
    
    # 2. 이제 col_button은 두 번째 컬럼 객체이므로 with 구문 사용이 가능합니다.
    with col_button:
        st.write("") 
        st.button("🏠 홈으로 돌아가기", on_click=set_page, args=("홈",), use_container_width=True)
    st.markdown("---")
elif menu == "거래 추가":
    _, col_button = st.columns([0.8, 0.2])
    with col_button:
        st.write("") 
        # 👇 key="btn_home_add" 추가
        st.button("🏠 홈으로 돌아가기", on_click=set_page, args=("홈",), use_container_width=True, key="btn_home_add")
    
    st.markdown("---")

elif menu == "매출 대시보드":
    _, col_button = st.columns([0.8, 0.2])
    with col_button:
        st.write("") 
        # 👇 key="btn_home_sales" 추가
        st.button("🏠 홈으로 돌아가기", on_click=set_page, args=("홈",), use_container_width=True, key="btn_home_sales")
        
    st.markdown("---")
elif menu == "기간별 분석":
    _, col_button = st.columns([0.8, 0.2])
    with col_button:
        st.write("") 
        # 👇 key="btn_home_period" 추가
        st.button("🏠 홈으로 돌아가기", on_click=set_page, args=("홈",), use_container_width=True, key="btn_home_period")
    st.markdown("---")

elif menu == "재고 관리":
    _, col_button = st.columns([0.8, 0.2])
    with col_button:
        st.write("") 
        # 👇 key="btn_home_inventory" 추가
        st.button("🏠 홈으로 돌아가기", on_click=set_page, args=("홈",), use_container_width=True, key="btn_home_inventory")
    st.markdown("---")

elif menu == "AI 비서":
    _, col_button = st.columns([0.8, 0.2])
    with col_button:
        st.write("") 
        # 👇 key="btn_home_ai" 추가
        st.button("🏠 홈으로 돌아가기", on_click=set_page, args=("홈",), use_container_width=True, key="btn_home_ai")
    st.markdown("---")
elif menu == "데이터 편집":
    _, col_button = st.columns([0.8, 0.2])
    with col_button:
        st.write("") 
        # 👇 key="btn_home_edit" 추가
        st.button("🏠 홈으로 돌아가기", on_click=set_page, args=("홈",), use_container_width=True, key="btn_home_edit")
    st.markdown("---")

elif menu == "거래 내역":
    _, col_button = st.columns([0.8, 0.2])
    with col_button:
        st.write("") 
        # 👇 key="btn_home_history" 추가
        st.button("🏠 홈으로 돌아가기", on_click=set_page, args=("홈",), use_container_width=True, key="btn_home_history")
    st.markdown("---")
    # try:
    #     df_raw, df_view = load_sales_with_id()

elif menu == "연구 검증":
    _, col_button = st.columns([0.8, 0.2])
    with col_button:
        st.write("") 
        # 👇 key="btn_home_research" 추가
        st.button("🏠 홈으로 돌아가기", on_click=set_page, args=("홈",), use_container_width=True, key="btn_home_research")
    st.markdown("---")

elif menu == "도움말":
    _, col_button = st.columns([0.8, 0.2])
    with col_button:
        st.write("") 
        # 👇 key="btn_home_help" 추가
        st.button("🏠 홈으로 돌아가기", on_click=set_page, args=("홈",), use_container_width=True, key="btn_home_help")
        
    st.markdown("---")

# ==============================================================
# 🧾 거래 추가
# (원본 코드 생략)
# ==============================================================
if menu == "거래 추가":
    st.header(" 거래 데이터 추가")
    
    today = datetime.now().date()
    if "prefill_order" not in st.session_state:
        st.session_state.prefill_order = None
        st.session_state.prefill_from_history = False
    st.session_state.setdefault("order_channel", "직접입력")

    with st.expander("🍽️ 메뉴 편집", expanded=False):
        render_menu_management("sales_add", title="🍽️ 메뉴 편집")

    st.subheader("📸 매출 영수증/포스 화면 기반 거래 입력")
    st.caption("매출 영수증 또는 POS 화면 캡처를 업로드하면 AI가 거래 내역을 입력해줍니다.")

    if "sales_receipt_result" not in st.session_state:
        st.session_state.sales_receipt_result = None
    if "sales_receipt_meta" not in st.session_state:
        st.session_state.sales_receipt_meta = None

    if st.session_state.sales_receipt_result is None:
        st.markdown("### 영수증 사진 업로드")
        with st.container(border=True):
            sales_uploaded = st.file_uploader(
                "드래그 앤 드롭 또는 클릭하여 파일 선택",
                type=["png", "jpg", "jpeg", "webp"],
                help="AI가 영수증 정보를 자동으로 추출해 드립니다.",
                key="sales_receipt_uploader",
            )
            if sales_uploaded is not None:
                st.image(sales_uploaded, caption="업로드된 영수증", width=300)
                if st.button("🤖 AI 분석 시작", type="primary", use_container_width=True, key="sales_receipt_analyze"):
                    with st.spinner("AI가 영수증을 읽고 있습니다... (약 5~10초 소요) 🧠"):
                        receipt_meta = save_receipt_image(sales_uploaded, "sales")
                        if not receipt_meta:
                            st.error("영수증 저장에 실패했습니다. 다시 시도해주세요.")
                        else:
                            st.session_state.sales_receipt_meta = receipt_meta
                            data = analyze_receipt_image(sales_uploaded)
                            if data:
                                st.session_state.sales_receipt_result = data
                                st.session_state.sales_receipt_image = sales_uploaded
                                update_receipt_metadata(receipt_meta.get("receipt_id"), {"analysis_result": data})
                                safe_rerun()
    else:
        st.markdown("### 📝 데이터 검토 및 수정")

        data = st.session_state.sales_receipt_result

        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(st.session_state.sales_receipt_image, caption="원본 이미지", use_container_width=True)
            receipt_meta = st.session_state.get("sales_receipt_meta") or {}
            signed_url = build_signed_url_from_meta(receipt_meta)
            console_url = build_storage_console_url(
                receipt_meta.get("storage_bucket"),
                receipt_meta.get("storage_path"),
            )
            if signed_url:
                st.link_button("저장된 영수증 파일 열기", signed_url, use_container_width=True)
            elif console_url:
                st.link_button("Storage에서 보기", console_url, use_container_width=True)
            if st.button("🔄 다른 영수증 올리기", key="sales_receipt_reset"):
                st.session_state.sales_receipt_result = None
                st.session_state.sales_receipt_image = None
                st.session_state.sales_receipt_meta = None
                safe_rerun()

        with col_info:
            st.markdown("#### 영수증 정보")
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                store_name = c1.text_input("상호명", value=data.get("store_name", ""), key="sales_receipt_store")
                date_val = c2.text_input("거래 날짜", value=data.get("date", ""), key="sales_receipt_date")
                time_val = c3.text_input("거래 시간", value=data.get("time", ""), key="sales_receipt_time")

        st.markdown("#### 📦 물품 목록")

        items_df = pd.DataFrame(data.get("items", []))
        if items_df.empty:
            items_df = pd.DataFrame(columns=["name", "qty", "price", "total"])

        edited_items = st.data_editor(
            items_df,
            column_config={
                "name": st.column_config.TextColumn("물품명"),
                "qty": st.column_config.NumberColumn("수량", min_value=1),
                "price": st.column_config.NumberColumn("단가", format="%d원"),
                "total": st.column_config.NumberColumn("총액", format="%d원"),
            },
            num_rows="dynamic",
            use_container_width=True,
            key="sales_receipt_editor",
        )

        st.markdown("---")
        try:
            calc_total = edited_items["total"].sum()
        except Exception:
            calc_total = 0
        ai_total = safe_float(data.get("total_amount", 0), 0)

        col_sum1, col_sum2 = st.columns([3, 1])
        with col_sum2:
            st.metric("계산된 총액", f"{calc_total:,.0f}원", delta=f"AI 인식 금액: {ai_total:,.0f}원")

        st.markdown("---")
        if st.button("💾 매출 데이터로 저장", type="primary", use_container_width=True, key="sales_receipt_save"):
            receipt_meta = st.session_state.sales_receipt_meta or {}
            receipt_id = receipt_meta.get("receipt_id")
            if not receipt_id and st.session_state.get("sales_receipt_image") is not None:
                receipt_meta = save_receipt_image(st.session_state.sales_receipt_image, "sales")
                receipt_id = receipt_meta.get("receipt_id") if receipt_meta else None

            sale_date = normalize_receipt_date(date_val, today)
            sale_time = normalize_receipt_time(time_val, datetime.now().strftime("%H:%M:%S"))

            menu_options = get_menu_options(df, df_inv)
            menu_lookup = build_menu_lookup(menu_options)
            menu_fuzzy = build_menu_fuzzy_index(menu_options)
            unmatched = []
            saved_ids = []
            final_items = []

            with st.spinner("매출 데이터를 저장 중..."):
                for _, row in edited_items.iterrows():
                    name = str(row.get("name", "")).strip()
                    if not name:
                        continue
                    price = safe_float(row.get("price", 0.0), 0.0)
                    total = safe_float(row.get("total", 0.0), 0.0)
                    qty = safe_float(row.get("qty", 0.0), 0.0)
                    if qty <= 0:
                        if price > 0 and total > 0:
                            qty = total / price
                        else:
                            qty = 1
                    qty = int(round(qty))
                    if qty <= 0:
                        continue
                    if price <= 0 and total > 0 and qty > 0:
                        price = total / qty
                    revenue = price * qty if price > 0 else total

                    menu_ko, matched = match_menu_name(name, menu_lookup, menu_fuzzy)
                    if not matched:
                        unmatched.append(name)

                    try:
                        recent_row = df[df['상품상세'] == menu_ko].sort_values('날짜').iloc[-1]
                        recent_cat = recent_row.get('상품카테고리', '기타')
                        recent_type = recent_row.get('상품타입', '기타')
                    except Exception:
                        recent_cat = "기타"
                        recent_type = "기타"

                    menu_en = apply_name_map(from_korean_detail(menu_ko))
                    save_doc = {
                        "날짜": str(sale_date),
                        "상품상세": menu_en,
                        "상품상세_ko": menu_ko,
                        "상품카테고리": rev_category_map.get(recent_cat, recent_cat),
                        "상품타입": rev_type_map.get(recent_type, recent_type),
                        "수량": qty,
                        "단가": price,
                        "수익": revenue,
                        "가게위치": "Firebase",
                        "가게ID": "LOCAL",
                        "채널": "영수증",
                        "시간": sale_time,
                    }
                    if receipt_id:
                        save_doc["receipt_id"] = receipt_id
                    if receipt_meta:
                        if receipt_meta.get("storage_path"):
                            save_doc["receipt_storage_path"] = receipt_meta.get("storage_path")
                        if receipt_meta.get("storage_uri"):
                            save_doc["receipt_storage_uri"] = receipt_meta.get("storage_uri")
                    if store_name:
                        save_doc["receipt_store_name"] = store_name
                    if data.get("total_amount") is not None:
                        save_doc["receipt_total_amount"] = data.get("total_amount")

                    doc_ref, _ = db.collection(SALES_COLLECTION).add(save_doc)
                    saved_ids.append(doc_ref.id)
                    final_items.append({
                        "name": name,
                        "menu_name": menu_ko,
                        "qty": qty,
                        "price": price,
                        "total": revenue,
                    })

                    adjust_inventory_by_recipe(
                        menu_en,
                        qty,
                        move_type="sale",
                        note=f"영수증 입력: {menu_ko} x{qty}",
                    )

            if saved_ids:
                update_receipt_metadata(receipt_id, {
                    "sales_doc_ids": saved_ids,
                    "sales_count": len(saved_ids),
                    "sales_saved_at": datetime.now().isoformat(),
                    "receipt_store_name": store_name,
                    "receipt_date": str(sale_date),
                    "receipt_time": sale_time,
                    "final_items": final_items,
                    "final_total": calc_total,
                    "unmatched_items": list(set(unmatched)),
                })
                st.success(f"✅ 매출 데이터 {len(saved_ids)}건 저장 완료! (재고 자동 차감)")
                if unmatched:
                    st.warning(f"메뉴 매칭 실패 항목: {', '.join(sorted(set(unmatched)))}")
                clear_cache_safe(load_all_core_data, load_inventory_df)
                st.session_state.sales_receipt_result = None
                st.session_state.sales_receipt_image = None
                st.session_state.sales_receipt_meta = None
                safe_rerun()
            else:
                st.warning("저장할 거래 항목이 없습니다.")

        st.markdown("---")

    st.subheader("📄 엑셀 업로드로 매출 입력")
    st.caption("지원 컬럼 예시: 상품상세, 수량, 단가/총액, 날짜, 시간 (csv/xlsx 가능)")
    with st.container(border=True):
        sales_excel = st.file_uploader(
            "엑셀/CSV 파일 업로드",
            type=["xlsx", "xls", "csv"],
            key="sales_excel_uploader",
        )
        sales_excel_meta = save_upload_file_once(sales_excel, "sales_excel_upload", "sales_excel")
        if sales_excel_meta:
            signed_url = build_signed_url_from_meta(sales_excel_meta)
            console_url = build_storage_console_url(
                sales_excel_meta.get("storage_bucket"),
                sales_excel_meta.get("storage_path"),
            )
            if signed_url:
                st.link_button("업로드 파일 열기", signed_url, use_container_width=True)
            elif console_url:
                st.link_button("Storage에서 보기", console_url, use_container_width=True)
            if sales_excel_meta.get("saved_storage") and sales_excel_meta.get("storage_uri"):
                st.caption(f"Storage 저장 경로: {sales_excel_meta.get('storage_uri')}")
            elif sales_excel_meta.get("saved_local") and sales_excel_meta.get("local_path"):
                st.caption(f"로컬 저장 경로: {sales_excel_meta.get('local_path')}")
        df_sales_upload = read_tabular_upload(sales_excel)
        if df_sales_upload is not None:
            st.dataframe(df_sales_upload.head(20), use_container_width=True)
            col_name = pick_column(df_sales_upload, SALES_UPLOAD_ALIASES["name"])
            col_qty = pick_column(df_sales_upload, SALES_UPLOAD_ALIASES["qty"])
            col_price = pick_column(df_sales_upload, SALES_UPLOAD_ALIASES["price"])
            col_total = pick_column(df_sales_upload, SALES_UPLOAD_ALIASES["total"])
            col_date = pick_column(df_sales_upload, SALES_UPLOAD_ALIASES["date"])
            col_time = pick_column(df_sales_upload, SALES_UPLOAD_ALIASES["time"])
            col_cat = pick_column(df_sales_upload, SALES_UPLOAD_ALIASES["category"])
            col_type = pick_column(df_sales_upload, SALES_UPLOAD_ALIASES["type"])
            col_channel = pick_column(df_sales_upload, SALES_UPLOAD_ALIASES["channel"])

            missing_cols = []
            if not col_name:
                missing_cols.append("상품상세/상품명")
            if not col_qty:
                missing_cols.append("수량")
            if not col_price and not col_total:
                missing_cols.append("단가 또는 총액")
            if missing_cols:
                st.warning(f"필수 컬럼이 부족합니다: {', '.join(missing_cols)}")
            else:
                if st.button("💾 엑셀 매출 저장", type="primary", use_container_width=True, key="sales_excel_save"):
                    default_time = datetime.now().strftime("%H:%M:%S")
                    menu_options = get_menu_options(df, df_inv)
                    menu_lookup = build_menu_lookup(menu_options)
                    menu_fuzzy = build_menu_fuzzy_index(menu_options)
                    unmatched = []
                    saved_ids = []

                    upload_id = sales_excel_meta.get("upload_id") if sales_excel_meta else None
                    with st.spinner("엑셀 매출 데이터를 저장 중..."):
                        for _, row in df_sales_upload.iterrows():
                            name = str(row.get(col_name, "")).strip()
                            if not name:
                                continue
                            price = safe_float(row.get(col_price, 0.0), 0.0) if col_price else 0.0
                            total = safe_float(row.get(col_total, 0.0), 0.0) if col_total else 0.0
                            qty = safe_float(row.get(col_qty, 0.0), 0.0)
                            if qty <= 0:
                                if price > 0 and total > 0:
                                    qty = total / price
                                else:
                                    qty = 1
                            qty = int(round(qty))
                            if qty <= 0:
                                continue
                            if price <= 0 and total > 0 and qty > 0:
                                price = total / qty
                            revenue = price * qty if price > 0 else total

                            date_raw = row.get(col_date) if col_date else None
                            time_raw = row.get(col_time) if col_time else None
                            sale_date = normalize_receipt_date(date_raw, today)
                            sale_time = normalize_receipt_time(time_raw, default_time)

                            menu_ko, matched = match_menu_name(name, menu_lookup, menu_fuzzy)
                            if not matched:
                                unmatched.append(name)

                            if col_cat:
                                recent_cat = normalize_cell_str(row.get(col_cat), "기타")
                            else:
                                try:
                                    recent_row = df[df['상품상세'] == menu_ko].sort_values('날짜').iloc[-1]
                                    recent_cat = recent_row.get('상품카테고리', '기타')
                                except Exception:
                                    recent_cat = "기타"

                            if col_type:
                                recent_type = normalize_cell_str(row.get(col_type), "기타")
                            else:
                                try:
                                    recent_row = df[df['상품상세'] == menu_ko].sort_values('날짜').iloc[-1]
                                    recent_type = recent_row.get('상품타입', '기타')
                                except Exception:
                                    recent_type = "기타"

                            channel_val = normalize_cell_str(row.get(col_channel), "엑셀") if col_channel else "엑셀"
                            menu_en = apply_name_map(from_korean_detail(menu_ko))
                            save_doc = {
                                "날짜": str(sale_date),
                                "상품상세": menu_en,
                                "상품상세_ko": menu_ko,
                                "상품카테고리": rev_category_map.get(recent_cat, recent_cat),
                                "상품타입": rev_type_map.get(recent_type, recent_type),
                                "수량": qty,
                                "단가": price,
                                "수익": revenue,
                                "가게위치": "Firebase",
                                "가게ID": "LOCAL",
                                "채널": channel_val or "엑셀",
                                "시간": sale_time,
                            }
                            if upload_id:
                                save_doc["upload_id"] = upload_id
                                if sales_excel_meta.get("storage_path"):
                                    save_doc["upload_storage_path"] = sales_excel_meta.get("storage_path")
                                if sales_excel_meta.get("storage_uri"):
                                    save_doc["upload_storage_uri"] = sales_excel_meta.get("storage_uri")
                            doc_ref, _ = db.collection(SALES_COLLECTION).add(save_doc)
                            saved_ids.append(doc_ref.id)

                            adjust_inventory_by_recipe(
                                menu_en,
                                qty,
                                move_type="sale",
                                note=f"엑셀 입력: {menu_ko} x{qty}",
                            )

                    if saved_ids:
                        st.success(f"✅ 엑셀 매출 데이터 {len(saved_ids)}건 저장 완료! (재고 자동 차감)")
                        if unmatched:
                            st.warning(f"메뉴 매칭 실패 항목: {', '.join(sorted(set(unmatched)))}")
                        if upload_id:
                            update_upload_metadata(upload_id, {
                                "sales_doc_ids": saved_ids,
                                "sales_count": len(saved_ids),
                                "linked_at": datetime.now().isoformat(),
                            })
                        clear_cache_safe(load_all_core_data, load_inventory_df)
                        safe_rerun()
                    else:
                        st.warning("저장할 거래 항목이 없습니다.")

    st.markdown("---")

    st.markdown("#### ⚡ 간편 입력")
    # c_toss, c_dg = st.columns(2)
    # with c_toss:
    #     if st.button("채널1 간편 추가", key="btn_toss_quick", type="primary", use_container_width=True):
    #         st.session_state.order_channel = "채널1"
    #         st.toast("채널1로 입력 준비됐어요. 메뉴/가격만 고르면 됩니다.", icon="✨")
    # with c_dg:
    #     if st.button("채널2 추가", key="btn_dg_quick", use_container_width=True):
    #         st.session_state.order_channel = "채널2"
    #         st.toast("채널2로 입력 준비됐어요. 메뉴/가격만 고르면 됩니다.", icon="🥕")
    # st.caption(f"현재 채널: **{st.session_state.order_channel}**")

    # 메뉴/카테고리/타입 옵션은 증강 CSV 기준으로 제한
    df_order = df_csv.copy()
    if df_order.empty:
        st.info("주문 가능한 메뉴가 없어서 시드 메뉴 5종을 임시로 채웠습니다.")
        df_order = pd.DataFrame({
            "상품상세": SEED_MENUS,
            "상품상세_en": [from_korean_detail(m) for m in SEED_MENUS],
            "상품카테고리": ["기타"] * len(SEED_MENUS),
            "상품타입": ["기타"] * len(SEED_MENUS),
            "단가": [5000.0] * len(SEED_MENUS),
            "수량": [1] * len(SEED_MENUS),
            "수익": [5000.0] * len(SEED_MENUS),
            "날짜": [pd.Timestamp.now()] * len(SEED_MENUS),
        })

    # 베스트셀러 Top 7 노출 (전체 판매량 기준)
    best_cards = []
    try:
        best_df = (
            df.groupby("상품상세")["수량"]
            .sum()
            .reset_index()
            .sort_values("수량", ascending=False)
        )
        best_cards = best_df.head(7).to_dict("records")
    except Exception:
        best_cards = []

    if best_cards:
        st.subheader("🏆 많이 팔린 메뉴 (전체 판매량 Top 7)")
        cols = st.columns(len(best_cards))
        for idx, item in enumerate(best_cards):
            detail_ko = item["상품상세"]
            last_row = df[df["상품상세"] == detail_ko].sort_values("날짜").tail(1)
            cat_ko = last_row["상품카테고리"].iloc[0] if not last_row.empty else "기타"
            type_ko = last_row["상품타입"].iloc[0] if not last_row.empty else "기타"
            price_val = float(last_row["단가"].iloc[0]) if not last_row.empty else 1000.0
            qty_sold = int(item.get("수량", 0))
            with cols[idx].container(border=True):
                st.markdown(f"**{detail_ko}**")
                st.caption(f"누적 판매량: {qty_sold:,}개")
                st.caption(f"{cat_ko} · {type_ko} · 최근단가 {int(price_val):,}원")
                if st.button("불러오기", key=f"best_load_{idx}", use_container_width=True):
                    st.session_state.prefill_order = {
                        "상품상세": detail_ko,
                        "상품카테고리": cat_ko,
                        "상품타입": type_ko,
                        "단가": price_val,
                        "수량": 1,
                        "날짜": today,
                    }
                    st.session_state.prefill_from_history = False
                    safe_rerun()
        st.divider()

    prefill = st.session_state.prefill_order or {}
    # 선택값 자동 반영
    if prefill:
        st.session_state.setdefault("order_cat", prefill.get("상품카테고리"))
        st.session_state.setdefault("order_detail", prefill.get("상품상세"))
    # 메뉴 선택만 노출 (카테고리는 자동 추론)
    detail_options = get_menu_options(df, df_inv)
    if prefill and prefill.get("상품상세") and prefill.get("상품상세") not in detail_options:
        detail_options = [prefill.get("상품상세")] + detail_options
    상품상세_ko = choose_option("메뉴 선택", detail_options, key="order_detail", placeholder="메뉴를 선택하세요...")
    if prefill and 상품상세_ko != prefill.get("상품상세"):
        st.session_state.prefill_from_history = False

    if 상품상세_ko:
        # 최근 거래 기준으로 카테고리/타입/단가 추론
        try:
            recent_row = df[df['상품상세'] == 상품상세_ko].sort_values('날짜').iloc[-1]
            recent_cat = recent_row.get('상품카테고리', '기타')
            recent_type = recent_row.get('상품타입', '기타')
            last_price = float(recent_row.get('단가', 1000.0))
        except Exception:
            recent_cat = "기타"
            recent_type = "기타"
            last_price = 1000.0

        default_price = prefill.get("단가", last_price)
        default_qty = int(prefill.get("수량", 1))

        # 메뉴가 바뀔 때마다 가격 입력값을 최신값으로 반영
        if st.session_state.get("last_price_menu") != 상품상세_ko:
            st.session_state["order_price_input"] = f"{int(default_price):,}"
            st.session_state["last_price_menu"] = 상품상세_ko

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            수량 = st.number_input("수량", min_value=1, value=default_qty)
        with col2:
            단가 = render_currency_input("단가(원)", value=default_price, key="order_price_input")
        
        날짜 = st.date_input("날짜", value=today, format="YYYY-MM-DD")
        date_txt = format_date_with_holiday(날짜)
        if is_holiday_date(날짜):
            st.caption(f":red[{date_txt}]")
        else:
            st.caption(date_txt)

        수익 = 수량 * 단가
        st.markdown(f"### 💰 계산된 수익: **{format_krw(수익)}**")
        
        submitted = st.button("🟢 저장하기", use_container_width=True)
        
        if submitted:
            st.session_state.prefill_from_history = st.session_state.prefill_from_history and bool(prefill)
            상품카테고리_en = rev_category_map.get(recent_cat, recent_cat)
            상품타입_en = rev_type_map.get(recent_type, recent_type)
            상품상세_en = from_korean_detail(상품상세_ko)
            save_doc = {
                "날짜": str(today if st.session_state.prefill_from_history else 날짜),
                "상품상세": 상품상세_en,
                "상품상세_ko": 상품상세_ko,
                "상품카테고리": 상품카테고리_en,
                "상품타입": 상품타입_en,
                "수량": 수량,
                "단가": 단가,
                "수익": 수익,
                "가게위치": "Firebase",
                "가게ID": "LOCAL",
                "채널": st.session_state.get("order_channel", "직접입력"),
                "시간": datetime.now().strftime("%H:%M:%S"),
            }
            try:
                db.collection(SALES_COLLECTION).add(save_doc)
                st.success("✅ 저장되었습니다. (재고 자동 차감)")
                with st.spinner("재고 자동 차감 적용 중..."):
                    adjust_inventory_by_recipe(
                        상품상세_en,
                        수량,
                        move_type="sale",
                        note=f"거래 추가: {상품상세_ko} x{수량}"
                    )
                st.success("✅ 재고 차감 완료!")
                clear_cache_safe(load_all_core_data, load_inventory_df)
                st.session_state.prefill_from_history = False
                safe_rerun()
            except Exception as e:
                st.error(f"데이터 추가 실패: {e}")
                st.info("다시 시도 버튼을 눌러주세요.")
                if st.button("재시도", key="order_retry"):
                    safe_rerun()

# ==============================================================
# 📊 경영 현황
# (원본 코드 생략)
# ==============================================================
# ==============================================================
# 📊 통합 경영 현황 (경영 현황 + 매출 대시보드 통합)
# ==============================================================
# ==============================================================
# 📊 통합 경영 현황 (경영 현황 + 매출 대시보드 통합) - 수정됨
# ==============================================================
# ==============================================================
# 📊 통합 경영 현황 (경영 현황 + 매출 대시보드 통합) - 수정됨 (Fix KeyError)
# ==============================================================
# ==============================================================
# 📊 통합 경영 현황 (KeyError 해결: CSS 중괄호 Escape {{ }})
# ==============================================================
elif menu == "경영 현황":
    # 1. 상단 네비게이션
    col_header, col_btn = st.columns([0.85, 0.15])
    with col_header:
        st.header("📊 통합 경영 대시보드")
    # with col_btn:
    #     st.button("🏠 홈으로", on_click=set_page, args=("홈",), use_container_width=True, key="btn_dashboard_home_final")

    if df.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        # -------------------------------------------------------
        # [전처리] 데이터 라벨링 (강제 치환)
        # -------------------------------------------------------
        df_dashboard = df.copy()
        
        df_dashboard['상품카테고리'] = df_dashboard['상품카테고리'].astype(str).str.strip().replace('nan', '기타')
        df_dashboard['상품상세'] = df_dashboard['상품상세'].astype(str).str.strip().replace('nan', '미지정')

        rename_map = {
            "Coffee": "원두/에스프레소", 
            "커피": "원두/에스프레소",
            "Branded": "MD/기획상품",
            "branded": "MD/기획상품",
            "Tea": "차(Tea)",
            "Bakery": "베이커리",
            "Packaged Chocolate": "초콜릿/스낵",
            "Loose Tea": "잎차"
        }
        
        df_dashboard['상품카테고리'] = df_dashboard['상품카테고리'].replace(rename_map)
        df_dashboard['상품상세'] = df_dashboard['상품상세'].replace(rename_map)
        # -------------------------------------------------------

        # --- [SECTION 1] 핵심 KPI 카드 ---
        total_revenue = df_dashboard['수익'].sum()
        total_sales_count = df_dashboard.shape[0]
        avg_revenue_per_sale = total_revenue / total_sales_count if total_sales_count > 0 else 0
        
        # [중요] CSS 중괄호를 {{ }}로 변경하여 .format() 충돌 방지
        st.markdown("""
        <style>
        .metric-container {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 24px 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        .metric-title {{
            font-size: 1.3rem;
            font-weight: 600;
            color: #555;
            margin-bottom: 12px;
        }}
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 800;
            color: #004aad;
            line-height: 1.2;
        }}
        </style>

        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-title">💰 총 매출</div>
                <div class="metric-value">{total_revenue}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">🧾 총 판매 건수</div>
                <div class="metric-value">{total_count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">💳 건당 평균 매출</div>
                <div class="metric-value">{avg_revenue}</div>
            </div>
        </div>
        """.format(
            total_revenue=format_krw(total_revenue),
            total_count=f"{total_sales_count:,}",
            avg_revenue=format_krw(avg_revenue_per_sale)
        ), unsafe_allow_html=True)

        # --- [SECTION 2] 최고 인기 상품 정보 ---
        try:
            top_cat = df_dashboard.groupby('상품카테고리')['수익'].sum().sort_values(ascending=False).head(1)
            top_prod = df_dashboard.groupby('상품타입')['수익'].sum().sort_values(ascending=False).head(1)
            st.info(f"🏆 **매출 1위 카테고리**: {top_cat.index[0]} ({format_krw(top_cat.iloc[0])})  |  🏆 **매출 1위 타입**: {top_prod.index[0]}")
        except Exception:
            pass

        st.markdown("---")

        # --- [SECTION 3] 차트 영역 ---
        st.subheader("📈 매출 추이 분석")
        col_t1, col_t2 = st.columns(2)
        
        # 1. 일자별 매출 추이
        with col_t1:
            st.markdown("#### 📅 일자별 매출 흐름")
            
            daily = df_dashboard.groupby('날짜')['수익'].sum().reset_index()
            daily_filtered = daily[daily['수익'] > 0]
            
            if not daily_filtered.empty:
                st.markdown("""
                <div style="margin-bottom: 10px; padding: 10px; border-radius: 8px; background-color: rgba(255,255,255,0.05);">
                    <span style="font-size: 0.85rem; color: #E0E0E0;">
                        ℹ️ <b>Tip:</b> 🔴 <b>빨간 점</b>은 최고 매출일을 의미합니다.
                    </span>
                </div>
                """, unsafe_allow_html=True)

                max_row = daily_filtered.loc[daily_filtered['수익'].idxmax()]
                max_date = max_row['날짜']
                max_val = max_row['수익']
                avg_val = daily_filtered['수익'].mean()

                fig_trend = px.line(daily_filtered, x='날짜', y='수익', title=None)
                fig_trend.update_traces(
                    line_color='#1E88E5', 
                    fill='tozeroy',
                    fillcolor='rgba(30, 136, 229, 0.1)',
                    hovertemplate="<b>%{x|%Y년 %m월 %d일}</b><br>매출: %{y:,.0f}원<extra></extra>"
                )

                fig_trend.add_scatter(
                    x=[max_date], y=[max_val],
                    mode='markers+text',
                    marker=dict(color='red', size=10, symbol='star'),
                    text=[f"🏆최고: {format_krw(max_val)}"],
                    textposition="top center",
                    name='최고 매출'
                )

                fig_trend.add_shape(
                    type="line",
                    x0=daily_filtered['날짜'].min(), y0=avg_val,
                    x1=daily_filtered['날짜'].max(), y1=avg_val,
                    line=dict(color="gray", width=2, dash="dot"),
                )
                fig_trend.add_annotation(
                    x=daily_filtered['날짜'].max(), y=avg_val,
                    text=f"평균: {format_krw(avg_val)}",
                    showarrow=False,
                    yshift=10, xshift=-30,
                    font=dict(color="gray", size=11)
                )

                fig_trend.update_layout(
                    yaxis_tickformat=',.0f', 
                    yaxis_ticksuffix='원',   
                    xaxis_tickformat='%Y년 %m월 %d일',
                    hovermode="x unified",
                    showlegend=False,
                    margin=dict(t=20, l=10, r=10, b=10),
                    height=350
                )
                st.plotly_chart(fig_trend, use_container_width=True, key="dash-trend-daily")
            else:
                st.info("일자별 데이터가 없습니다.")

        # 2. 월별/카테고리별 누적 매출 (그라데이션 + 기타 회색)
       # 2. 월별/카테고리별 누적 매출 (진한색 아래 배치 + 기타 최상단 + 툴팁)
        with col_t2:
            st.markdown("#### 📊 월별 카테고리 누적 매출 (시간순)")
            
            df_clean = df_dashboard.dropna(subset=['날짜', '상품카테고리'])
            if not df_clean.empty:
                # [1] 순서 및 색상 로직 정의
                # '기타'를 제외한 나머지 카테고리를 매출 높은 순(내림차순)으로 정렬
                df_no_etc = df_clean[df_clean['상품카테고리'] != '기타']
                cat_revenue_rank = df_no_etc.groupby('상품카테고리')['수익'].sum().sort_values(ascending=False).index.tolist()
                
                # [중요] '기타'는 맨 마지막(그래프의 최상단)에 오도록 리스트 맨 뒤에 추가
                if '기타' in df_clean['상품카테고리'].unique():
                    cat_revenue_rank.append('기타')

                # [2] 색상 매핑 (진한 파랑 -> 연한 파랑, 기타=회색)
                blues = px.colors.sequential.Blues_r  # 진한색부터 시작
                
                # 색상 개수 맞추기 (기타 제외한 개수만큼)
                rank_len = len(cat_revenue_rank) - (1 if '기타' in cat_revenue_rank else 0)
                if rank_len > len(blues):
                    colors = blues * (rank_len // len(blues) + 1)
                else:
                    colors = blues

                # 딕셔너리로 매핑
                color_map_monthly = {cat: color for cat, color in zip(cat_revenue_rank, colors)}
                color_map_monthly['기타'] = '#E0E0E0' # 기타는 회색 고정

                # [3] 데이터 집계
                monthly_stacked_df = df_clean.groupby([
                    df_clean['날짜'].dt.to_period("M"), '상품카테고리'
                ])['수익'].sum().reset_index()
                monthly_stacked_df['날짜'] = monthly_stacked_df['날짜'].dt.to_timestamp()
                monthly_stacked_df['월(한글)'] = monthly_stacked_df['날짜'].dt.strftime('%Y년 %m월')

                # 날짜 정렬
                monthly_stacked_df = monthly_stacked_df.sort_values(['날짜', '상품카테고리'], ascending=[True, True])
                
                # 전월 대비 증감률 계산
                pct_change = monthly_stacked_df.groupby('상품카테고리')['수익'].pct_change().fillna(0) * 100
                monthly_stacked_df['전월대비'] = pct_change.round(0).astype(int)

                # [Tip 상단]
                st.markdown("""
                <div style="margin-bottom: 10px; padding: 10px; border-radius: 8px; background-color: rgba(255,255,255,0.05);">
                    <span style="font-size: 0.85rem; color: #E0E0E0;">
                        ℹ️ <b>Tip:</b> <b>아래쪽(진한 색)일수록 매출 비중이 큰 효자 상품</b>입니다.<br>
                        (마우스를 올리시면 전월 대비 증감률을 보실 수 있습니다.)
                    </span>
                </div>
                """, unsafe_allow_html=True)

                fig_stacked = px.bar(
                    monthly_stacked_df, x='월(한글)', y='수익', color='상품카테고리',
                    title=None, 
                    custom_data=['전월대비'],
                    # [핵심] category_orders를 통해 '매출 높은 순'이 '아래쪽'부터 쌓이게 설정
                    category_orders={'상품카테고리': cat_revenue_rank},
                    color_discrete_map=color_map_monthly
                )
                
                fig_stacked.update_layout(
                    yaxis_tickformat=',.0f', 
                    yaxis_ticksuffix='원',
                    xaxis_title="월", 
                    # xaxis 정렬 고정
                    xaxis={'categoryorder': 'array', 'categoryarray': sorted(monthly_stacked_df['월(한글)'].unique())},
                    margin=dict(t=20, l=10, r=10, b=10),
                    height=350
                )
                
                # 툴팁 설정
                fig_stacked.update_traces(hovertemplate="<b>%{data.name}</b><br>매출: %{y:,.0f}원<br>전월대비: %{customdata[0]:+d}%<extra></extra>")
                st.plotly_chart(fig_stacked, use_container_width=True, key="dash-monthly-stacked")
            else:
                st.info("월별 데이터를 집계할 수 없습니다.")

        st.markdown("---")

        # [ROW 2] 상품 및 카테고리 분석
        st.subheader("🛍 상품 및 카테고리 분석")
        col_p1, col_p2 = st.columns([0.65, 0.35]) 

        # 3. 상품 구조별 매출 트리맵
        with col_p1:
            df_tree = df_dashboard.copy()

            def assign_color_group(row):
                full_text = f"{row['상품카테고리']} {row['상품타입']} {row['상품상세']}".lower()
                blue_keywords = ['커피', '차', 'coffee', 'tea', 'beverage', 'drink', 'latte', 'espresso', 'americano', '아메리카노', '라떼', '원두']
                
                if any(k in full_text for k in blue_keywords):
                    return "커피/음료 (Blue)"
                else:
                    return "베이커리/MD (Orange)"

            df_tree['색상그룹'] = df_tree.apply(assign_color_group, axis=1)

            prod_sales = df_tree.groupby(['색상그룹', '상품카테고리', '상품상세'])['수익'].sum().reset_index()
            
            if not prod_sales.empty:
                simple_color_map = {
                    "커피/음료 (Blue)": "#90CAF9", 
                    "베이커리/MD (Orange)": "#FFAB91"
                }

                st.markdown("""
                <div style="margin-bottom: 15px; padding: 15px; border-radius: 12px; background-color: rgba(255,255,255,0.05);">
                    <div style="font-size: 1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
                        🔲 상품 구조별 매출 (트리맵)
                    </div>
                    <div style="display: flex; gap: 15px; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center;">
                            <span style="width: 10px; height: 10px; background-color: #90CAF9; border-radius: 3px; margin-right: 6px;"></span>
                            <span style="font-size: 0.85rem; color: #FFFFFF; font-weight: 500;">커피/음료</span>
                        </div>
                        <div style="display: flex; align-items: center;">
                            <span style="width: 10px; height: 10px; background-color: #FFAB91; border-radius: 3px; margin-right: 6px;"></span>
                            <span style="font-size: 0.85rem; color: #FFFFFF; font-weight: 500;">베이커리/MD</span>
                        </div>
                    </div>
                    <div style="font-size: 0.8rem; color: #CCCCCC;">
                        • <b>박스 크기</b> = <b>매출액</b> (클릭하여 확대)
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig_treemap = px.treemap(
                    prod_sales, 
                    path=['상품카테고리', '상품상세'], 
                    values='수익',
                    color='색상그룹', 
                    color_discrete_map=simple_color_map
                )
                
                fig_treemap.update_layout(
                    height=500,
                    margin=dict(t=0, l=0, r=0, b=0)
                )

                fig_treemap.update_traces(
                    hovertemplate="<b>%{label}</b><br>매출: %{value:,.0f}원<extra></extra>",
                    textinfo="label+value", 
                    textposition='middle center', 
                    textfont_size=14
                )
                st.plotly_chart(fig_treemap, use_container_width=True, key="dash-treemap-structure")
            else:
                st.info("트리맵 데이터 없음")

        # 4. 카테고리별 매출 바 차트
        with col_p2:
            st.markdown("#### 🏆 카테고리별 매출 순위") 
            
            cat_sales = df_dashboard.groupby('상품카테고리')['수익'].sum().reset_index().sort_values('수익', ascending=True)
            
            fig_cat = px.bar(cat_sales, x='수익', y='상품카테고리', orientation='h', title=None)
            
            fig_cat.update_layout(
                xaxis_tickformat=',.0f',
                xaxis_ticksuffix='원',
                yaxis_title=None,
                margin=dict(t=10, l=10, r=10, b=10),
                height=400
            )
            fig_cat.update_traces(hovertemplate="매출: %{x:,.0f}원<extra></extra>")
            st.plotly_chart(fig_cat, use_container_width=True, key="dash-cat-rank")

            # st.markdown("""
            # <div style="margin-top: 10px; padding: 10px; border-radius: 8px; background-color: rgba(255,255,255,0.05);">
            #     <span style="font-size: 0.85rem; color: #E0E0E0;">
            #         ℹ️ <b>Tip:</b> 우리 가게 <b>효자 카테고리</b> 순위입니다.<br>
            #         (막대가 길수록 매출 기여도가 높습니다)
            #     </span>
            # </div>
            # """, unsafe_allow_html=True)
# ==============================================================
# 📈 기간별 분석
# (원본 코드 생략)
# ==============================================================
elif menu == "기간별 분석":
    # -----------------------------------------------------------
    # 📈 기간별 분석 (React UI 포팅 버전)
    # -----------------------------------------------------------
    st.header("📈 기간별 분석")
    
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        # [0] 날짜 필터 상태 관리 (세션 스테이트 사용)
        # 앱이 리로드되어도 날짜 설정이 유지되도록 합니다.
        if 'anl_start_date' not in st.session_state:
            st.session_state.anl_start_date = df['날짜'].max().date() - pd.Timedelta(days=29) # 기본 1개월
        if 'anl_end_date' not in st.session_state:
            st.session_state.anl_end_date = df['날짜'].max().date()

        # [1] 상단 컨트롤 패널 (날짜 선택 + 퀵 버튼 + KPI 카드)
        # React의 레이아웃: 좌측(날짜 컨트롤) / 우측(매출 요약)
        
        # 전체를 감싸는 컨테이너 스타일
        st.markdown("""
        <style>
        .control-panel {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 24px;
        }
        .metric-box {
            padding: 16px 20px;
            border-radius: 12px;
            border: 1px solid;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            color: #1e293b;
        }
        </style>
        """, unsafe_allow_html=True)

        with st.container():
            col_ctrl, col_kpi = st.columns([1, 1.2])

            # --- 좌측: 날짜 선택 및 퀵 버튼 ---
            with col_ctrl:
                st.markdown("### 📅 조회 기간 설정")
                
                # 퀵 버튼 로직
                def set_period(days):
                    end = df['날짜'].max().date() # 기준은 데이터의 가장 최근 날짜
                    start = end - pd.Timedelta(days=days - 1) # inclusive 계산 (7일이면 오늘 포함 7일전)
                    st.session_state.anl_start_date = start
                    st.session_state.anl_end_date = end

                # 퀵 버튼 UI
                b_col1, b_col2, b_col3, _ = st.columns([1, 1, 1, 2])
                if b_col1.button("1주일"): set_period(7); safe_rerun()
                if b_col2.button("1개월"): set_period(30); safe_rerun()
                if b_col3.button("3개월"): set_period(90); safe_rerun()

                # 날짜 선택기 (세션 스테이트와 연동)
                c_d1, c_d2 = st.columns(2)
                new_start = c_d1.date_input("시작일", value=st.session_state.anl_start_date, max_value=df['날짜'].max().date())
                new_end = c_d2.date_input("종료일", value=st.session_state.anl_end_date, min_value=new_start, max_value=df['날짜'].max().date())
                
                # 수동 변경 감지 시 업데이트
                if new_start != st.session_state.anl_start_date or new_end != st.session_state.anl_end_date:
                    st.session_state.anl_start_date = new_start
                    st.session_state.anl_end_date = new_end
                    safe_rerun()

            # --- 데이터 필터링 ---
            # 선택된 날짜로 데이터 필터링
            mask = (df['날짜'].dt.date >= st.session_state.anl_start_date) & (df['날짜'].dt.date <= st.session_state.anl_end_date)
            filtered_df = df[mask]
            
            # --- 우측: KPI 카드 (총 매출 & 비교 분석) ---
            with col_kpi:
                if filtered_df.empty:
                    st.warning("선택한 기간에 데이터가 없습니다.")
                    total_revenue = 0
                    diff_revenue = 0
                    duration_days = 0
                    percent_change = 0
                else:
                    # 1. 현재 기간 매출
                    total_revenue = filtered_df['수익'].sum()

                    # 2. 직전 기간 매출 비교 로직
                    start_date = pd.to_datetime(st.session_state.anl_start_date)
                    end_date = pd.to_datetime(st.session_state.anl_end_date)

                    # 기간 일수 계산 (inclusive)
                    duration_days = (end_date - start_date).days + 1

                    # 직전 기간 계산
                    prev_end = start_date - pd.Timedelta(days=1)
                    prev_start = prev_end - pd.Timedelta(days=duration_days - 1)

                    prev_mask = (df['날짜'] >= prev_start) & (df['날짜'] <= prev_end)
                    prev_revenue = df[prev_mask]['수익'].sum()

                    # 👉 이전 기간 매출이 0이면 '비교 불가' 상태로 처리
                    if prev_revenue == 0:
                        diff_revenue = 0
                        compare_label = f"지난 {duration_days}일 대비"
                        diff_text = "이전 기간 데이터 없음"
                        is_comparable = False
                    else:
                        diff_revenue = total_revenue - prev_revenue
                        compare_label = f"지난 {duration_days}일 대비"
                        diff_text = f"{abs(diff_revenue):,.0f}원"
                        is_comparable = True

                    
                    # HTML/CSS로 KPI 카드 렌더링 (React 디자인 포팅)
                    # 색상 결정
                    if diff_revenue > 0:
                        bg_color = "linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%)" # Emerald-50
                        border_color = "#d1fae5" # Emerald-100
                        text_color = "#059669" # Emerald-600
                        icon = "▲"
                    elif diff_revenue < 0:
                        bg_color = "linear-gradient(135deg, #fff1f2 0%, #ffffff 100%)" # Rose-50
                        border_color = "#ffe4e6" # Rose-100
                        text_color = "#e11d48" # Rose-600
                        icon = "▼"
                    else:
                        bg_color = "#f8fafc"
                        border_color = "#e2e8f0"
                        text_color = "#64748b"
                        icon = "-"
                    
                    c_card1, c_card2 = st.columns(2)
                    with c_card1:
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%); border: 1px solid #dbeafe; border-radius: 12px; padding: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="color: #2563eb; font-weight: 700; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">총 매출</div>
                                <div style="display: flex; align-items: baseline; gap: 4px;">
                                    <span style="font-size: 2.2rem; font-weight: 800; color: #0f172a;">{total_revenue:,.0f}</span>
                                    <span style="font-size: 1.2rem; font-weight: 700; color: #64748b;">원</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with c_card2:
                        st.markdown(
                            f"""
                            <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 12px; padding: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <div style="color: {text_color}; font-weight: 700; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">지난 {duration_days}일 대비</div>
                                    <div style="background-color: {text_color}20; color: {text_color}; padding: 2px 6px; border-radius: 99px; font-size: 0.75rem; font-weight: bold;">{icon}</div>
                                </div>
                                <div style="display: flex; align-items: baseline; gap: 4px;">
                                    <span style="font-size: 1.8rem; font-weight: 800; color: #0f172a;">{abs(diff_revenue):,.0f}</span>
                                    <span style="font-size: 1.0rem; font-weight: 700; color: #64748b;">원</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


            
        st.markdown("---")

        if not filtered_df.empty:
            c_chart1, c_chart2 = st.columns(2)
            
            # -----------------------------------------------------------
            # [Chart 1] 요일별 매출 (WeeklyChart.tsx 포팅)
            # -----------------------------------------------------------
            with c_chart1:
                st.subheader("📊 요일별 매출")
                
                # 금요일 주말 포함 토글
                col_head, col_tog = st.columns([2, 1])
                with col_tog:
                    include_friday = st.toggle("금요일 주말 포함", value=True)
                
                # 데이터 집계
                week_sales = filtered_df.groupby('요일')['수익'].sum().reindex(weekday_order_kr).fillna(0)
                
                # 색상 결정 로직
                colors = []
                for day in week_sales.index:
                    if day in ['토', '일']:
                        colors.append('#f97316') # 주말 (Orange)
                    elif day == '금' and include_friday:
                        colors.append('#f97316') # 금요일 주말 포함 시
                    else:
                        colors.append('#3b82f6') # 평일 (Blue)

                # Y축 최소값 계산 (10만 단위 내림)
                min_rev = week_sales[week_sales > 0].min() if not week_sales[week_sales > 0].empty else 0
                max_rev = week_sales.max()
                y_min = (min_rev // 100000) * 100000
                y_max = max_rev * 1.1 # 여유 공간

                # Plotly GO 사용 (세밀한 제어)
                fig_week = go.Figure()
                fig_week.add_trace(go.Bar(
                    x=week_sales.index,
                    y=week_sales.values,
                    marker_color=colors,
                    hovertemplate='<b>%{x}요일</b><br>매출: %{y:,.0f}원<extra></extra>'
                ))
                
                fig_week.update_layout(
                    yaxis=dict(
                        range=[y_min, y_max],
                        tickformat=',.0f', # '만' 단위 처리는 텍스트 대체가 복잡하므로 콤마 포맷 사용
                        title=None
                    ),
                    xaxis=dict(title=None),
                    plot_bgcolor='rgba(0,0,0,0.02)',
                    margin=dict(t=10, b=0, l=0, r=0),
                    showlegend=False,
                    height=350
                )
                
                st.plotly_chart(fig_week, use_container_width=True, key="period-week")
                
                # 범례 (HTML)
                st.markdown("""
                <div style="display: flex; justify-content: center; gap: 16px; margin-top: -10px; font-size: 0.8rem; color: #64748b;">
                    <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 10px; height: 10px; background-color: #3b82f6; border-radius: 50%;"></span> 평일</div>
                    <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 10px; height: 10px; background-color: #f97316; border-radius: 50%;"></span> 주말</div>
                </div>
                """, unsafe_allow_html=True)

            # -----------------------------------------------------------
            # [Chart 2] 시간대별 매출 추이 (HourlyChart.tsx 포팅)
            # -----------------------------------------------------------
            with c_chart2:
                st.subheader("⏰ 시간대별 매출 추이 (이상 감지)")
                
                # 영업 시간 설정 필터
                h_c1, h_c2 = st.columns(2)
                with h_c1:
                    start_h = st.selectbox("영업 시작", range(0, 24), index=9)
                with h_c2:
                    end_h = st.selectbox("영업 종료", range(0, 24), index=22)
                
                if start_h > end_h:
                    st.error("시작 시간이 종료 시간보다 늦을 수 없습니다.")
                    end_h = start_h
                
                # 데이터 집계
                hourly_sales = filtered_df.groupby('시')['수익'].sum().reindex(range(24)).fillna(0).reset_index()
                
                # 필터링
                hourly_sales = hourly_sales[(hourly_sales['시'] >= start_h) & (hourly_sales['시'] <= end_h)]
                
                # 통계 계산 (평균, 주의, 위험)
                non_zero = hourly_sales[hourly_sales['수익'] > 0]['수익']

                if not non_zero.empty:
                    mean_val = non_zero.mean()
                else:
                    mean_val = 0

                warning_val = mean_val * 0.6
                critical_val = mean_val * 0.3
                
                # 색상 결정 로직 (점 색상)
                point_colors = []
                for val in hourly_sales['수익']:
                    if val < critical_val:
                        point_colors.append('#ef4444') # Red (저조)
                    elif val < warning_val:
                        point_colors.append('#eab308') # Yellow (주의)
                    else:
                        point_colors.append('#08519c') # Blue (정상)

                # 그라디언트 라인 시뮬레이션 (Marker + Line)
                fig_hour = go.Figure()
                
                # 1. 연결 선 (기본 회색/파란색 톤)
                fig_hour.add_trace(go.Scatter(
                    x=hourly_sales['시'],
                    y=hourly_sales['수익'],
                    mode='lines',
                    line=dict(color='#cbd5e1', width=2), # 기본 선은 연하게
                    hoverinfo='skip'
                ))
                
                # 2. 데이터 포인트 (상태별 색상)
                fig_hour.add_trace(go.Scatter(
                    x=hourly_sales['시'],
                    y=hourly_sales['수익'],
                    mode='markers',
                    marker=dict(
                        color=point_colors,
                        size=8,
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>%{x}시</b><br>매출: %{y:,.0f}원<extra></extra>'
                ))
                
                # 3. 평균선 (Reference Line)
                fig_hour.add_shape(
                    type="line",
                    x0=start_h, x1=end_h,
                    y0=mean_val, y1=mean_val,
                    line=dict(color="#94a3b8", width=1, dash="dash"),
                )
                fig_hour.add_annotation(
                    x=end_h, y=mean_val,
                    text="평균",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=10, color="#64748b")
                )

                fig_hour.update_layout(
                    yaxis=dict(
                        tickformat=',.0f',
                        title=None
                    ),
                    xaxis=dict(
                        title="시간 (시)",
                        tickmode='linear',
                        dtick=2 if (end_h - start_h) > 12 else 1
                    ),
                    plot_bgcolor='white',
                    margin=dict(t=10, b=0, l=0, r=0),
                    showlegend=False,
                    height=350
                )
                
                st.plotly_chart(fig_hour, use_container_width=True, key="period-hour")
                
                # 범례 (HTML)
                st.markdown("""
                <div style="display: flex; justify-content: center; gap: 12px; margin-top: -10px; font-size: 0.75rem; color: #64748b;">
                    <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 8px; height: 8px; background-color: #08519c; border-radius: 50%;"></span> 정상</div>
                    <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 8px; height: 8px; background-color: #eab308; border-radius: 50%;"></span> 주의 (<60%)</div>
                    <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 8px; height: 8px; background-color: #ef4444; border-radius: 50%;"></span> 저조 (<30%)</div>
                </div>
                """, unsafe_allow_html=True)
# ==============================================================
# 📦 재고 관리
# (원본 코드 생략, [AI/ML 통합 수정]이 적용된 함수를 사용)
# ==============================================================
# ==============================================================
# 📦 재고 관리
# ==============================================================
elif menu == "재고 관리":
    st.header("📦 재고 관리")
    # 데이터/예측 캐시를 강제로 갱신하고 싶을 때
    if st.button("🔄 데이터 새로 불러오기 (캐시 초기화)", key="btn_reload_inventory"):
        clear_cache_safe(load_all_core_data, load_inventory_df, load_sku_params, get_item_forecast)
        st.toast("캐시를 비우고 데이터를 다시 불러옵니다.")
        safe_rerun()
    
    # [수정] 모든 로직 전에 재고/파라미터를 먼저 로드
    df_inv = load_inventory_df()
    df_params = load_sku_params()

    # === 핵심 요약 라인 및 재고 의미화 ===
    usage_map = estimate_ingredient_daily_usage(df, RECIPES, days=30)
    df_usage = df_inv.copy()
    if not df_params.empty:
        df_params_lookup = df_params.set_index("sku_en").to_dict("index")
    else:
        df_params_lookup = {}

    def _lead_time_for_row(row):
        params = df_params_lookup.get(row["상품상세_en"], {})
        return safe_float(params.get("lead_time_days", row.get("supply_lead_days", DEFAULT_SUPPLY_LEAD_DAYS)))

    def _supply_mode_for_row(row):
        params = df_params_lookup.get(row["상품상세_en"], {})
        return params.get("supply_mode", row.get("supply_mode", DEFAULT_SUPPLY_MODE))

    df_usage["일평균소진(추정)"] = df_usage["상품상세_en"].map(usage_map).fillna(0.0)
    df_usage["lead_time_days"] = df_usage.apply(_lead_time_for_row, axis=1)
    df_usage["supply_mode"] = df_usage.apply(_supply_mode_for_row, axis=1)
    df_usage["판매 가능 일수"] = df_usage.apply(
        lambda r: round(r["현재재고"] / max(r["일평균소진(추정)"], 0.01), 1) if r["일평균소진(추정)"] > 0 else float("inf"),
        axis=1
    )
    df_usage["발주 추천일수"] = df_usage["판매 가능 일수"] - df_usage["lead_time_days"]
    df_usage["잔 환산"] = df_usage.apply(lambda r: convert_stock_to_cups(r["현재재고"], r["uom"], DEFAULT_GRAMS_PER_CUP), axis=1)
    df_usage["D-day"] = df_usage["판매 가능 일수"].apply(lambda x: "D-∞" if x == float("inf") else f"D-{max(int(round(x)),0)}")

    def _status(days_left, lead):
        if days_left == float("inf"):
            return "충분"
        if days_left <= max(lead, 0):
            return "위험"
        if days_left <= max(lead, 0) + 3:
            return "주의"
        return "충분"

    df_usage["상태"] = df_usage.apply(lambda r: _status(r["판매 가능 일수"], r["lead_time_days"]), axis=1)
    status_chip = {"충분": "🟢 충분", "주의": "🟡 주의", "위험": "🔴 위험"}
    df_usage["상태표시"] = df_usage["상태"].map(status_chip)

    # 재고 요약/현황 테이블 준비
    ing_usage_view = df_usage[df_usage["is_ingredient"] == True].copy()
    if not ing_usage_view.empty:
        ing_usage_view["현재 재고"] = ing_usage_view.apply(
            lambda r: f"{r['현재재고']:,.0f}{r['uom']} (약 {r['잔 환산']:,.0f}잔)" if r["잔 환산"] else f"{r['현재재고']:,.0f}{r['uom']}",
            axis=1
        )
        ing_usage_view["일평균 소진"] = ing_usage_view.apply(
            lambda r: (
                f"{r['일평균소진(추정)']:.2f}{r['uom']} "
                f"(약 {convert_stock_to_cups(r['일평균소진(추정)'], r['uom'], DEFAULT_GRAMS_PER_CUP):.1f}잔)"
            ) if r["uom"] == "g" else f"{r['일평균소진(추정)']:.2f}{r['uom']}",
            axis=1
        )
        ing_usage_view["발주 시점"] = ing_usage_view["lead_time_days"].apply(lambda x: f"{x:.0f}일 전")
        ing_usage_view["판매 가능 일수"] = ing_usage_view["판매 가능 일수"].replace(float("inf"), 9999)
        display_cols = ["상품상세", "상태표시", "현재 재고", "일평균 소진", "판매 가능 일수", "D-day", "발주 시점", "supply_mode"]
        summary_table = ing_usage_view[display_cols].rename(columns={
            "상품상세": "품목",
            "상태표시": "상태",
            "판매 가능 일수": "판매 가능 일수(일)",
            "발주 시점": "발주 시점",
            "supply_mode": "공급 방식"
        })
    else:
        summary_table = pd.DataFrame(columns=["품목","상태","현재 재고","일평균 소진","판매 가능 일수","D-day","발주 시점","공급 방식"])

    # key_items = df_usage[(df_usage["is_ingredient"] == True) & (df_usage["일평균소진(추정)"] > 0)]
    # if not key_items.empty:
    #     key_items_sorted = key_items.sort_values("판매 가능 일수")
    #     target_df = key_items_sorted[key_items_sorted["상품상세"].str.contains("원두")]
    #     if target_df.empty:
    #         target_df = key_items_sorted
    #     target = target_df.iloc[0]
    #     today_txt = format_date_with_holiday(datetime.now().date())
    #     order_val = safe_float(target.get("발주 추천일수"), float("inf"))
    #     if math.isfinite(order_val):
    #         order_txt = f"{max(int(order_val), 0)}일 후"
    #     else:
    #         order_txt = "계산 불가"
    #     lead_days_val = safe_float(target.get("lead_time_days"), float("nan"))
    #     lead_txt = f"{lead_days_val:.0f}일 리드타임" if math.isfinite(lead_days_val) else "리드타임 미설정"
    #     st.success(
    #         f"대표님, 오늘({today_txt}) 기준 **{target['상품상세']}** 소진 예상 {target['D-day']} "
    #         f"(약 {target['판매 가능 일수']:.1f}일 후) · 발주 추천: **{order_txt}** · 공급: {target['supply_mode']} ({lead_txt})"
    #     )
    # else:
    #     st.info("판매 데이터/레시피가 부족해 소진 예정일을 계산할 수 없습니다. 최근 거래와 레시피를 먼저 등록해주세요.")

    # # 🤖 메뉴별 AI 재고 영향도 분석 (Prophet 예측 기반)
    # st.divider()
    # st.subheader("🤖 메뉴별 재고 영향도 분석 (AI 예측)")
    # # 항상 증강 CSV 기준 메뉴를 모두 보여주고, 레시피/판매 데이터가 없으면 경고만 표시
    # menu_list_kr = MENU_MASTER_KR
    # menu_list_en = MENU_MASTER_EN
    # selected_menu_kr = st.selectbox("분석할 메뉴를 선택하세요", menu_list_kr, key="inv_ai_menu")
    # selected_menu_en = from_korean_detail(selected_menu_kr)

    # try:
    #     report_df = compute_ingredient_metrics_for_menu(
    #         selected_menu_en,
    #         df,
    #         df_inv,
    #         df_params
    #     )
    #     if report_df.empty:
    #         st.warning(f"'{selected_menu_kr}' 레시피 또는 판매 데이터가 부족합니다. 레시피를 저장하고 거래를 추가해 주세요.")
    #     else:
    #         display_cols = ['상품상세', '상태', '현재재고', 'uom', '권장발주', '커버일수', '일평균소진', 'ROP']
    #         formatted_df = report_df[display_cols].copy()
    #         formatted_df['상태'] = formatted_df['상태'].replace({
    #             '🚨 발주요망': '🔴 위험',
    #             '✅ 정상': '🟢 충분'
    #         })
    #         formatted_df['현재재고'] = formatted_df.apply(lambda r: f"{r['현재재고']:,.1f} {r['uom']}", axis=1)
    #         formatted_df['권장발주'] = formatted_df.apply(lambda r: f"{r['권장발주']:,.1f} {r['uom']}", axis=1)
    #         formatted_df['일평균소진'] = formatted_df.apply(lambda r: f"{r['일평균소진']:,.1f} {r['uom']}", axis=1)
    #         formatted_df['ROP'] = formatted_df.apply(lambda r: f"{r['ROP']:,.1f} {r['uom']}", axis=1)
    #         formatted_df['커버일수'] = formatted_df['커버일수'].apply(lambda x: f"{x}일")
    #         st.dataframe(
    #             formatted_df[['상품상세', '상태', '현재재고', '권장발주', '커버일수', '일평균소진', 'ROP']].rename(
    #                 columns={"커버일수": "판매 가능 일수", "ROP": "발주 시점"}
    #             ),
    #             use_container_width=True
    #         )
    # except Exception as e:
    #     st.error(f"AI 재고 영향도 분석 중 오류: {e}")

    # ing_usage_view = df_usage[df_usage["is_ingredient"] == True].copy()
    # if not ing_usage_view.empty:
    #     st.subheader("주요 재고 현황 (잔/개 단위로 직관적으로)")
    #     ing_usage_view["현재 재고"] = ing_usage_view.apply(
    #         lambda r: f"{r['현재재고']:,.0f}{r['uom']} (약 {r['잔 환산']:,.0f}잔)" if r["잔 환산"] else f"{r['현재재고']:,.0f}{r['uom']}",
    #         axis=1
    #     )
    #     ing_usage_view["일평균 소진"] = ing_usage_view.apply(
    #         lambda r: (
    #             f"{r['일평균소진(추정)']:.2f}{r['uom']} "
    #             f"(약 {convert_stock_to_cups(r['일평균소진(추정)'], r['uom'], DEFAULT_GRAMS_PER_CUP):.1f}잔)"
    #         ) if r["uom"] == "g" else f"{r['일평균소진(추정)']:.2f}{r['uom']}",
    #         axis=1
    #     )
    #     ing_usage_view["발주 시점"] = ing_usage_view["lead_time_days"].apply(lambda x: f"{x:.0f}일 전")
    #     ing_usage_view["판매 가능 일수"] = ing_usage_view["판매 가능 일수"].replace(float("inf"), 9999)
    #     display_cols = ["상품상세", "상태표시", "현재 재고", "일평균 소진", "판매 가능 일수", "D-day", "발주 시점", "supply_mode"]
    #     st.dataframe(
    #         ing_usage_view[display_cols].rename(columns={
    #             "상품상세": "품목",
    #             "상태표시": "상태",
    #             "판매 가능 일수": "판매 가능 일수(일)",
    #             "발주 시점": "발주 시점",
    #             "supply_mode": "공급 방식"
    #         }),
    #         use_container_width=True
    #     )

    # 상단 보고용 요약 2줄
    key_items = df_usage[(df_usage["is_ingredient"] == True) & (df_usage["일평균소진(추정)"] > 0)]
    if not key_items.empty:
        key_items_sorted = key_items.sort_values("판매 가능 일수")
        target_df = key_items_sorted[key_items_sorted["상품상세"].str.contains("원두")]
        if target_df.empty:
            target_df = key_items_sorted
        target = target_df.iloc[0]
        today_txt = format_date_with_holiday(datetime.now().date())
        order_val = safe_float(target.get("발주 추천일수"), float("inf"))
        order_txt = f"{max(int(order_val), 0)}일 후" if math.isfinite(order_val) else "계산 불가"
        lead_days_val = safe_float(target.get("lead_time_days"), float("nan"))
        lead_txt = f"{lead_days_val:.0f}일 리드타임" if math.isfinite(lead_days_val) else "리드타임 미설정"
        st.success(
            f"대표님, 오늘({today_txt}) 기준 **{target['상품상세']}** 소진 예상 {target['D-day']} "
            f"(약 {target['판매 가능 일수']:.1f}일 후))"
        )
        st.info("제안: 재고 요약/AI 영향도 탭에서 권장발주와 커버일수를 확인하고, 재고 입력 탭에서 즉시 반영하세요.")
    else:
        st.info("판매/레시피 데이터가 부족해 소진 예정일을 계산할 수 없습니다. 거래 추가 및 레시피 등록 후 재고 관리 탭을 다시 확인하세요.")
    
    tab_labels = [
        "📋 원가 및 레시피 관리",
        "💸 원재료 시세",
        "📸 재고 입력",
        "📦 재고 요약",
        "🤖 AI 영향도",
    ]
    st.session_state.setdefault("inv_active_tab", tab_labels[0])
    active_idx = tab_labels.index(st.session_state.get("inv_active_tab", tab_labels[0]))
    selected_tab = st.radio(
        "탭 선택",
        tab_labels,
        horizontal=True,
        index=active_idx,
        key="inv_tab_radio",
    )
    st.session_state.inv_active_tab = selected_tab


    # ==============================================================
    # TAB 0: (신규) 원가 및 레시피 관리
    # ==============================================================
    # ==============================================================
    # TAB 0: (신규) 원가 및 레시피 관리
    # ==============================================================
    if selected_tab == "📋 원가 및 레시피 관리":
        st.header("📋 원가 및 레시피 관리")

        # ------------------------------------------------------------------
        # [Helper] 재료 마스터 데이터 준비
        # ------------------------------------------------------------------
        ingredients_df = df_inv[df_inv['is_ingredient'] == True].copy()
        ing_options = sorted(ingredients_df['상품상세'].unique().tolist())
        ing_lookup = {}
        for _, r in ingredients_df.iterrows():
            ing_lookup[r['상품상세']] = {
                'uom': r['uom'],
                'unit_cost': r['unit_cost'],
                'en': r['상품상세_en']
            }
            
        def _get_ing_info(name_kr):
            return ing_lookup.get(name_kr, {'uom': 'ea', 'unit_cost': 0.0, 'en': from_korean_detail(name_kr)})

        # ------------------------------------------------------------------
        # 1. New Menu Creation (Interactive)
        # ------------------------------------------------------------------
        with st.expander("➕ 새 메뉴 등록", expanded=False):
            st.caption("메뉴 기본 정보와 레시피를 한번에 등록하세요.")
            
            # State Management for New Menu Recipe
            if "new_menu_recipe" not in st.session_state:
                st.session_state.new_menu_recipe = [{"재료": None, "사용량": 0.0, "단위": "-", "단위당 원가": 0.0, "재료비": 0.0}]

            c_new1, c_new2 = st.columns([3, 1])
            new_menu_name = c_new1.text_input("메뉴 이름", placeholder="예: 아이스 아메리카노", key="new_menu_name_input")
            new_menu_price = c_new2.number_input("판매가 (원)", min_value=0, step=100, key="new_menu_price_input")
            
            st.markdown("###### 📝 레시피 구성 (재료 선택 시 단위/원가 자동 입력)")
            
            # Data Editor for New Menu
            new_recipe_df = pd.DataFrame(st.session_state.new_menu_recipe)
            edited_new_recipe = st.data_editor(
                new_recipe_df,
                column_config={
                    "재료": st.column_config.SelectboxColumn("재료", options=ing_options, required=True),
                    "사용량": st.column_config.NumberColumn("사용량", min_value=0.0),
                    "단위": st.column_config.TextColumn("단위", disabled=True),
                    "단위당 원가": st.column_config.NumberColumn("단위당 원가", disabled=True, format="%d원"),
                    "재료비": st.column_config.NumberColumn("재료비", disabled=True, format="%d원"),
                },
                num_rows="dynamic",
                use_container_width=True,
                key="new_menu_editor"
            )
            
            # [Logic] Auto-update Unit/Cost/Total based on selection
            updated_rows = []
            total_cost_new = 0.0
            has_change = False
            
            # Reconstruct DataFrame with lookups
            for idx, row in edited_new_recipe.iterrows():
                r_name = row.get("재료")
                r_qty = safe_float(row.get("사용량", 0))
                
                if r_name:
                    info = _get_ing_info(r_name)
                    r_uom = info['uom']
                    r_cost = info['unit_cost']
                    r_total = r_qty * r_cost
                else:
                    r_uom = "-"
                    r_cost = 0.0
                    r_total = 0.0
                
                # Check for changes to trigger UI update (if needed)
                # But here we just rebuild the "next state" to save or display
                updated_rows.append({
                    "재료": r_name,
                    "사용량": r_qty,
                    "단위": r_uom,
                    "단위당 원가": r_cost,
                    "재료비": r_total
                })
                total_cost_new += r_total
            
            # [Added] Persist the calculated view for the *next* rerun/interaction
            # This ensures that after selecting "Milk", the "ml" and "Cost" columns fill in.
            st.session_state.new_menu_recipe = updated_rows
            
            # Display Totals
            st.markdown(f"**총 재료비: :red[{total_cost_new:,.0f}원]**")

            if st.button("메뉴 생성", type="primary"):
                if not new_menu_name:
                    st.error("메뉴 이름을 입력해주세요.")
                else:
                    sku_en = from_korean_detail(new_menu_name)
                    doc_id = safe_doc_id(sku_en)
                    
                    existing_docs = [d.id for d in db.collection(INVENTORY_COLLECTION).list_documents()]
                    if doc_id in existing_docs:
                        st.error("이미 존재하는 메뉴입니다.")
                    else:
                        # 1. Save Inventory
                        db.collection(INVENTORY_COLLECTION).document(doc_id).set({
                            "상품상세": new_menu_name,
                            "상품상세_en": sku_en,
                            "is_ingredient": False,
                            "sale_price": new_menu_price,
                            "uom": "ea",
                            "supply_mode": "Self",
                            "현재재고": 0,
                            "cost_per_unit": new_menu_price 
                        })
                        
                        # 2. Save Recipe
                        final_ingredients = []
                        for row in updated_rows:
                            if row['재료'] and row['사용량'] > 0:
                                info = _get_ing_info(row['재료'])
                                final_ingredients.append({
                                    "ingredient_en": info['en'],
                                    "qty": row['사용량'],
                                    "uom": info['uom']
                                })
                        
                        db.collection(RECIPES_COLLECTION).document(doc_id).set({
                            "menu_name_ko": new_menu_name,
                            "menu_sku_en": sku_en,
                            "sale_price": new_menu_price,
                            "ingredients": final_ingredients
                        })
                        
                        st.success(f"메뉴 '{new_menu_name}' 생성 완료!")
                        # Reset State
                        st.session_state.new_menu_recipe = [{"재료": None, "사용량": 0.0, "단위": "-", "단위당 원가": 0.0, "재료비": 0.0}]
                        clear_cache_safe(load_inventory_df, load_all_core_data)
                        safe_rerun()

        st.divider()

        # ------------------------------------------------------------------
        # 2. List Menus (cards)
        # ------------------------------------------------------------------
        menus_df = df_inv[df_inv['is_ingredient'] == False].copy()
        current_menu_cogs = calculate_menu_cogs(df_inv, RECIPES)

        for idx, row in menus_df.iterrows():
            menu_name = row['상품상세']
            menu_en = row['상품상세_en']
            menu_doc_id = safe_doc_id(menu_en)
            recipe_data = RECIPES.get(menu_en, [])
            
            sale_price = safe_float(row.get('sale_price', row.get('cost_per_unit', 0))) 
            total_cost = current_menu_cogs.get(menu_en, 0) # This is cached global cost
            
            # Recalculate cost dynamically inside the loop if we want real-time update in title?
            # Ideally, but complex. Let's rely on saved data for the header title.
            
            cost_ratio = (total_cost / sale_price * 100) if sale_price > 0 else 0
            
            if cost_ratio >= 30: ratio_color = "red"; ratio_icon = "🔴"
            elif cost_ratio >= 20: ratio_color = "orange"; ratio_icon = "🟡"
            else: ratio_color = "green"; ratio_icon = "🟢"

            label = f"**{menu_name}** | 판매가: {sale_price:,.0f}원 | 원가: {total_cost:,.0f}원 | 원가율: :{ratio_color}[{cost_ratio:.1f}%] {ratio_icon}"
            
            with st.container(border=True):
                c_head, c_btn = st.columns([0.9, 0.1])
                with c_head:
                    expanded = st.expander(label, expanded=False)
                with c_btn:
                    if st.button("🗑️", key=f"del_menu_{menu_doc_id}", help="삭제"):
                        db.collection(INVENTORY_COLLECTION).document(menu_doc_id).delete()
                        db.collection(RECIPES_COLLECTION).document(menu_doc_id).delete()
                        st.success("삭제됨")
                        clear_cache_safe(load_all_core_data)
                        safe_rerun()
                
                with expanded:
                    # -----------------------------------------
                    # Edit Logic (Interactive, No Form for Recipe)
                    # -----------------------------------------
                    c_edit1, c_edit2, c_save = st.columns([2, 2, 1])
                    new_name_edit = c_edit1.text_input("메뉴 이름", value=menu_name, key=f"name_{menu_doc_id}")
                    new_price_edit = c_edit2.number_input("판매가 (원)", value=float(sale_price), step=100.0, key=f"price_{menu_doc_id}")
                    
                    # Prepare initial data for editor
                    if f"editor_init_{menu_doc_id}" not in st.session_state:
                         rows = []
                         for ing in recipe_data:
                             i_en = ing.get('ingredient_en')
                             info = next((v for k, v in ing_lookup.items() if v['en'] == i_en), {})
                             # Fallback name
                             i_kr = next((k for k, v in ing_lookup.items() if v['en'] == i_en), to_korean_detail(i_en))
                             
                             qty = safe_float(ing.get('qty', 0))
                             unit_cost = info.get('unit_cost', 0.0)
                             rows.append({
                                 "재료": i_kr,
                                 "사용량": qty,
                                 "단위": info.get('uom', 'g'),
                                 "단위당 원가": unit_cost,
                                 "재료비": qty * unit_cost
                             })
                         if not rows:
                             rows = [{"재료": None, "사용량": 0.0, "단위": "-", "단위당 원가": 0.0, "재료비": 0.0}]
                         st.session_state[f"editor_init_{menu_doc_id}"] = pd.DataFrame(rows)

                    st.markdown("###### 📝 레시피 구성")
                    
                    # Interactive Editor
                    edited_receipe = st.data_editor(
                        st.session_state[f"editor_init_{menu_doc_id}"],
                        column_config={
                            "재료": st.column_config.SelectboxColumn("재료", options=ing_options, required=True),
                            "사용량": st.column_config.NumberColumn("사용량", min_value=0.0),
                            "단위": st.column_config.TextColumn("단위", disabled=True),
                            "단위당 원가": st.column_config.NumberColumn("단위당 원가", disabled=True, format="%d원"),
                            "재료비": st.column_config.NumberColumn("재료비", disabled=True, format="%d원"),
                        },
                        num_rows="dynamic",
                        use_container_width=True,
                        key=f"editor_{menu_doc_id}"
                    )
                    
                    # Calculate Footer (Totals)
                    current_total_cost = 0.0
                    
                    # Processing edits for display update & save preparation
                    # We can't easily "push back" to the editor in real-time loop without causing UX jumps,
                    # so we calculate totals based on what the user SELECTED.
                    
                    final_ingredients_to_save = []
                    
                    for _, r_row in edited_receipe.iterrows():
                        r_name = r_row.get("재료")
                        r_qty = safe_float(r_row.get("사용량", 0))
                        
                        if r_name:
                            info = _get_ing_info(r_name)
                            line_total = r_qty * info['unit_cost']
                            current_total_cost += line_total
                            
                            # Prepare for save
                            final_ingredients_to_save.append({
                                "ingredient_en": info['en'],
                                "qty": r_qty,
                                "uom": info['uom']
                            })
                            
                            # [Added] Prepare for state sync (for UI update)
                            row_for_state = {
                                "재료": r_name,
                                "사용량": r_qty,
                                "단위": info['uom'],
                                "단위당 원가": info['unit_cost'],
                                "재료비": line_total
                            }
                        else:
                            row_for_state = {
                                "재료": None, "사용량": 0.0,
                                "단위": "-", "단위당 원가": 0.0, "재료비": 0.0
                            }
                        
                        # We need to reconstruct the dataframe to update the disabled columns
                        # We can't do it in-place easily, so we rebuild the list
                        # Note: This logic effectively runs "one step behind" for the visual table update
                        # unless we force rerun.
                        
                    # [Added] Sync back to session state so disabled columns update on next interaction
                    # But we can't easily rebuild the list inside the loop above because we need to preserve order/rows?
                    # Streamlit data_editor returns all rows.
                    
                    # Re-loop to build the full state dataframe
                    state_rows = []
                    for _, r_row in edited_receipe.iterrows():
                         r_name_s = r_row.get("재료")
                         r_qty_s = safe_float(r_row.get("사용량", 0))
                         if r_name_s:
                             info_s = _get_ing_info(r_name_s)
                             state_rows.append({
                                 "재료": r_name_s,
                                 "사용량": r_qty_s,
                                 "단위": info_s['uom'],
                                 "단위당 원가": info_s['unit_cost'],
                                 "재료비": r_qty_s * info_s['unit_cost']
                             })
                         else:
                             state_rows.append(r_row.to_dict()) # Keep empty/custom
                    
                    st.session_state[f"editor_init_{menu_doc_id}"] = pd.DataFrame(state_rows)
                    
                    # [Added] Calculate Real-time Ratio
                    live_sale_price = float(new_price_edit)
                    if live_sale_price > 0:
                        live_ratio = (current_total_cost / live_sale_price) * 100
                    else:
                        live_ratio = 0.0
                    
                    if live_ratio >= 30: r_color = "red"; r_icon = "🔴"
                    elif live_ratio >= 20: r_color = "orange"; r_icon = "🟡"
                    else: r_color = "green"; r_icon = "🟢"

                    # Show LIVE Total & Ratio
                    st.markdown(f"**실시간 총 원가: :red[{current_total_cost:,.0f}원]** | **예상 원가율: :{r_color}[{live_ratio:.1f}%] {r_icon}**")

                    # SAVE ACTION
                    with c_save:
                        st.write("") # Spacer
                        st.write("") 
                        if st.button("💾 저장", key=f"btn_save_{menu_doc_id}", type="primary"):
                             # 1. Update Inventory
                             updates = {}
                             if new_name_edit != menu_name: updates['상품상세'] = new_name_edit
                             if new_price_edit != sale_price: updates['sale_price'] = new_price_edit
                             if updates:
                                 db.collection(INVENTORY_COLLECTION).document(menu_doc_id).update(updates)
                             
                             # 2. Update Recipes
                             db.collection(RECIPES_COLLECTION).document(menu_doc_id).set({
                                 "menu_name_ko": new_name_edit,
                                 "menu_sku_en": menu_en,
                                 "sale_price": new_price_edit,
                                 "ingredients": final_ingredients_to_save
                             }, merge=True)
                             
                             st.success("저장 완료!")
                             # Clear init state to force reload
                             del st.session_state[f"editor_init_{menu_doc_id}"]
                             clear_cache_safe(load_all_core_data, load_recipe) 
                             safe_rerun()

    # 재고 요약 탭
    if selected_tab == "📦 재고 요약":
        st.subheader("주요 재고 현황 (잔/개 단위로 직관적으로)")
        st.dataframe(summary_table, use_container_width=True)
        if not ing_usage_view.empty:
            with st.expander("왜 그렇지? (상세 설명 모아보기)"):
                for _, r in ing_usage_view.iterrows():
                    order_days_val = safe_float(r.get("발주 추천일수"), float("inf"))
                    order_txt = f"{max(int(order_days_val), 0)}일 후" if math.isfinite(order_days_val) else "계산 불가"
                    lead_days_val = safe_float(r.get("lead_time_days"), float("nan"))
                    lead_txt = f"{lead_days_val:.0f}일" if math.isfinite(lead_days_val) else "미설정"
                    st.markdown(
                        f"""
                        **{r['상품상세']}**
                        - 현재 재고: {r['현재 재고']}
                        - 일평균 소진량(추정): {r['일평균소진(추정)']:.2f}{r['uom']}{" (약 " + str(round(convert_stock_to_cups(r['일평균소진(추정)'], r['uom'], DEFAULT_GRAMS_PER_CUP),1)) + "잔" if r['uom']=='g' else ""}
                        - 판매 가능 일수: {r['판매 가능 일수']}일 ({r['D-day']})
                        - 발주 추천일: {order_txt} (리드타임 {lead_txt}, 공급 방식: {r['supply_mode']})
                        """
                        )

    # 원재료 시세 탭
    if selected_tab == "💸 원재료 시세":
        st.subheader("💸 원재료 시세 비교")

        mapping_rows = load_mapping(DATA_DIR / "price_mapping.csv")
        search_kw_default = ""
        unit = ""
        selected_item: str | None = None

        if mapping_rows:
            options = [
                r.get("item") or r.get("search_keyword")
                for r in mapping_rows
                if (r.get("item") or r.get("search_keyword"))
            ]
            selected_item = st.selectbox("비교할 품목(선택은 옵션)", options, key="price_item_select")

            changed_selection = st.session_state.get("price_selected_item") != selected_item
            if changed_selection:
                st.session_state.price_selected_item = selected_item
                st.session_state.price_rows = None
                st.session_state.price_notes = []

            target = next(
                (r for r in mapping_rows if (r.get("item") or r.get("search_keyword")) == selected_item),
                None,
            )
            if target:
                search_kw_default = str(target.get("search_keyword", "")).strip()
                unit = str(target.get("unit", "")).strip()
                st.caption(f"추천 검색어: {search_kw_default or '입력 필요'} · 단위: {unit or '-'}")
                if changed_selection:
                    st.session_state.price_manual_kw = search_kw_default
        else:
            st.warning("`data/price_mapping.csv`에 품목과 검색어를 추가해 주세요.")
            st.code(
                "item,search_keyword,unit\n"
                "우유 1L,서울우유 1L,1L\n"
                "원두 1kg,원두 1kg 스페셜티,1kg\n"
                "휘핑크림 1L,앵커 휘핑크림 1L,1L",
                language="text",
            )

        # 검색어 직접 입력 (선택값을 기본값으로 자동 채움)
        if "price_manual_kw" not in st.session_state:
            st.session_state.price_manual_kw = search_kw_default
        # 선택 변경 시 입력값도 최신 기본값으로 덮어쓰기
        if search_kw_default and selected_item and st.session_state.get("price_selected_item") == selected_item:
            if st.session_state.price_manual_kw in {"", None}:
                st.session_state.price_manual_kw = search_kw_default

        search_kw_input = st.text_input(
            "검색어 직접 입력",
            key="price_manual_kw",
            placeholder="예: 딸기 1kg, 계피 스틱 1kg",
        )
        search_kw = (search_kw_input or "").strip() or search_kw_default
        if search_kw_default and not search_kw_input:
            st.caption(f"미입력 시 추천 검색어 사용: {search_kw_default}")

        naver_id = get_secret("NAVER_CLIENT_ID")
        naver_secret = get_secret("NAVER_CLIENT_SECRET")
        st.caption(f"키 상태 · NAVER: {'✅' if naver_id and naver_secret else '❌ 없음'}")

        if st.button("시세 불러오기", key="btn_fetch_price"):
            rows_all = []
            notes = []
            if not search_kw:
                notes.append("검색어를 입력해 주세요.")
            else:
                n_rows, n_err = fetch_naver_prices(search_kw, naver_id, naver_secret, limit=1000)
                rows_all = merge_price_rows(rows_all, n_rows)
                if n_err:
                    notes.append(n_err)

            st.session_state.price_rows = rows_all
            st.session_state.price_notes = notes

        for msg in st.session_state.get("price_notes") or []:
            st.warning(msg)

        rows = st.session_state.get("price_rows") or []
        if rows:
            df_prices = pd.DataFrame(rows)
            if not df_prices.empty:
                df_prices["price"] = pd.to_numeric(df_prices["price"], errors="coerce")
                df_prices = df_prices[df_prices["price"] > 0]
                df_prices["title"] = df_prices["title"].apply(
                    lambda x: re.sub(r"<.*?>", "", str(x or ""))
                )
                # 관련도 점수 우선 정렬
                sort_cols = ["match_score", "price"] if "match_score" in df_prices.columns else ["price"]
                df_prices = df_prices.sort_values(sort_cols, ascending=[False, True])
                for col in ["source", "title", "price", "market", "link"]:
                    if col not in df_prices.columns:
                        df_prices[col] = None

            if df_prices.empty:
                st.info("가격 데이터를 불러오지 못했습니다. 검색어/코드나 API 키를 확인해 주세요.")
            else:
                best = df_prices.sort_values("price").iloc[0]
                avg_price = df_prices["price"].mean()
                c1, c2 = st.columns(2)
                c1.metric("최저가", format_krw(best["price"]), f"{best.get('source', '')}")
                c2.metric("평균가", format_krw(avg_price))

                if "link" in df_prices.columns:
                    df_prices["링크"] = df_prices["link"].apply(lambda url: url if isinstance(url, str) else None)

                display_df = df_prices.sort_values("price")[
                    ["source", "title", "price", "market", "링크"]
                ].rename(
                    columns={
                        "source": "출처",
                        "title": "상품명",
                        "price": "가격",
                        "market": "시장/판매처",
                        "링크": "링크",
                    }
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "링크": st.column_config.LinkColumn("링크", display_text="열기"),
                    },
                )
        else:
            st.info("시세를 불러오려면 검색어 입력 후 버튼을 눌러주세요.")

    # AI 영향도 탭
    if selected_tab == "🤖 AI 영향도":
        st.subheader("🤖 메뉴별 재고 영향도 분석 (AI 예측)")
        menu_list_kr = MENU_MASTER_KR
        selected_menu_kr = st.selectbox("분석할 메뉴를 선택하세요", menu_list_kr, key="inv_ai_menu")
        selected_menu_en = from_korean_detail(selected_menu_kr)
        try:
            report_df = compute_ingredient_metrics_for_menu(
                selected_menu_en,
                df,
                df_inv,
                df_params
            )
            if report_df.empty:
                st.warning(f"'{selected_menu_kr}' 레시피 또는 판매 데이터가 부족합니다. 레시피를 저장하고 거래를 추가해 주세요.")
            else:
                display_cols = ['상품상세', '상태', '현재재고', 'uom', '권장발주', '커버일수', '일평균소진', 'ROP']
                formatted_df = report_df[display_cols].copy()
                formatted_df['상태'] = formatted_df['상태'].replace({
                    '🚨 발주요망': '🔴 위험',
                    '✅ 정상': '🟢 충분'
                })
                formatted_df['현재재고'] = formatted_df.apply(lambda r: f"{r['현재재고']:,.1f} {r['uom']}", axis=1)
                formatted_df['권장발주'] = formatted_df.apply(lambda r: f"{r['권장발주']:,.1f} {r['uom']}", axis=1)
                formatted_df['일평균소진'] = formatted_df.apply(lambda r: f"{r['일평균소진']:,.1f} {r['uom']}", axis=1)
                formatted_df['ROP'] = formatted_df.apply(lambda r: f"{r['ROP']:,.1f} {r['uom']}", axis=1)
                formatted_df['커버일수'] = formatted_df['커버일수'].apply(lambda x: f"{x}일")
                st.dataframe(
                    formatted_df[['상품상세', '상태', '현재재고', '권장발주', '커버일수', '일평균소진', 'ROP']].rename(
                        columns={"커버일수": "판매 가능 일수", "ROP": "발주 시점"}
                    ),
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"AI 재고 리포트 생성 중 오류: {e}")



    # ==============================================================
    # TAB 1: (신규) 재고 입력 (영수증 AI)
    # ==============================================================
    if selected_tab == "📸 재고 입력":
        st.subheader("📸 영수증 기반 재고 입고")
        st.caption("원재료 구매 영수증을 업로드하면 AI가 자동으로 내역을 입력해줍니다.")

        # 세션 상태 초기화 (분석 결과를 저장하기 위함)
        if "receipt_result" not in st.session_state:
            st.session_state.receipt_result = None
        if "receipt_meta" not in st.session_state:
            st.session_state.receipt_meta = None

        # --- [화면 1] 업로드 UI ---
        # 분석 결과가 없으면 업로드 화면을 보여줌
        if st.session_state.receipt_result is None:
            st.markdown("### 영수증 사진 업로드")
            
            with st.container(border=True):
                uploaded_file = st.file_uploader(
                    "드래그 앤 드롭 또는 클릭하여 파일 선택", 
                    type=["png", "jpg", "jpeg", "webp"],
                    help="AI가 영수증 정보를 자동으로 추출해 드립니다."
                )
                
                if uploaded_file is not None:
                    # 이미지 미리보기
                    st.image(uploaded_file, caption="업로드된 영수증", width=300)
                    
                    if st.button("🤖 AI 분석 시작", type="primary", use_container_width=True):
                        with st.spinner("AI가 영수증을 읽고 있습니다... (약 5~10초 소요) 🧠"):
                            receipt_meta = save_receipt_image(uploaded_file, "inventory")
                            if not receipt_meta:
                                st.error("영수증 저장에 실패했습니다. 다시 시도해주세요.")
                            else:
                                st.session_state.receipt_meta = receipt_meta
                                # API 호출
                                data = analyze_receipt_image(uploaded_file)
                                
                                if data:
                                    st.session_state.receipt_result = data
                                    st.session_state.receipt_image = uploaded_file # 이미지도 유지
                                    update_receipt_metadata(receipt_meta.get("receipt_id"), {"analysis_result": data})
                                    safe_rerun() # 화면 갱신하여 결과 화면으로 이동

        # --- [화면 2] 분석 결과 확인 및 수정 UI ---
        else:
            st.markdown("### 📝 데이터 검토 및 수정")
            
            data = st.session_state.receipt_result
            
            # 상단: 원본 이미지와 헤더 정보
            col_img, col_info = st.columns([1, 2])
            
            with col_img:
                st.image(st.session_state.receipt_image, caption="원본 이미지", use_container_width=True)
                receipt_meta = st.session_state.get("receipt_meta") or {}
                signed_url = build_signed_url_from_meta(receipt_meta)
                console_url = build_storage_console_url(
                    receipt_meta.get("storage_bucket"),
                    receipt_meta.get("storage_path"),
                )
                if signed_url:
                    st.link_button("저장된 영수증 파일 열기", signed_url, use_container_width=True)
                elif console_url:
                    st.link_button("Storage에서 보기", console_url, use_container_width=True)
                if st.button("🔄 다른 영수증 올리기"):
                    st.session_state.receipt_result = None
                    st.session_state.receipt_image = None
                    st.session_state.receipt_meta = None
                    safe_rerun()

            with col_info:
                st.markdown("#### 영수증 정보")
                with st.container(border=True):
                    c1, c2, c3 = st.columns(3)
                    # AI가 추출한 정보로 초기값 설정
                    store_name = c1.text_input("상호명", value=data.get("store_name", ""))
                    date_val = c2.text_input("거래 날짜", value=data.get("date", ""))
                    time_val = c3.text_input("거래 시간", value=data.get("time", ""))

            st.markdown("#### 📦 물품 목록")
            
            # 품목 리스트를 DataFrame으로 변환
            items_df = pd.DataFrame(data.get("items", []))
            
            # 데이터가 비어있을 경우를 대비해 컬럼 보장
            if items_df.empty:
                items_df = pd.DataFrame(columns=["name", "qty", "price", "total"])
            
            # Data Editor로 표시 (수정 가능하도록)
            edited_items = st.data_editor(
                items_df,
                column_config={
                    "name": st.column_config.TextColumn("물품명"),
                    "qty": st.column_config.NumberColumn("수량", min_value=1),
                    "price": st.column_config.NumberColumn("단가", format="%d원"),
                    "total": st.column_config.NumberColumn("총액", format="%d원"),
                },
                num_rows="dynamic", # 행 추가/삭제 가능
                use_container_width=True,
                key="receipt_editor"
            )

            # 총액 계산 및 표시
            st.markdown("---")
            
            # 계산된 총액 (Data Editor 수정값 반영)
            try:
                calc_total = edited_items["total"].sum()
            except:
                calc_total = 0
                
            ai_total = safe_float(data.get("total_amount", 0), 0)

            col_sum1, col_sum2 = st.columns([3, 1])
            with col_sum2:
                st.metric("계산된 총액", f"{calc_total:,.0f}원", delta=f"AI 인식 금액: {ai_total:,.0f}원")

            # 하단 버튼 액션 (DB 저장 X)
            st.markdown("---")
            btn_col1, btn_col2 = st.columns([1, 4])
            with btn_col2:
                if st.button("💾 DB에 저장 (재고 반영)", type="primary", use_container_width=True):
                    receipt_meta = st.session_state.receipt_meta or {}
                    receipt_id = receipt_meta.get("receipt_id")
                    if not receipt_id and st.session_state.get("receipt_image") is not None:
                        receipt_meta = save_receipt_image(st.session_state.receipt_image, "inventory")
                        receipt_id = receipt_meta.get("receipt_id") if receipt_meta else None

                    stock_date = normalize_receipt_date(date_val, datetime.now().date())
                    stock_time = normalize_receipt_time(time_val, datetime.now().strftime("%H:%M:%S"))

                    inv_lookup = build_inventory_lookup(df_inv)
                    inv_fuzzy = build_inventory_fuzzy_index(df_inv)
                    uom_map = {}
                    if not df_inv.empty and "상품상세_en" in df_inv.columns:
                        uom_map = df_inv.set_index("상품상세_en")["uom"].to_dict()

                    unmatched = []
                    updated = 0
                    final_items = []

                    with st.spinner("재고 데이터를 저장 중..."):
                        for _, row in edited_items.iterrows():
                            name = str(row.get("name", "")).strip()
                            if not name:
                                continue
                            price = safe_float(row.get("price", 0.0), 0.0)
                            total = safe_float(row.get("total", 0.0), 0.0)
                            qty = safe_float(row.get("qty", 0.0), 0.0)
                            if qty == 0 and price > 0 and total > 0:
                                qty = total / price
                            if qty == 0:
                                qty = 1

                            ingredient_en, matched = match_inventory_name(name, inv_lookup, inv_fuzzy)
                            if not matched:
                                unmatched.append(name)

                            uom_val = uom_map.get(ingredient_en, "ea")
                            update_inventory_qty(
                                ingredient_en,
                                qty,
                                uom=uom_val,
                                is_ingredient=True,
                                mode="add",
                                move_type="receipt_restock",
                                note=f"영수증 입고: {name}",
                            )
                            updated += 1
                            final_items.append({
                                "name": name,
                                "ingredient_en": ingredient_en,
                                "qty": qty,
                                "uom": uom_val,
                                "price": price,
                                "total": total,
                            })

                    if updated:
                        update_receipt_metadata(receipt_id, {
                            "inventory_saved_at": datetime.now().isoformat(),
                            "inventory_count": updated,
                            "inventory_items": final_items,
                            "inventory_unmatched_items": list(set(unmatched)),
                            "receipt_store_name": store_name,
                            "receipt_date": str(stock_date),
                            "receipt_time": stock_time,
                            "inventory_total": calc_total,
                        })
                        st.success(f"✅ 재고 {updated}건 반영 완료!")
                        if unmatched:
                            st.warning(f"매칭 실패 품목: {', '.join(sorted(set(unmatched)))}")
                        clear_cache_safe(load_inventory_df, load_all_core_data)
                        safe_rerun()
                    else:
                        st.warning("반영할 재고 항목이 없습니다.")

        st.markdown("---")
        st.subheader("📄 엑셀 업로드로 재고 입고")
        st.caption("지원 컬럼 예시: 상품상세, 수량, 단위(uom), 매입가 (csv/xlsx 가능)")
        mode_label = st.radio(
            "업로드 적용 방식",
            ["입고 수량 추가", "현재 재고로 덮어쓰기"],
            horizontal=True,
            key="inv_excel_mode",
        )
        mode_value = "add" if mode_label == "입고 수량 추가" else "set"

        inv_excel = st.file_uploader(
            "엑셀/CSV 파일 업로드",
            type=["xlsx", "xls", "csv"],
            key="inventory_excel_uploader",
        )
        inv_excel_meta = save_upload_file_once(inv_excel, "inventory_excel_upload", "inventory_excel")
        if inv_excel_meta:
            signed_url = build_signed_url_from_meta(inv_excel_meta)
            console_url = build_storage_console_url(
                inv_excel_meta.get("storage_bucket"),
                inv_excel_meta.get("storage_path"),
            )
            if signed_url:
                st.link_button("업로드 파일 열기", signed_url, use_container_width=True)
            elif console_url:
                st.link_button("Storage에서 보기", console_url, use_container_width=True)
            if inv_excel_meta.get("saved_storage") and inv_excel_meta.get("storage_uri"):
                st.caption(f"Storage 저장 경로: {inv_excel_meta.get('storage_uri')}")
            elif inv_excel_meta.get("saved_local") and inv_excel_meta.get("local_path"):
                st.caption(f"로컬 저장 경로: {inv_excel_meta.get('local_path')}")
        df_inv_upload = read_tabular_upload(inv_excel)
        if df_inv_upload is not None:
            st.dataframe(df_inv_upload.head(20), use_container_width=True)
            col_name = pick_column(df_inv_upload, INVENTORY_UPLOAD_ALIASES["name"])
            col_sku = pick_column(df_inv_upload, INVENTORY_UPLOAD_ALIASES["sku"])
            col_qty = pick_column(df_inv_upload, INVENTORY_UPLOAD_ALIASES["qty"])
            col_uom = pick_column(df_inv_upload, INVENTORY_UPLOAD_ALIASES["uom"])
            col_cost_unit_size = pick_column(df_inv_upload, INVENTORY_UPLOAD_ALIASES["cost_unit_size"])
            col_cost_per_unit = pick_column(df_inv_upload, INVENTORY_UPLOAD_ALIASES["cost_per_unit"])
            col_is_ingredient = pick_column(df_inv_upload, INVENTORY_UPLOAD_ALIASES["is_ingredient"])

            missing_cols = []
            if not col_name and not col_sku:
                missing_cols.append("상품상세 또는 상품상세_en")
            if not col_qty:
                missing_cols.append("수량")
            if missing_cols:
                st.warning(f"필수 컬럼이 부족합니다: {', '.join(missing_cols)}")
            else:
                if st.button("💾 엑셀 재고 반영", type="primary", use_container_width=True, key="inventory_excel_save"):
                    inv_lookup = build_inventory_lookup(df_inv)
                    inv_fuzzy = build_inventory_fuzzy_index(df_inv)
                    unmatched = []
                    updated = 0
                    upload_id = inv_excel_meta.get("upload_id") if inv_excel_meta else None

                    with st.spinner("엑셀 재고 데이터를 반영 중..."):
                        for _, row in df_inv_upload.iterrows():
                            raw_name = str(row.get(col_name, "")).strip() if col_name else ""
                            raw_sku = str(row.get(col_sku, "")).strip() if col_sku else ""
                            if raw_sku:
                                ingredient_en = raw_sku
                                matched = True
                            else:
                                ingredient_en, matched = match_inventory_name(raw_name, inv_lookup, inv_fuzzy)

                            if not ingredient_en:
                                continue

                            qty = safe_float(row.get(col_qty, 0.0), 0.0)
                            if qty == 0:
                                continue

                            uom = row.get(col_uom) if col_uom else None
                            uom_val = normalize_uom(uom) if uom else "ea"

                            cost_unit_size = None
                            if col_cost_unit_size:
                                cost_unit_size = safe_float(row.get(col_cost_unit_size, None), 0.0)
                                if cost_unit_size <= 0:
                                    cost_unit_size = None

                            cost_per_unit = None
                            if col_cost_per_unit:
                                cost_per_unit = safe_float(row.get(col_cost_per_unit, None), 0.0)
                                if cost_per_unit < 0:
                                    cost_per_unit = None

                            is_ingredient = True
                            if col_is_ingredient:
                                flag_raw = row.get(col_is_ingredient)
                                if normalize_cell_str(flag_raw, ""):
                                    is_ingredient = parse_truthy(flag_raw)

                            update_inventory_qty(
                                ingredient_en,
                                qty,
                                uom=uom_val,
                                is_ingredient=is_ingredient,
                                mode=mode_value,
                                cost_unit_size=cost_unit_size,
                                cost_per_unit=cost_per_unit,
                                move_type="excel_import",
                                note=f"엑셀 재고 입력: {raw_name or ingredient_en}",
                            )

                            if not matched:
                                unmatched.append(raw_name or ingredient_en)
                            updated += 1

                    if updated:
                        st.success(f"✅ 엑셀 재고 {updated}건 반영 완료!")
                        if unmatched:
                            st.warning(f"신규/매칭 실패 품목: {', '.join(sorted(set(unmatched)))}")
                        if upload_id:
                            update_upload_metadata(upload_id, {
                                "inventory_updates": updated,
                                "inventory_mode": mode_value,
                                "linked_at": datetime.now().isoformat(),
                                "unmatched_items": list(set(unmatched)),
                            })
                        clear_cache_safe(load_inventory_df, load_all_core_data)
                        safe_rerun()
                    else:
                        st.warning("반영할 재고 항목이 없습니다.")


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
# =============================================================
# 🤖 AI 비서 (SPRINT 1)
# === [UX 개선] '일방적 조언' -> '선택형 브리핑'으로 변경 ===
# =============================================================
# =============================================================
# 🤖 AI 비서 (SPRINT 1)
# === [UX 개선] '새로 분석하기' 버튼 추가 ===
# =============================================================
elif menu == "AI 비서":
    st.header("🤖 AI 비서 (선택형 브리핑)")

    # [수정] 대화 기록을 세션 상태에 저장
    if "messages_ai_v2" not in st.session_state:
        st.session_state.messages_ai_v2 = [{"role": "assistant", "content": "안녕하세요, 사장님! 데이터 분석을 완료했습니다. 무엇을 먼저 브리핑해 드릴까요?"}]
    
    # [수정] 분석 결과를 저장할 세션 상태 초기화
    if "analysis_context" not in st.session_state:
        st.session_state.analysis_context = {
            "risk": None,
            "profit": None,
            "pattern": None
        }

    # [UX 개선 3] '새로 분석하기' 버튼 추가
    if st.button("🔄 최신 데이터로 새로 분석하기", help="새로 추가된 거래 내역을 반영하여 AI 분석을 다시 실행합니다."):
        with st.spinner("최신 데이터를 다시 분석 중입니다... ⏳"):
            # 1. AI 분석 함수를 '강제로' 다시 실행
            risk_report = find_inventory_risks(df, df_inv, df_params)
            profit_report = find_profit_insights(df)
            pattern_report = find_top_correlations(df)
            
            # 2. 세션 상태에 '최신' 분석 결과를 덮어쓰기
            st.session_state.analysis_context['risk'] = risk_report
            st.session_state.analysis_context['profit'] = profit_report
            st.session_state.analysis_context['pattern'] = pattern_report
            
            # 3. 대화 기록도 리셋
            st.session_state.messages_ai_v2 = [{"role": "assistant", "content": "✅ 최신 데이터 분석을 완료했습니다! 무엇을 브리핑해 드릴까요?"}]
            
            st.success("새로 분석 완료!")
            safe_rerun() # 화면 새로고침

    st.markdown("데이터 분석이 완료되었습니다. 브리핑을 요청하세요.")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚨 재고 위험 보기", use_container_width=True):
            # [수정] 버튼 클릭 시점에 '아직 분석 안 한' 경우에만 분석 실행
            if st.session_state.analysis_context.get('risk') is None:
                st.session_state.analysis_context['risk'] = find_inventory_risks(df, df_inv, df_params)
                
            st.session_state.messages_ai_v2.append({"role": "user", "content": "재고 위험 브리핑해줘."})
            st.session_state.messages_ai_v2.append({"role": "assistant", "content": f"**[AI 사실 리포트: 재고 위험]**\n\n{st.session_state.analysis_context['risk']}"})
            safe_rerun()

    with col2:
        if st.button("💰 마진 분석 보기", use_container_width=True):
            if st.session_state.analysis_context.get('profit') is None:
                st.session_state.analysis_context['profit'] = find_profit_insights(df)
                
            st.session_state.messages_ai_v2.append({"role": "user", "content": "마진 분석 브리핑해줘."})
            st.session_state.messages_ai_v2.append({"role": "assistant", "content": f"**[AI 사실 리포트: 마진 분석]**\n\n{st.session_state.analysis_context['profit']}"})
            safe_rerun()

    with col3:
        if st.button("📈 판매 패턴 보기", use_container_width=True):
            if st.session_state.analysis_context.get('pattern') is None:
                st.session_state.analysis_context['pattern'] = find_top_correlations(df)
                
            st.session_state.messages_ai_v2.append({"role": "user", "content": "판매 패턴 브리핑해줘."})
            st.session_state.messages_ai_v2.append({"role": "assistant", "content": f"**[AI 사실 리포트: 판매 패턴]**\n\n{st.session_state.analysis_context['pattern']}"})
            safe_rerun()
            
    st.divider()

    # --- 2. 대화창 UI (기존과 동일) ---
    
    # 이전 대화 내용 표시
    for message in st.session_state.messages_ai_v2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # [수정] st.chat_input 사용
    if prompt := st.chat_input("위 분석 내용에 대해 더 물어보시거나, 다른 것을 요청하세요..."):
        # 사용자 메시지 표시
        st.session_state.messages_ai_v2.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("AI가 분석 내용과 사장님의 질문을 함께 생각 중입니다... 🧠"):
                
                # [수정] AI가 현재까지의 모든 분석 결과를 '사실'로 인지하도록 컨텍스트 조합
                full_context = f"""
                [AI 분석 리포트 1: 재고 위험]
                {st.session_state.analysis_context.get('risk', '아직 분석 안 함')}
                
                [AI 분석 리포트 2: 마진 분석]
                {st.session_state.analysis_context.get('profit', '아직 분석 안 함')}
                
                [AI 분석 리포트 3: 핵심 판매 패턴]
                {st.session_state.analysis_context.get('pattern', '아직 분석 안 함')}
                """
                
                result_text = call_gemini_api(
                    user_prompt=prompt,
                    data_context=full_context
                )
                
                if result_text:
                    st.markdown(result_text)
                    st.session_state.messages_ai_v2.append({"role": "assistant", "content": result_text})
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
# ==============================================================
# ✏️ 데이터 편집
# === [UX 개선] tab2(기능 중복) 삭제, '수익' 자동계산, 삭제 UI 간소화 ===
# ==============================================================
elif menu == "데이터 편집":
    # [수정] 헤더를 통합 편집으로 변경
    st.header("✏️ 데이터 편집")
    render_menu_management("data_edit")

    st.markdown("---")
    st.subheader("🧾 거래 수정/삭제")
    
    df_raw, df_view = load_sales_with_id()
    if df_view.empty:
        st.info("수정할 Firebase 거래 데이터가 없습니다. (CSV는 읽기 전용)")
    else:
        st.caption("💡 Firebase에 저장된 거래 내역만 수정/삭제할 수 있습니다. (가게위치=Firebase)")
        
        if '가게위치' in df_view.columns:
            df_view_fb = df_view[df_view['가게위치'] == 'Firebase'].copy()
        else:
            df_view_fb = pd.DataFrame(columns=df_view.columns) 
        
        if df_view_fb.empty:
            st.info("아직 앱을 통해 추가된(수정 가능한) 거래 데이터가 없습니다.")
        else:
            
            # [수정] '수익' 컬럼을 수정 불가능(disabled)하게 변경
            edited_df = st.data_editor(
                df_view_fb[['_id','날짜','상품상세','수량','단가','수익']],
                column_config={
                    "_id": st.column_config.TextColumn("문서ID", disabled=True),
                    "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
                    "수량": st.column_config.NumberColumn("수량", min_value=0), # 0으로 수정 가능
                    "단가": st.column_config.NumberColumn("단가(원)", format="%d원"),
                    "수익": st.column_config.NumberColumn(
                        "수익 (자동계산)", 
                        disabled=True, # 👈 [핵심 수정] 사용자가 직접 수정 불가
                        format="%d원"
                    )
                },
                hide_index=True,
                num_rows="dynamic" # 👈 [핵심] 여기서 행 삭제 가능
            )
            
        with st.form(key="sales_edit_form"):
            # [유지] 이 체크박스는 디자이너님 요청대로 남겨둡니다.
            reflect_inv = st.checkbox("저장 시 재고에 반영(차감/복원)", value=True)
            
            submit_sales_edit = st.form_submit_button("변경된 내용 저장하기 💾")
        
        if submit_sales_edit:
            changed = 0 # 수정된 행
            deleted = 0 # 삭제된 행
                
            # [수정] data_editor에서 삭제된 행을 먼저 감지
            orig_ids = set(df_view_fb['_id'])
            edited_ids = set(edited_df['_id'])
            deleted_ids = list(orig_ids - edited_ids)

            if deleted_ids:
                for doc_id in deleted_ids:
                    if reflect_inv: # 삭제 시 재고 복원
                        try:
                            orig = df_raw[df_raw['_id'] == doc_id].iloc[0]
                            qty_to_restore = -int(orig.get('수량', 0)) # 수량을 음수로
                            detail_en = orig.get('상품상세')
                            if qty_to_restore != 0 and detail_en:
                                adjust_inventory_by_recipe(detail_en, qty_to_restore, move_type="delete_restore", note=f"Deleted: {doc_id}")
                        except Exception as e:
                            st.warning(f"{doc_id} 재고 복원 실패: {e}")
                    
                    db.collection(SALES_COLLECTION).document(doc_id).delete()
                    deleted += 1

                # [수정] 수정된 행 처리
                for i, new in edited_df.iterrows():
                    doc_id = new['_id']
                    if pd.isna(doc_id): # 새로 추가된 행은 무시 (이 탭은 '수정/삭제' 전용)
                        continue
                        
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
                    
                    # [핵심 수정] '수익' 자동 계산 로직
                    qty_new = int(new['수량'])
                    unit_new = float(new['단가'])
                    rev_new_calculated = qty_new * unit_new # 👈 '수익' 자동 계산

                    qty_changed = qty_new != int(orig.get('수량', 0))
                    unit_changed = unit_new != float(orig.get('단가', 0))
                    
                    if qty_changed:
                        patch['수량'] = qty_new
                    if unit_changed:
                        patch['단가'] = unit_new

                    # 수량/단가가 바뀌었거나, 원래 수익이 잘못 계산됐었다면 '수익' 업데이트
                    if (qty_changed or unit_changed) or (rev_new_calculated != float(orig.get('수익', 0))):
                        patch['수익'] = rev_new_calculated
                    
                    if patch:
                        if reflect_inv and '수량' in patch: # 재고 반영
                            diff = qty_new - int(orig.get('수량', 0))
                            adjust_inventory_by_recipe(detail_en, diff, move_type="edit_adjust", note=str(doc_id))
                        
                        db.collection(SALES_COLLECTION).document(doc_id).update(patch)
                        changed += 1
                
                if changed > 0 or deleted > 0:
                    st.success(f"✅ {changed}건 수정, {deleted}건 삭제 완료")
                    safe_rerun()
                else:
                    st.info("변경된 내용이 없습니다.")

            # [수정] ID 기반의 multiselect 삭제 기능 (st.markdown("---") 이하) 모두 삭제
    
    # [수정] `with tab2:` 블록 전체 삭제

# ==============================================================

# 📋 거래 내역
# ==============================================================
elif menu == "거래 내역":
    st.header("📋 전체 거래 내역")
    if df.empty:
        st.info("표시할 거래 데이터가 없습니다.")
        st.caption(f"디버그: CSV {len(df_csv)}건, Firestore {len(df_fb)}건, 경로 {CSV_PATH}")
    else:
        
        # --- [UX 개선 1] 필터 및 검색 기능 추가 ---
        max_date = df['날짜'].max().date()
        min_date = df['날짜'].min().date()

        # 1. 날짜 필터 (기본값을 전체 구간으로 설정)
        default_start_date = min_date
        default_end_date = max_date

        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input("조회 시작일", value=default_start_date, max_value=max_date)
        with col_date2:
            end_date = st.date_input("조회 종료일", value=default_end_date, min_value=start_date, max_value=max_date)
        
        # 2. 텍스트 검색 필터
        search_query = st.text_input("상품 검색 (상품상세 또는 카테고리)", "")

        # 3. 데이터 필터링 적용 (날짜 필터)
        filtered_df = df[
            (df['날짜'].dt.date >= start_date) & 
            (df['날짜'].dt.date <= end_date)
        ]
        
        # 4. 텍스트 검색 필터 적용
        if search_query:
            # 검색 쿼리가 있으면 상품 상세 또는 카테고리에서 필터링
            filtered_df = filtered_df[
                filtered_df['상품상세'].str.contains(search_query, case=False) |
                filtered_df['상품카테고리'].str.contains(search_query, case=False)
            ]

        # --- [UX 개선 2] 요약 정보 표시 ---
        st.markdown(f"---")
        total_filtered_rows = len(filtered_df)
        total_filtered_revenue = filtered_df['수익'].sum()
        
        c_metric1, c_metric2 = st.columns(2)
        c_metric1.metric("표시된 거래 건수", f"{total_filtered_rows:,} 건")
        c_metric2.metric("표시된 매출 총합", format_krw(total_filtered_revenue))
        
        st.caption(f"총 {len(df)}건의 거래 내역 중, 필터링된 **{total_filtered_rows}건**을 표시하고 있습니다.")
        
        # 5. 필터링된 데이터 출력
        cols = ['날짜','상품상세','수량','단가','수익','상품카테고리','요일','시']
        cols = [c for c in cols if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[cols].sort_values('날짜', ascending=False), 
            width=None, 
            use_container_width=True
        )

elif menu == "연구 검증":
    st.header("🎓 연구 검증 및 기술 실증 (Validation)")
    st.markdown("""
    본 연구는 **시스템 성능**, **AI 모델 신뢰도**, **비용 모델**의 세 가지 핵심 성과를 제시합니다.
    """)
    st.divider()

    # --- 1. 시스템 성능 (Speed) ---
    st.subheader("핵심 성과 1: 시스템 성능 (데이터 처리 속도) 🚀")
    st.metric(f"주 데이터셋 (총 {row_count:,}건) 로딩 및 전처리 시간", f"{load_time:.4f} 초")
    st.caption("증강/현장 CSV를 수 초 내로 로딩해 대시보드에 반영합니다.")
    
    st.divider()

    # --- 2. AI 모델 성능 (MAPE) - 자동 실행 ---
    st.subheader("핵심 성과 2: AI 수요 예측 모델 신뢰도 (백테스팅) 🧠")
    st.markdown("""
    Prophet 모델을 **초기 5개월 데이터로 훈련**하고, 이후 기간의 실제 판매량과 비교합니다.
    """)

    unique_days = df_csv['날짜'].nunique() if not df_csv.empty else 0
    max_test_days = max(1, min(60, unique_days - 10)) if unique_days else 0

    if max_test_days < 3:
        st.warning("검증할 데이터가 충분하지 않습니다. (최소 10일 훈련 + 3일 검증 필요)")
    else:
        default_test_days = min(30, max_test_days)
        test_days_input = st.number_input(
            "검증 기간(일) 선택", 
            min_value=1, max_value=int(max_test_days), value=int(default_test_days),
            help="데이터셋의 마지막 N일을 '검증용(실제값)'으로 사용하여 모델을 테스트합니다."
        )

        with st.spinner(f"모델을 검증하는 중입니다... (테스트 기간: {test_days_input}일) ⏳"):
            mape, fig, msg = run_prophet_backtesting(df_csv, test_days=test_days_input)
        
        if mape is not None:
            st.metric("수요 예측 모델 평균 오차율 (MAPE)", f"{mape:.2f} %")
            st.caption(f"**(연구 결과 해석)** 모델은 향후 {test_days_input}일을 예측할 때 **평균 약 {mape:.2f}%의 오차**를 보였습니다.")
            st.pyplot(fig) 
        else:
            st.error(f"검증 실패: {msg}")
            
    st.divider()

    # --- 2-1. 전달받은 별도 CSV 뷰어 (국문 리포트) ---
    st.subheader("현장 CSV 스냅샷 (별도 제공 파일) 📂")
    tab_aug, tab_prod, tab_hour, tab_top = st.tabs([
        "데이터 증강", "상품매출현황", "시간대별 매출분석", "카피엔드 Top5"
    ])

    with tab_aug:
        df_aug = load_augmented_sales()
        if df_aug is None or df_aug.empty:
            st.info("data/데이터 증강.csv 파일을 찾을 수 없습니다.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("거래 건수", f"{len(df_aug):,} 건")
            c2.metric("총 매출", format_krw(df_aug['price'].sum()))
            latest_ts = df_aug['timestamp'].max()
            latest_txt = latest_ts.strftime("%Y-%m-%d %H:%M") if pd.notna(latest_ts) else "-"
            c3.metric("마지막 거래시각", latest_txt)

            hourly = df_aug.groupby('hour')['price'].sum().reset_index()
            fig_aug = px.bar(hourly, x='hour', y='price', title="시간대별 매출 (증강 데이터)")
            st.plotly_chart(fig_aug, use_container_width=True, key="aug-hourly")
            st.dataframe(df_aug.head(200), use_container_width=True)
            with st.expander("컬럼 구조"):
                st.markdown("""
                - timestamp: 거래 발생 시각 (datetime)
                - menu_item: 메뉴명
                - price: 판매 금액
                - day_of_week: 요일 문자열 (e.g., MONDAY)
                - hour: 시(hour) 숫자
                - day_type: WEEKDAY / WEEKEND 구분
                """)

    with tab_prod:
        df_prod = load_product_status()
        if df_prod is None or df_prod.empty:
            st.info("data/상품매출현황.csv 파일을 찾을 수 없습니다.")
        else:
            c1, c2 = st.columns(2)
            if '판매금액' in df_prod.columns:
                c1.metric("총 판매금액", format_krw(df_prod['판매금액'].sum()))
            if '수량' in df_prod.columns:
                c2.metric("총 판매수량", f"{int(df_prod['수량'].sum()):,} 개")

            if '판매금액' in df_prod.columns:
                top5 = df_prod.sort_values('판매금액', ascending=False).head(5)
                fig_prod = px.bar(top5, x='상품명', y='판매금액', title="상품매출 Top 5", text='판매금액')
                st.plotly_chart(fig_prod, use_container_width=True, key="prod-top5")
            st.dataframe(df_prod, use_container_width=True)
            with st.expander("컬럼 구조"):
                st.markdown("""
                - 상품명: 메뉴 이름
                - 상품코드: 내부 코드
                - 수량: 판매 수량
                - 점유율(수량): 수량 기준 비중 (%)
                - 판매금액: 총 매출액
                - 점유율(금액): 매출액 기준 비중 (%)
                """)

    with tab_hour:
        df_hour = load_hourly_sales()
        if df_hour is None or df_hour.empty:
            st.info("data/시간대별 매출분석.csv 파일을 찾을 수 없습니다.")
        else:
            if {'hour', '총액'}.issubset(df_hour.columns):
                fig_hour = px.bar(df_hour, x='hour', y='총액', title="시간대별 총액", labels={'hour': '시간'})
                st.plotly_chart(fig_hour, use_container_width=True, key="hourly-csv-total")
            st.dataframe(df_hour, use_container_width=True)
            with st.expander("컬럼 구조"):
                st.markdown("""
                - hour: 0~23시
                - 총액: 해당 시간대 총 매출
                - 현금 / 카드 / 현금영수증: 결제 수단별 매출
                - 할인: 할인 적용 금액
                - 거래건수: 트랜잭션 수
                """)

    with tab_top:
        df_top = load_top5_recipe()
        if df_top is None or df_top.empty:
            st.info("data/카피엔드_커피_Top5.csv 파일을 찾을 수 없습니다.")
        else:
            st.caption("Top5 메뉴의 레시피/원가 테이블입니다. (원본 그대로 표시)")
            st.dataframe(df_top, use_container_width=True)
            with st.expander("컬럼 구조"):
                st.markdown("""
                - 메 뉴: 메뉴명
                - 품 목: 재료명
                - 단가(원), 단위, 수량, 개별가: 재료 단가/규격
                - 사용량, 사용 단가, 사용단가 합계: 레시피 소요량 및 원가
                - 판매가격: 메뉴 판매가
                - 원가율: 원가/판매가 비율
                """)

    st.divider()

    # --- 3. 실용적 비용 모델 설계 (Cost Model) ---
    st.subheader("핵심 성과 3: 실용적 비용 모델 설계 (Trade-off 분석) 💰")
    st.markdown("""
    인터뷰 결과(비용 민감도)와 기술적 실증(AI 비용)을 토대로, 본 연구는 2가지 상용화 모델을 제안합니다.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.info("**A. 기본형 (월 $35-50 고정비)**")
        st.markdown("""
        * **포함:** 재고 관리, 데이터 집계, BOM/ROP 계산
        * **대상:** 비용에 극도로 민감하며, 운영 자동화가 최우선인 카페
        """)
    with col2:
        st.warning("**B. AI 확장형 (월 $50 + 변동비)**")
        st.markdown("""
        * **포함:** 기본형 + AI 비서 (Gemini), 수요 예측 (Prophet)
        * **대상:** 마케팅, 신메뉴 개발 등 데이터 기반 의사결정이 필요한 카페
        """)
    st.caption("이는 소상공인이 자신의 예산과 필요에 맞춰 합리적인 DX(디지털 전환)를 선택할 수 있게 하는 실용적인 설계안입니다.")
# ==============================================================
# ❓ 도움말
# ==============================================================
# ==============================================================
# ❓ 도움말
# ==============================================================
elif menu == "도움말":
    st.header("☕️ 커피 원두 재고관리 파이프라인 쉽게 이해하기")
    
    st.markdown("""
> **“커피 원두가 어떻게 들어오고, 얼마나 쓰이고, 언제 다시 주문돼야 하는지를 자동으로 관리하자!”** 엑셀 대신 ERP가 자동으로 계산해줍니다.
""")

    # ------------------------------------------------------------------
    # ✅ (추가됨) 웹사이트 사용법 섹션
    # ------------------------------------------------------------------
    st.subheader("📚 웹사이트 사용법 가이드: 메뉴별 핵심 기능")
    st.markdown("---")
    
    st.markdown("""
    본 대시보드는 **데이터 확인**, **재고/원가 관리**, **거래 입력/수정**, **AI 분석**의 네 가지 영역으로 나뉩니다.
    
    ### 1. 📊 데이터 확인 및 분석
    
    | 메뉴 | 목적 | 주요 표시 항목 |
    | :--- | :--- | :--- |
    | **경영 현황** | 전체 총괄 요약 (매출액, 판매 건수) | 총 매출, 건당 평균 매출, 카테고리별/일자별 매출 추이 |
    | **매출 대시보드** | 주요 매출 추세 파악 | 월별/카테고리별 누적 매출, 상품 구조별 (트리맵) 매출 기여도 |
    | **기간별 분석** | 특정 기간의 판매 특성 분석 | 요일별 매출 (바 차트), 시간대별 매출 추이 (라인 차트) |
    | **거래 내역** | 원천 데이터 조회 | 필터링/검색 기능으로 특정 기간, 상품의 거래 내역 확인 |
    
    ---
    
    ### 2. ⚙️ 관리 및 편집
    
    | 메뉴 | 목적 | 주요 작업 |
    | :--- | :--- | :--- |
    | **거래 추가** | 신규 판매 거래 입력 | 카테고리/타입/상세 선택 후 수량/단가 입력 → **재고 자동 차감** |
    | **데이터 편집** | 기존 거래 수정/삭제 | Firebase 데이터 수정/삭제, **수량 변경 시 재고 자동 조정** |
    | **재고 관리** | 재료/메뉴 마스터 및 레시피(BOM) 관리 | 재료/원가/현재 재고 수기 입력, 메뉴별 레시피 등록 및 수정 |
    
    ---
    
    ### 3. 🤖 AI 기반 의사 결정
    
    | 메뉴 | 목적 | 주요 제공 기능 |
    | :--- | :--- | :--- |
    | **AI 비서** | 데이터 기반 실행 조언 제공 | **재고 위험**, **마진 분석**, **판매 패턴** 등 3대 리포트 및 질의응답 |
    | **재고 관리** (레시피 탭) | 메뉴 제조에 필요한 재료의 적정 발주량 계산 | AI 예측 기반 **권장 발주량, 커버 일수, ROP(발주점)** 계산 및 차트 제공 |
    
    """)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    
    st.markdown("---")
    # [AI/ML 통합 수정] 도움말 내용 업데이트 (기존 ERP 파이프라인)
    st.markdown("## 📊 ERP 파이프라인 작동 원리")
    st.markdown("""
### 1. (ML) 스마트 발주 로직 (재고 관리 탭)
| 단계 | 하는 일 |
| --- | --- |
| **1. 수요 예측** | Prophet (ML)이 "아메리카노"의 **미래 21일** 판매량을 [500잔]으로 예측 |
| **2. 소진량 계산** | [500잔] x [레시피: 잔당 20g] = **[10,000g]** (예상 총 소진량) |
| **3. 권장 발주량** | [10,000g] - [현재 재고: 3,000g] = **[7,000g]** (권장 발주량) |
| **4. ROP (발주점)** | (일평균소진 * 리드타임) + 안전재고. 이보다 재고가 낮으면 **'🚨 발주요망'** 알림 |
### 2. (AI) 마케팅 보조 (AI 비서 탭)
| 기능 | 설명 |
| --- | --- |
| **요청에 대한 응답 생성** | 현재 데이터를 기반으로 AI가 전략, 홍보 문구를 자동 생성합니다. |
| **운영 보고** | 일일 매출, 판매 건수 등을 요약하여 간결한 보고서를 생성합니다. |
### 3. 기본 데이터 흐름
| 단계 | 하는 일 |
| --- | --- |
| **1. 원두 입고** | '재고 관리' > '재료/원가 마스터'에서 [+10,000g] 수동 입력 및 저장 |
| **2. 판매 발생** | '거래 추가' 탭 또는 POS에서 '아메리카노' 1잔 판매 (Firestore 'coffee_sales'에 기록) |
| **3. 자동 차감** | 시스템이 '아메리카노' 레시피(BOM)를 조회하여 [원두: 20g] 사용 확인 |
| **4. 재고 반영** | 'inventory' DB의 '원두' 재고를 [-20g] 자동 차감 (재고 이동 로그 기록) |
""")
