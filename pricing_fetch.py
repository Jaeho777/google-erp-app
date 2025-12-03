import csv
from datetime import date
from pathlib import Path
from typing import Iterable, List, Tuple

import requests


def load_mapping(path: Path | str = "data/price_mapping.csv") -> List[dict]:
    """Read the mapping CSV that links items to search keywords and KAMIS codes."""
    try:
        p = Path(path)
        if not p.exists():
            return []
        with p.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _parse_price(val) -> float:
    try:
        if val is None:
            return 0.0
        return float(str(val).replace(",", "").strip())
    except Exception:
        return 0.0


def fetch_naver_prices(
    query: str,
    client_id: str | None,
    client_secret: str | None,
    limit: int = 10,
) -> Tuple[List[dict], str | None]:
    """Call Naver Shopping Search API and return price rows."""
    if not query:
        return [], "검색어가 비어 있습니다."
    if not client_id or not client_secret:
        return [], "네이버 API 키를 st.secrets 또는 환경변수에 설정하세요."

    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {"query": query, "display": limit, "sort": "asc"}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=8)
        res.raise_for_status()
        items = res.json().get("items", [])
        rows = []
        for item in items:
            rows.append(
                {
                    "source": "네이버",
                    "title": item.get("title"),
                    "price": _parse_price(item.get("lprice")),
                    "link": item.get("link"),
                }
            )
        return rows, None
    except Exception as e:
        return [], f"네이버 검색 실패: {e}"


def fetch_kamis_prices(
    item_code: str | None,
    kamis_id: str | None,
    kamis_key: str | None,
    for_date: date | None = None,
) -> Tuple[List[dict], str | None]:
    """Call KAMIS daily price API for a given product code."""
    if not item_code:
        return [], "KAMIS 품목코드가 없습니다."
    if not kamis_id or not kamis_key:
        return [], "KAMIS 인증키(ID/key)를 st.secrets에 설정하세요."

    day = (for_date or date.today()).strftime("%Y-%m-%d")
    url = "https://www.kamis.or.kr/service/price/xml.do"
    params = {
        "action": "dailyPriceByCategoryList",
        "p_productno": item_code,
        "p_regday": day,
        "p_cert_key": kamis_key,
        "p_cert_id": kamis_id,
        "p_returntype": "json",
    }
    try:
        res = requests.get(url, params=params, timeout=8)
        res.raise_for_status()
        data = res.json().get("data", [])
        rows = []
        for d in data:
            rows.append(
                {
                    "source": "KAMIS",
                    "title": d.get("item_name"),
                    "price": _parse_price(d.get("dpr1") or d.get("dpr2")),
                    "market": d.get("market_name"),
                }
            )
        return rows, None
    except Exception as e:
        return [], f"KAMIS 호출 실패: {e}"


def merge_price_rows(*sources: Iterable[dict]) -> List[dict]:
    """Flatten iterable price rows into a list."""
    merged: List[dict] = []
    for src in sources:
        if not src:
            continue
        merged.extend(list(src))
    return merged
