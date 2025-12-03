import csv
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import requests

BAD_KEYWORDS = {
    "받침대",
    "거치대",
    "케이스",
    "모형",
    "장식",
    "액자",
    "스티커",
    "쿠션",
    "담요",
    "인형",
    "키링",
    "브로치",
    "텀블러",
    "보틀",
    "머그",
    "정리함",
    "선반",
    "수납",
    "마감",
    "판매마감",
    "접수마감",
    "예약",
    "사전예약",
    "프리오더",
    "품절",
    "매진",
    # 식자재와 무관한 가공/토핑/부자재
    "시럽",
    "원액",
    "청",
    "수제청",
    "스프레드",
    "가루",
    "분말",
    "파우더",
    "크런치",
    "쿠키",
    "초콜릿",
    "코팅",
    "레진",
    "향",
    "향료",
    "오일",
    "펄",
    "모래",
    "스프링클",
    "데코",
    "토핑",
    "슬러시",
    "슬러쉬",
    "젤리",
    "젤라틴",
    "젤",
}

ALLOWED_CATEGORY1 = {"식품", "가공식품", "축산물", "수산물", "농산물"}


def load_mapping(path: Path | str = "data/price_mapping.csv") -> List[dict]:
    """Read the mapping CSV that links items to search keywords."""
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


def _normalize_title(title: str) -> str:
    txt = re.sub(r"<.*?>", "", str(title or ""))
    return txt.lower()


def _tokenize_query(query: str) -> List[str]:
    tokens = re.split(r"[\s/,+]+", query.lower())
    skip = {"kg", "g", "l", "ml", ""}
    return [t for t in tokens if t not in skip]


def _extract_qty_tokens(query: str) -> List[str]:
    pattern = re.compile(r"(\d+\s*(?:kg|g|l|ml|입|box|팩|개))", re.IGNORECASE)
    return list({m.group(1).strip().lower() for m in pattern.finditer(query or "")})


def _match_score(title: str, tokens: List[str]) -> float:
    title_tokens = [t for t in re.split(r"[^0-9a-zA-Z가-힣]+", title) if t]
    if not tokens:
        return 1.0
    hits = 0
    for t in tokens:
        if len(t) <= 1:
            hits += 1 if t in title_tokens else 0
        else:
            hits += 1 if t in title else 0
    return hits / len(tokens)


def fetch_naver_prices(
    query: str,
    client_id: str | None,
    client_secret: str | None,
    limit: int = 100,
) -> Tuple[List[dict], str | None]:
    """Call Naver Shopping Search API and return price rows (filtered by relevance)."""
    if not query:
        return [], "검색어가 비어 있습니다."
    if not client_id or not client_secret:
        return [], "네이버 API 키를 st.secrets 또는 환경변수에 설정하세요."

    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    display = max(1, min(limit, 100))  # NAVER API 최대 100
    params = {"query": query, "display": display, "sort": "sim"}  # sim: 네이버 기본 연관도
    tokens = _tokenize_query(query)
    qty_tokens = _extract_qty_tokens(query)
    try:
        res = requests.get(url, headers=headers, params=params, timeout=8)
        res.raise_for_status()
        items = res.json().get("items", [])
        rows: List[dict] = []
        fallback_rows: List[dict] = []
        seen_titles = set()
        for item in items:
            title_clean = _normalize_title(item.get("title"))
            if any(bad in title_clean for bad in BAD_KEYWORDS):
                continue
            score = _match_score(title_clean, tokens)
            title_tokens = [t for t in re.split(r"[^0-9a-zA-Z가-힣]+", title_clean) if t]
            # 수량/단위 토큰 가중치
            if qty_tokens:
                if any(qt in title_clean for qt in qty_tokens):
                    score += 0.5
                else:
                    score -= 0.2
            # 연관도 필터: 토큰 대비 60% 이상 일치(단일 토큰이면 0.4)
            min_score = 0.6 if len(tokens) > 1 else 0.4
            hits = sum(1 for t in tokens if (t in title_clean or t in title_tokens))
            min_hits = len(tokens) if len(tokens) <= 2 else len(tokens) - 1

            # 중복 제목 제거
            if title_clean in seen_titles:
                continue
            seen_titles.add(title_clean)

            row = {
                "source": "네이버",
                "title": item.get("title"),
                "price": _parse_price(item.get("lprice")),
                "link": item.get("link"),
                "match_score": score,
                "hits": hits,
                "category1": item.get("category1"),
            }

            # 모든 결과를 일단 fallback 후보에 넣고, 점수는 그대로 사용
            fallback_rows.append(row)

            # 엄격 필터
            cat1 = str(item.get("category1", "")).strip()
            if ALLOWED_CATEGORY1 and cat1 and cat1 not in ALLOWED_CATEGORY1:
                continue
            if score < min_score:
                continue
            if hits < min_hits:
                continue
            rows.append(row)

        # 1차(엄격) 결과가 너무 적으면 완화 버전 사용
        if len(rows) < 8 and fallback_rows:
            loose = [
                r for r in fallback_rows
                if r.get("match_score", 0) >= 0.25 and r.get("hits", 0) >= 1
            ]
            rows = loose

        rows = sorted(rows, key=lambda r: (-r.get("match_score", 0), r.get("price", 0)))
        return rows, None
    except Exception as e:
        return [], f"네이버 검색 실패: {e}"


def merge_price_rows(*sources: Iterable[dict]) -> List[dict]:
    """Flatten iterable price rows into a list."""
    merged: List[dict] = []
    for src in sources:
        if not src:
            continue
        merged.extend(list(src))
    return merged
