"""
Normalize Firestore menu names to the 증강 CSV 7개 기준.

Targets:
  - recipes: rename docs to normalized IDs (delete docs outside the 7개)
  - coffee_sales & sales: normalize '상품상세' field (delete docs outside the 7개)
  - inventory: rename non-ingredient docs if they map into the 7개 (keep others)

Usage:
  python scripts/normalize_firestore_names.py --apply   # 실제 반영
  python scripts/normalize_firestore_names.py           # 드라이런 (기본)
"""

import argparse
import json
import os
import re
from collections import defaultdict

import firebase_admin
from firebase_admin import credentials, firestore

# --- 기준 메뉴 (증강 CSV 7개) ---
MENU_MASTER = {
    "Americano (I/H)",
    "Caffè Latte (I/H)",
    "Dolce Latte (Iced)",
    "Hazelnut Americano (Iced)",
    "Honey Americano (Iced)",
    "Shakerato (Iced)",
    "Vanilla Bean Latte (Iced)",
}

# --- 이름 정규화 매핑 (caffiend.py과 동일) ---
NAME_MAP = {
    # 영문 변형
    "Americano_(I_H)": "Americano (I/H)",
    "Americano (I H)": "Americano (I/H)",
    "Caffè_Latte_(I_H)": "Caffè Latte (I/H)",
    "Latte": "Caffè Latte (I/H)",
    "Hazelnut_Americano_(Iced)": "Hazelnut Americano (Iced)",
    "Honey_Americano_(Iced)": "Honey Americano (Iced)",
    "Shakerato_(Iced)": "Shakerato (Iced)",
    "Vanilla_Bean_Latte_(Iced)": "Vanilla Bean Latte (Iced)",
    "Dolce_Latte_(Iced)": "Dolce Latte (Iced)",
    # 한글/레거시 변형
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
}


def apply_name_map(name: str | None) -> str:
    if name is None:
        return ""
    raw = str(name).strip()
    if raw in NAME_MAP:
        return NAME_MAP[raw]
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


def init_firestore():
    if firebase_admin._apps:
        return firestore.client()
    # env json
    gac_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if gac_json:
        cred = credentials.Certificate(json.loads(gac_json))
        firebase_admin.initialize_app(cred)
        return firestore.client()
    # file path
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and os.path.exists(os.path.expanduser(gac)):
        firebase_admin.initialize_app()
        return firestore.client()
    # keys/serviceAccount.json
    sa_path = os.path.join(os.path.dirname(__file__), "..", "keys", "serviceAccount.json")
    sa_path = os.path.abspath(sa_path)
    if os.path.exists(sa_path):
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
        return firestore.client()
    raise SystemExit("Firebase credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS or provide keys/serviceAccount.json")


def normalize_sales(db, collection: str, dry_run: bool):
    print(f"\n[Sales] Collection: {collection}")
    docs = list(db.collection(collection).stream())
    stats = defaultdict(int)
    for d in docs:
        data = d.to_dict() or {}
        orig = data.get("상품상세") or data.get("product_detail")
        norm = apply_name_map(orig)
        if not orig:
            stats["skip_no_name"] += 1
            continue
        if norm not in MENU_MASTER:
            stats["delete"] += 1
            if not dry_run:
                d.reference.delete()
            continue
        if norm != orig:
            stats["update"] += 1
            if not dry_run:
                d.reference.update({"상품상세": norm})
        else:
            stats["keep"] += 1
    print(f"  keep={stats['keep']}, update={stats['update']}, delete={stats['delete']}, skip_no_name={stats['skip_no_name']}")


def normalize_recipes(db, dry_run: bool):
    print("\n[Recipes]")
    col = db.collection("recipes")
    docs = list(col.stream())
    stats = defaultdict(int)
    for d in docs:
        doc_id = d.id
        norm = apply_name_map(doc_id)
        doc_id_safe = norm.replace("/", "_")  # Firestore 문서 ID에 '/' 불가
        data = d.to_dict() or {}
        if norm not in MENU_MASTER:
            stats["delete"] += 1
            if not dry_run:
                d.reference.delete()
            continue
        if doc_id_safe == doc_id:
            stats["keep"] += 1
            continue
        stats["rename"] += 1
        if dry_run:
            continue
        target_ref = col.document(doc_id_safe)
        if target_ref.get().exists:
            # If target exists, prefer existing; drop source
            d.reference.delete()
            continue
        target_ref.set(data)
        d.reference.delete()
    print(f"  keep={stats['keep']}, rename={stats['rename']}, delete={stats['delete']}")


def normalize_inventory(db, dry_run: bool):
    print("\n[Inventory] (메뉴명만 정규화, 재료는 그대로 둠)")
    col = db.collection("inventory")
    docs = list(col.stream())
    stats = defaultdict(int)
    for d in docs:
        doc_id = d.id
        data = d.to_dict() or {}
        is_ing = bool(data.get("is_ingredient", False))
        if is_ing:
            stats["skip_ingredient"] += 1
            continue
        norm = apply_name_map(doc_id)
        doc_id_safe = norm.replace("/", "_")
        if norm not in MENU_MASTER:
            stats["keep_other"] += 1  # leave as-is
            continue
        if doc_id_safe == doc_id:
            stats["keep"] += 1
            continue
        stats["rename"] += 1
        if dry_run:
            continue
        target_ref = col.document(doc_id_safe)
        if target_ref.get().exists:
            # keep existing, drop current
            d.reference.delete()
            continue
        target_ref.set(data | {"상품상세_en": norm})
        d.reference.delete()
    print(f"  keep={stats['keep']}, rename={stats['rename']}, keep_other={stats['keep_other']}, skip_ingredient={stats['skip_ingredient']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="실제 Firestore에 반영 (기본은 드라이런)")
    args = parser.parse_args()
    dry_run = not args.apply
    print(f"### DRY_RUN = {dry_run}")
    db = init_firestore()
    normalize_recipes(db, dry_run)
    normalize_sales(db, "coffee_sales", dry_run)
    normalize_sales(db, "sales", dry_run)
    normalize_inventory(db, dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
