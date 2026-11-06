"""Generate Dobby-compatible products and orders fixtures."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

BUSINESS_ID = "shop_01"
OUT = Path(__file__).resolve().parent.parent / "data"
RNG = random.Random(42)
UTC = timezone.utc


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def person(identifier: str, first_name: str, last_name: str, role: str) -> dict:
    return {"id": f"{BUSINESS_ID}_{identifier}", "first_name": first_name, "last_name": last_name, "role": role}


CATEGORIES = [
    ("cat_coffee", "Καφές", "Coffee"),
    ("cat_food", "Φαγητό", "Food"),
    ("cat_dessert", "Γλυκά", "Desserts"),
]


def category(identifier: str, name_gr: str, name_en: str, sequence: int) -> dict:
    now = iso(datetime(2026, 1, 1, tzinfo=UTC))
    return {
        "id": f"{BUSINESS_ID}_{identifier}", "business_id": BUSINESS_ID,
        "name_gr": name_gr, "name_en": name_en, "internal_name": name_en.lower(),
        "names": [{"name": name_gr, "locale": "gr"}, {"name": name_en, "locale": "en"}],
        "print_order": sequence, "sequence": sequence, "product_sequence": None,
        "order_types": ["dine_in", "delivery", "takeaway"],
        "created_at": now, "updated_at": now,
    }


CATEGORY_OBJECTS = {identifier: category(identifier, gr, en, index) for index, (identifier, gr, en) in enumerate(CATEGORIES, 1)}
PRODUCT_DEFINITIONS = [
    ("prod_freddo_espresso", "Freddo espresso", "Freddo espresso", "cat_coffee", 350, 380, 360, 13),
    ("prod_freddo_cappuccino", "Freddo cappuccino", "Freddo cappuccino", "cat_coffee", 400, 430, 410, 13),
    ("prod_koulouri", "Κουλούρι ολικής", "Whole wheat koulouri", "cat_food", 60, 70, 65, 13),
    ("prod_club_sandwich", "Club sandwich", "Club sandwich", "cat_food", 850, 900, 880, 13),
    ("prod_cheesecake", "Cheesecake", "Cheesecake", "cat_dessert", 450, 480, 460, 24),
]


def make_products() -> list[dict]:
    now = iso(datetime(2026, 1, 1, tzinfo=UTC))
    products = []
    for identifier, name_gr, name_en, category_id, store, delivery, takeaway, vat in PRODUCT_DEFINITIONS:
        cat = CATEGORY_OBJECTS[category_id]
        products.append({
            "id": f"{BUSINESS_ID}_{identifier}", "business_id": BUSINESS_ID, "barcode_id": "",
            "name_gr": name_gr, "name_en": name_en, "internal_name": name_en.lower(),
            "names": [{"name": name_gr, "locale": "gr"}, {"name": name_en, "locale": "en"}],
            "description_gr": name_gr, "description_en": name_en, "visibility": {"menu": True},
            "category": cat, "price_store": store, "price_delivery": delivery,
            "price_takeaway": takeaway, "vat_rate": vat, "vat_rate_delivery": vat,
            "vat_rate_takeaway": vat, "taxes": [], "my_data_classification": {},
            "published": True, "available": True,
            "order_types": ["dine_in", "delivery", "takeaway"],
            "quantity_total": 1000, "quantity_portion": 1, "quantity_type": "pieces",
            "quantity_notify": 20, "enable_measurement": False, "optional_ingredients": False,
            "ingredients_ticket_show_selected": False, "created_at": now, "updated_at": now,
        })
    return products


def make_item(product: dict, order_type: str, item_index: int, created_at: datetime, quantity: int) -> dict:
    price = {"dine_in": product["price_store"], "delivery": product["price_delivery"], "takeaway": product["price_takeaway"]}[order_type]
    vat_rate = product["vat_rate"]
    vat = round(price * vat_rate / (100 + vat_rate))
    net = price - vat
    category = product["category"]
    return {
        "id": f"{BUSINESS_ID}_item_{item_index}", "product_id": product["id"],
        "name_gr": product["name_gr"], "name_en": product["name_en"], "internal_name": product["internal_name"],
        "category": {key: category[key] for key in ("id", "internal_name", "name_gr", "name_en")},
        "price": price, "price_with_extras": price, "price_base": price, "vat_rate": vat_rate,
        "quantity": quantity, "added_at": iso(created_at), "selections": [], "extras": [],
        "ingredients": [], "taxes": [], "net_price": net, "total_vat": vat,
        "total_discount": 0, "total_amount": price, "discounts": [],
        "payment": [{"type": "card", "recipient": person("cashier_01", "Maria", "Dobby", "cashier"),
                     "paid_at": iso(created_at + timedelta(minutes=10)), "amount": price}],
        "quantity_printed": quantity, "quantity_forms_printed": quantity, "quantity_invoiced": quantity,
        "my_data_classification": {"category": "category1_1", "type": "E3_561_003"},
        "invoice_classifications": [], "source": {"menu_id": f"{BUSINESS_ID}_menu_main", "menu_name": "Βασικός κατάλογος"},
    }


def make_orders(products: list[dict], count: int = 180) -> list[dict]:
    by_name = {product["internal_name"]: product for product in products}
    orders = []
    for index in range(count):
        created = datetime(2026, 9, 1, 8, tzinfo=UTC) + timedelta(days=index % 60, hours=RNG.choice([1, 4, 7, 10, 12]))
        order_type = RNG.choice(["dine_in", "delivery", "takeaway"])
        main = by_name[RNG.choice(["freddo espresso", "club sandwich", "cheesecake"])]
        selected = [main]
        if RNG.random() < 0.6:
            selected.append(by_name["koulouri ολικής"] if "koulouri ολικής" in by_name else products[2])
        items = [make_item(product, order_type, index * 3 + item_index, created, 1) for item_index, product in enumerate(selected)]
        total = sum(item["total_amount"] * item["quantity"] for item in items)
        total_vat = sum(item["total_vat"] * item["quantity"] for item in items)
        paid_at = iso(created + timedelta(minutes=10))
        order = {
            "id": f"{BUSINESS_ID}_order_{index + 1:05d}", "daily_id": str(index % 30 + 1),
            "short_id": f"{index + 1:04d}", "business_id": BUSINESS_ID, "type": order_type,
            "status": "completed", "fulfillment": {"status": "picked_up", "preparing_at": iso(created), "picked_up_at": paid_at},
            "items": items, "initiator": person("waiter_01", "Nikos", "Dobby", "waiter"),
            "assignee": person("waiter_01", "Nikos", "Dobby", "waiter"),
            "price_analysis": {
                "total_vat": {"category_a": {"net_amount": total - total_vat, "vat_amount": total_vat},
                              "category_b": {"net_amount": 0, "vat_amount": 0}, "category_c": {"net_amount": 0, "vat_amount": 0},
                              "category_d": {"net_amount": 0, "vat_amount": 0}, "category_e": {"net_amount": 0, "vat_amount": 0},
                              "sum": {"net_amount": total - total_vat, "vat_amount": total_vat}},
                "total_products": sum(item["quantity"] for item in items), "total_amount_before_discount": total,
                "total_discount": 0, "total_amount": total, "total_paid": total, "total_remaining": 0,
            },
            "provider": {} if order_type != "delivery" else {"name": "mock_delivery"},
            "created_at": iso(created), "updated_at": paid_at,
        }
        if order_type == "dine_in":
            order["table"] = {"id": f"{BUSINESS_ID}_table_01", "name": "1",
                              "position": {"id": f"{BUSINESS_ID}_position_main", "name": "Main room"}, "party_size": 2}
        if order_type == "delivery":
            order["customer"] = {"id": f"{BUSINESS_ID}_customer_01", "name": "Eleni Papadopoulou",
                                  "phone_id": f"{BUSINESS_ID}_phone_01", "phone": "+302100000000",
                                  "address_id": f"{BUSINESS_ID}_address_01", "street_name": "Ερμού",
                                  "street_number": "10", "area": "Αθήνα", "zip_code": "10563", "floor": "2", "doorbell": "E. Papadopoulou"}
            order["courier"] = {**person("courier_01", "Giorgos", "Dobby", "delivery"), "assigned_at": paid_at}
        orders.append(order)
    return orders


def make_sales_stats(products: list[dict], orders: list[dict]) -> list[dict]:
    totals: dict[str, dict[str, int]] = {}
    for order in orders:
        for item in order["items"]:
            product_id = item["product_id"]
            record = totals.setdefault(product_id, {"quantity": 0, "total_amount": 0})
            record["quantity"] += item["quantity"]
            record["total_amount"] += item["total_amount"] * item["quantity"]

    stats = []
    for product in products:
        total = totals.get(product["id"], {"quantity": 0, "total_amount": 0})
        category = product["category"]
        stats.append({
            "id": product["id"],
            "name_gr": product["name_gr"],
            "name_en": product["name_en"],
            "internal_name": product["internal_name"],
            "category": {
                "id": category["id"],
                "internal_name": category["internal_name"],
                "name_gr": category["name_gr"],
                "name_en": category["name_en"],
            },
            "price": product["price_store"],
            "quantity": total["quantity"],
            "total_amount": total["total_amount"],
            "discount_types": {},
            "sources": [{
                "source": {"menu_id": f"{BUSINESS_ID}_menu_main", "menu_name": "Βασικός κατάλογος"},
                "quantity": total["quantity"],
                "total_amount": total["total_amount"],
            }],
        })
    return stats


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    products = make_products()
    orders = make_orders(products)
    stats = make_sales_stats(products, orders)
    for filename, payload in (
        ("products.json", products),
        ("orders.json", orders),
        ("sales_stats.json", stats),
    ):
        (OUT / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(products)} products, {len(orders)} orders, and {len(stats)} sales stats in {OUT}")
