"""Generate deterministic, multi-business Dobby-compatible restaurant fixtures."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

BUSINESS_TIMEZONE = ZoneInfo("Europe/Athens")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
DEFAULT_SEED = 42
DEFAULT_DAYS = 300
DEFAULT_ORDERS_PER_DAY = 12
ORDER_TYPES = ("dine_in", "delivery", "takeaway")
BUSINESSES = (
    {"id": "shop_university", "orders": 1.15, "weekend": 0.62, "season": 0.95,
     "peaks": (1.45, 0.85, 0.85, 0.55), "types": (0.32, 0.18, 0.50)},
    {"id": "shop_commercial", "orders": 1.0, "weekend": 0.72, "season": 1.0,
     "peaks": (0.75, 1.55, 0.95, 0.85), "types": (0.42, 0.25, 0.33)},
    {"id": "shop_tourist", "orders": 0.82, "weekend": 1.35, "season": 1.18,
     "peaks": (0.65, 0.85, 1.15, 1.5), "types": (0.50, 0.30, 0.20)},
)

CATEGORY_DEFINITIONS = [
    ("cat_coffee", "Καφές", "Coffee"), ("cat_cold_beverage", "Κρύα ροφήματα", "Cold beverages"),
    ("cat_hot_beverage", "Ζεστά ροφήματα", "Hot beverages"), ("cat_breakfast", "Πρωινό", "Breakfast"),
    ("cat_sandwich", "Σάντουιτς", "Sandwiches"), ("cat_food", "Φαγητό", "Food"),
    ("cat_snack", "Σνακ", "Snacks"), ("cat_dessert", "Γλυκά", "Desserts"),
]

PRODUCT_DEFINITIONS = [
    ("freddo_espresso", "Freddo espresso", "Freddo espresso", "cat_coffee", 350, 380, 360, 13, 1.35),
    ("freddo_cappuccino", "Freddo cappuccino", "Freddo cappuccino", "cat_coffee", 400, 430, 410, 13, 1.20),
    ("espresso", "Εσπρέσο", "Espresso", "cat_coffee", 250, 280, 260, 13, 0.65),
    ("filter_coffee", "Καφές φίλτρου", "Filter coffee", "cat_coffee", 300, 330, 310, 13, 0.75),
    ("iced_latte", "Κρύο latte", "Iced latte", "cat_coffee", 430, 460, 440, 13, 0.85),
    ("hot_chocolate", "Ζεστή σοκολάτα", "Hot chocolate", "cat_hot_beverage", 420, 450, 430, 13, 0.48),
    ("tea", "Τσάι βουνού", "Mountain tea", "cat_hot_beverage", 300, 330, 310, 13, 0.38),
    ("fresh_orange", "Φυσικός χυμός πορτοκάλι", "Fresh orange juice", "cat_cold_beverage", 450, 490, 460, 13, 0.62),
    ("lemonade", "Σπιτική λεμονάδα", "Homemade lemonade", "cat_cold_beverage", 400, 440, 410, 13, 0.48),
    ("soft_drink", "Αναψυκτικό 330ml", "Soft drink 330ml", "cat_cold_beverage", 250, 290, 260, 24, 0.58),
    ("sparkling_water", "Ανθρακούχο νερό", "Sparkling water", "cat_cold_beverage", 180, 210, 190, 24, 0.42),
    ("water", "Νερό 500ml", "Water 500ml", "cat_cold_beverage", 100, 120, 110, 13, 0.85),
    ("koulouri", "Κουλούρι ολικής", "Whole wheat koulouri", "cat_breakfast", 120, 140, 130, 13, 0.92),
    ("croissant", "Κρουασάν βουτύρου", "Butter croissant", "cat_breakfast", 230, 260, 240, 13, 0.70),
    ("spinach_pie", "Σπανακόπιτα", "Spinach pie", "cat_breakfast", 320, 360, 330, 13, 0.55),
    ("yogurt_honey", "Γιαούρτι με μέλι", "Greek yogurt with honey", "cat_breakfast", 420, 460, 430, 13, 0.40),
    ("club_sandwich", "Κλαμπ σάντουιτς", "Club sandwich", "cat_sandwich", 850, 920, 880, 13, 0.92),
    ("chicken_wrap", "Wrap κοτόπουλο", "Chicken wrap", "cat_sandwich", 780, 850, 800, 13, 0.82),
    ("tuna_baguette", "Μπαγκέτα με τόνο", "Tuna baguette", "cat_sandwich", 720, 790, 740, 13, 0.55),
    ("veggie_toast", "Τοστ λαχανικών", "Veggie toast", "cat_sandwich", 560, 620, 580, 13, 0.45),
    ("beef_burger", "Burger μοσχαρίσιο", "Beef burger", "cat_food", 980, 1080, 1020, 13, 0.72),
    ("chicken_burger", "Burger κοτόπουλο", "Chicken burger", "cat_food", 900, 990, 940, 13, 0.58),
    ("fries", "Πατάτες τηγανητές", "French fries", "cat_food", 350, 390, 360, 13, 0.85),
    ("greek_salad", "Χωριάτικη σαλάτα", "Greek salad", "cat_food", 780, 850, 800, 13, 0.48),
    ("caesar_salad", "Σαλάτα Caesar", "Caesar salad", "cat_food", 820, 890, 840, 13, 0.42),
    ("nuts", "Μείγμα ξηρών καρπών", "Mixed nuts", "cat_snack", 300, 340, 310, 24, 0.35),
    ("granola_bar", "Μπάρα δημητριακών", "Granola bar", "cat_snack", 250, 280, 260, 13, 0.38),
    ("cheesecake", "Τσίζκεϊκ", "Cheesecake", "cat_dessert", 520, 570, 540, 24, 0.65),
    ("chocolate_cake", "Σοκολατόπιτα", "Chocolate cake", "cat_dessert", 500, 550, 520, 24, 0.55),
    ("apple_pie", "Μηλόπιτα", "Apple pie", "cat_dessert", 450, 500, 470, 13, 0.38),
    ("ice_cream", "Παγωτό δύο μπάλες", "Two-scoop ice cream", "cat_dessert", 480, 530, 500, 13, 0.58),
]


def iso(value: datetime) -> str:
    return value.astimezone(BUSINESS_TIMEZONE).isoformat(timespec="milliseconds")


def person(business_id: str, identifier: str, first_name: str, last_name: str, role: str) -> dict[str, str]:
    return {"id": f"{business_id}_{identifier}", "first_name": first_name,
            "last_name": last_name, "role": role}


def category(business_id: str, definition: tuple[str, str, str], sequence: int, now: str) -> dict[str, Any]:
    identifier, name_gr, name_en = definition
    return {
        "id": f"{business_id}_{identifier}", "business_id": business_id,
        "name_gr": name_gr, "name_en": name_en, "internal_name": name_en.lower(),
        "names": [{"name": name_gr, "locale": "gr"}, {"name": name_en, "locale": "en"}],
        "print_order": sequence, "sequence": sequence, "product_sequence": None,
        "order_types": list(ORDER_TYPES), "created_at": now, "updated_at": now,
    }


def make_catalog(profile: dict[str, Any], start: datetime) -> list[dict[str, Any]]:
    business_id = profile["id"]
    now = iso(start)
    categories = {
        definition[0]: category(business_id, definition, index, now)
        for index, definition in enumerate(CATEGORY_DEFINITIONS, 1)
    }
    products = []
    for definition in PRODUCT_DEFINITIONS:
        identifier, name_gr, name_en, category_id, store, delivery, takeaway, vat, popularity = definition
        cat = categories[category_id]
        products.append({
            "id": f"{business_id}_prod_{identifier}", "business_id": business_id, "barcode_id": "",
            "name_gr": name_gr, "name_en": name_en, "internal_name": name_en.lower(),
            "names": [{"name": name_gr, "locale": "gr"}, {"name": name_en, "locale": "en"}],
            "description_gr": name_gr, "description_en": name_en, "visibility": {"menu": True},
            "category": cat, "price_store": store, "price_delivery": delivery, "price_takeaway": takeaway,
            "vat_rate": vat, "vat_rate_delivery": vat, "vat_rate_takeaway": vat, "taxes": [],
            "my_data_classification": {}, "published": True, "available": True,
            "order_types": list(ORDER_TYPES), "quantity_total": 1000, "quantity_portion": 1,
            "quantity_type": "pieces", "quantity_notify": 20, "enable_measurement": False,
            "optional_ingredients": False, "ingredients_ticket_show_selected": False,
            "created_at": now, "updated_at": now, "_popularity": popularity,
        })
    return products


def poisson(rng: random.Random, mean: float) -> int:
    threshold, product, draws = math.exp(-mean), 1.0, 0
    while product > threshold:
        product *= rng.random()
        draws += 1
    return max(0, draws - 1)


def category_name(product: dict[str, Any]) -> str:
    return product["category"]["id"].rsplit("_", 1)[-1]


def weighted_product(rng: random.Random, products: list[dict[str, Any]], profile: dict[str, Any],
                     order_type: str, hour: int, weekday: int, month: int) -> dict[str, Any]:
    peak = profile["peaks"][0 if hour < 11 else 1 if hour < 15 else 2 if hour < 18 else 3]
    weights = []
    for product in products:
        kind, weight = category_name(product), product["_popularity"] * peak
        if kind == "cat_coffee":
            weight *= 1.6 if hour < 11 else 0.75 if hour >= 18 else 1.0
        if kind in {"cat_breakfast", "cat_sandwich"}:
            weight *= 1.55 if hour < 15 else 0.68
        if kind == "cat_food":
            weight *= 1.5 if 12 <= hour < 15 else 0.8
        if kind == "cat_dessert":
            weight *= 1.4 if hour >= 15 else 0.7
        if kind == "cat_cold_beverage":
            weight *= 1.45 if month in {6, 7, 8, 9} else 0.82
        if weekday >= 5 and kind in {"cat_food", "cat_dessert", "cat_cold_beverage"}:
            weight *= 1.2
        if order_type == "delivery" and kind in {"cat_food", "cat_sandwich"}:
            weight *= 1.35
        if order_type == "takeaway" and kind in {"cat_coffee", "cat_breakfast"}:
            weight *= 1.2
        weights.append(weight)
    return rng.choices(products, weights=weights, k=1)[0]


def make_item(product: dict[str, Any], order_type: str, item_id: str, created: datetime,
              quantity: int, cashier: dict[str, str]) -> dict[str, Any]:
    price = {"dine_in": product["price_store"], "delivery": product["price_delivery"],
             "takeaway": product["price_takeaway"]}[order_type]
    vat_rate = product["vat_rate"]
    unit_vat = round(price * vat_rate / (100 + vat_rate))
    cat = product["category"]
    return {
        "id": item_id, "product_id": product["id"], "name_gr": product["name_gr"],
        "name_en": product["name_en"], "internal_name": product["internal_name"],
        "category": {key: cat[key] for key in ("id", "internal_name", "name_gr", "name_en")},
        "price": price, "price_with_extras": price, "price_base": price, "vat_rate": vat_rate,
        "quantity": quantity, "added_at": iso(created), "selections": [], "extras": [],
        "ingredients": [], "taxes": [], "net_price": price - unit_vat, "total_vat": unit_vat,
        "total_discount": 0, "total_amount": price, "discounts": [],
        "payment": [{"type": "card", "recipient": cashier, "paid_at": iso(created + timedelta(minutes=10)),
                     "amount": price * quantity}],
        "quantity_printed": quantity, "quantity_forms_printed": quantity, "quantity_invoiced": quantity,
        "my_data_classification": {"category": "category1_1", "type": "E3_561_003"},
        "invoice_classifications": [],
        "source": {"menu_id": f"{product['business_id']}_menu_main", "menu_name": "Βασικός κατάλογος"},
    }


def make_orders(products: list[dict[str, Any]], profile: dict[str, Any], start: datetime,
                days: int, target_per_day: int, rng: random.Random, order_offset: int) -> list[dict[str, Any]]:
    business_id = profile["id"]
    by_id = {product["id"]: product for product in products}
    coffee = [p for p in products if category_name(p) == "cat_coffee"]
    breakfast = [p for p in products if category_name(p) in {"cat_breakfast", "cat_sandwich"}]
    beverages = [p for p in products if category_name(p) in {"cat_coffee", "cat_cold_beverage"}]
    food = [p for p in products if category_name(p) in {"cat_food", "cat_sandwich"}]
    dessert = [p for p in products if category_name(p) == "cat_dessert"]
    cashier = person(business_id, "cashier_01", "Maria", "Dobby", "cashier")
    staff = [person(business_id, "waiter_01", "Nikos", "Dobby", "waiter"),
             person(business_id, "waiter_02", "Sofia", "Dobby", "waiter")]
    orders, item_number = [], 0
    for day_offset in range(days):
        day = start + timedelta(days=day_offset)
        month_factor = {1: .86, 2: .90, 3: 1.0, 4: 1.05, 5: 1.10, 6: 1.18,
                        7: 1.30, 8: 1.25, 9: 1.08, 10: .98, 11: .92, 12: 1.05}[day.month]
        trend = 1 + .12 * math.sin(day_offset / max(days, 1) * math.tau - .7)
        weekend = profile["weekend"] if day.weekday() >= 5 else 1.0
        spike = 1.6 if rng.random() < .025 else 1.0
        count = poisson(rng, target_per_day * profile["orders"] * profile["season"] *
                        month_factor * trend * weekend * spike)
        for _ in range(count):
            hour = rng.choices(range(7, 23), weights=[2, 5, 10, 14, 16, 13, 9, 8, 7, 6, 7, 8, 6, 4, 3, 2], k=1)[0]
            created = day.replace(hour=hour, minute=rng.randrange(60), second=rng.randrange(60),
                                  tzinfo=BUSINESS_TIMEZONE)
            order_type = rng.choices(ORDER_TYPES, weights=profile["types"], k=1)[0]
            selected = [weighted_product(rng, products, profile, order_type, hour, day.weekday(), day.month)]
            if rng.random() < (.30 if order_type == "delivery" else .45):
                selected.append(weighted_product(rng, products, profile, order_type, hour, day.weekday(), day.month))
            if rng.random() < .18:
                selected.append(weighted_product(rng, products, profile, order_type, hour, day.weekday(), day.month))
            anchor_kind = category_name(selected[0])
            if anchor_kind == "cat_coffee" and rng.random() < .48:
                selected.append(rng.choice([p for p in breakfast if p["id"] not in {x["id"] for x in selected}]))
            elif anchor_kind in {"cat_food", "cat_sandwich"} and rng.random() < .42:
                side = [p for p in products if p["id"].endswith(("_fries", "_soft_drink")) and
                        p["id"] not in {x["id"] for x in selected}]
                if side:
                    selected.append(rng.choice(side))
            elif anchor_kind == "cat_dessert" and rng.random() < .52:
                selected.append(rng.choice(coffee))
            elif anchor_kind in {"cat_food", "cat_sandwich"} and rng.random() < .18:
                selected.append(rng.choice(beverages))
            quantities = Counter(item["id"] for item in selected)
            order_number = order_offset + len(orders) + 1
            items = [make_item(by_id[product_id], order_type,
                               f"{business_id}_item_{item_number + index:06d}", created, quantity, cashier)
                     for index, (product_id, quantity) in enumerate(quantities.items())]
            item_number += len(items)
            total = sum(item["total_amount"] * item["quantity"] for item in items)
            total_vat = sum(item["total_vat"] * item["quantity"] for item in items)
            paid_at = created + timedelta(minutes=rng.randint(6, 24))
            waiter = rng.choice(staff)
            order = {
                "id": f"{business_id}_order_{order_number:06d}", "daily_id": str(order_number % 1000 or 1000),
                "short_id": f"{order_number:06d}", "business_id": business_id, "type": order_type,
                "status": "completed", "fulfillment": {"status": "picked_up", "preparing_at": iso(created),
                                                        "picked_up_at": iso(paid_at)},
                "items": items, "initiator": waiter, "assignee": waiter,
                "price_analysis": {
                    "total_vat": {"category_a": {"net_amount": total - total_vat, "vat_amount": total_vat},
                                  **{f"category_{letter}": {"net_amount": 0, "vat_amount": 0}
                                     for letter in "bcde"},
                                  "sum": {"net_amount": total - total_vat, "vat_amount": total_vat}},
                    "total_products": sum(item["quantity"] for item in items),
                    "total_amount_before_discount": total, "total_discount": 0,
                    "total_amount": total, "total_paid": total, "total_remaining": 0,
                },
                "provider": {} if order_type != "delivery" else {"name": "mock_delivery"},
                "created_at": iso(created), "updated_at": iso(paid_at),
            }
            if order_type == "dine_in":
                table_id = 1 + order_number % 12
                order["table"] = {"id": f"{business_id}_table_{table_id:02d}", "name": str(table_id),
                                  "position": {"id": f"{business_id}_position_main", "name": "Main room"},
                                  "party_size": rng.randint(1, 5)}
            if order_type == "delivery":
                customer_id = 1 + order_number % 180
                order["customer"] = {"id": f"{business_id}_customer_{customer_id:03d}",
                                     "name": f"Customer {customer_id:03d}",
                                     "phone_id": f"{business_id}_phone_{customer_id:03d}",
                                     "phone": f"+30210000{customer_id:04d}",
                                     "address_id": f"{business_id}_address_{customer_id:03d}",
                                     "street_name": "Example Street", "street_number": str(1 + customer_id % 90),
                                     "area": "Athina", "zip_code": "10563", "floor": str(1 + customer_id % 4),
                                     "doorbell": f"Customer {customer_id:03d}"}
                order["courier"] = {**person(business_id, "courier_01", "Giorgos", "Dobby", "delivery"),
                                    "assigned_at": iso(paid_at)}
            orders.append(order)
    return orders


def make_sales_stats(products: list[dict[str, Any]], orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"quantity": 0, "total_amount": 0})
    for order in orders:
        for item in order["items"]:
            totals[item["product_id"]]["quantity"] += item["quantity"]
            totals[item["product_id"]]["total_amount"] += item["total_amount"] * item["quantity"]
    stats = []
    for product in products:
        total, cat = totals[product["id"]], product["category"]
        stats.append({
            "id": product["id"], "name_gr": product["name_gr"], "name_en": product["name_en"],
            "internal_name": product["internal_name"],
            "category": {key: cat[key] for key in ("id", "internal_name", "name_gr", "name_en")},
            "price": product["price_store"], "quantity": total["quantity"], "total_amount": total["total_amount"],
            "discount_types": {}, "sources": [{"source": {"menu_id": f"{product['business_id']}_menu_main",
            "menu_name": "Βασικός κατάλογος"}, "quantity": total["quantity"], "total_amount": total["total_amount"]}],
        })
    return stats


def validate(products: list[dict[str, Any]], orders: list[dict[str, Any]],
             stats: list[dict[str, Any]], days: int) -> dict[str, Any]:
    errors: list[str] = []
    business_ids = {profile["id"] for profile in BUSINESSES}
    product_ids = {product["id"] for product in products}
    category_ids = {product["category"]["id"] for product in products}
    if len(business_ids) != len(BUSINESSES):
        errors.append("duplicate configured business IDs")
    if len(product_ids) != len(products):
        errors.append("duplicate product IDs")
    order_ids, item_ids = set(), set()
    quantities, category_quantities = Counter(), Counter()
    daily, hourly, monthly, weekdays = Counter(), Counter(), Counter(), Counter()
    per_business: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baskets: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for product in products:
        if product["business_id"] not in business_ids or product["id"] in product_ids - {product["id"]}:
            errors.append(f"invalid or duplicate product {product['id']}")
        if not isinstance(product["price_store"], int) or product["price_store"] <= 0:
            errors.append(f"invalid price {product['id']}")
        if product["category"]["id"] not in category_ids or product["category"]["business_id"] != product["business_id"]:
            errors.append(f"invalid category {product['id']}")
    for order in orders:
        oid, business_id = order["id"], order["business_id"]
        if oid in order_ids or business_id not in business_ids or order["type"] not in ORDER_TYPES:
            errors.append(f"invalid order identity/type {oid}")
        order_ids.add(oid)
        timestamp = datetime.fromisoformat(order["created_at"])
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            errors.append(f"ambiguous timestamp {oid}")
        date_key, hour_key = timestamp.date().isoformat(), str(timestamp.hour)
        daily[date_key] += 1; hourly[hour_key] += 1; monthly[date_key[:7]] += 1; weekdays[str(timestamp.weekday())] += 1
        per_business[business_id].append(order)
        calculated = 0
        basket = []
        for item in order["items"]:
            if item["id"] in item_ids or item["product_id"] not in product_ids:
                errors.append(f"invalid item reference {item['id']}")
            item_ids.add(item["id"]); product = next(p for p in products if p["id"] == item["product_id"])
            if product["business_id"] != business_id or item["category"]["id"] not in category_ids:
                errors.append(f"cross-business/category reference {item['id']}")
            if not isinstance(item["price"], int) or item["price"] <= 0 or item["quantity"] <= 0:
                errors.append(f"invalid quantity/price {item['id']}")
            expected_price = {"dine_in": product["price_store"], "delivery": product["price_delivery"],
                              "takeaway": product["price_takeaway"]}[order["type"]]
            if item["price"] != expected_price or item["price"] != item["total_amount"]:
                errors.append(f"price mismatch {item['id']}")
            expected_vat = round(item["price"] * item["vat_rate"] / (100 + item["vat_rate"]))
            if item["total_vat"] != expected_vat or item["net_price"] + item["total_vat"] != item["price"]:
                errors.append(f"VAT mismatch {item['id']}")
            calculated += item["price"] * item["quantity"]; quantities[item["product_id"]] += item["quantity"]
            category_quantities[item["category"]["id"]] += item["quantity"]; basket.append(item["product_id"])
        if calculated != order["price_analysis"]["total_amount"] or calculated != order["price_analysis"]["total_paid"]:
            errors.append(f"order total mismatch {oid}")
        baskets[business_id][tuple(sorted(basket))] += 1
    stats_by_id = {stat["id"]: stat for stat in stats}
    if len(stats_by_id) != len(stats) or len(stats_by_id) != len(products):
        errors.append("duplicate or incomplete sales statistics")
    for product in products:
        expected_amount = sum(item["total_amount"] * item["quantity"] for order in orders for item in order["items"]
                              if item["product_id"] == product["id"])
        stat = stats_by_id.get(product["id"])
        if not stat or (stat["quantity"], stat["total_amount"]) != (quantities[product["id"]], expected_amount):
            errors.append(f"sales stat mismatch {product['id']}")
    expected_dates = {(datetime(2026, 1, 1) + timedelta(days=index)).date().isoformat() for index in range(days)}
    missing_dates = sorted(expected_dates - set(daily))
    report_businesses = {}
    for business_id, business_orders in per_business.items():
        product_counts = Counter(item["product_id"] for order in business_orders for item in order["items"])
        business_dates = Counter(o["created_at"][:10] for o in business_orders)
        business_hours = Counter(datetime.fromisoformat(o["created_at"]).hour for o in business_orders)
        business_months = Counter(o["created_at"][:7] for o in business_orders)
        business_weekdays = Counter(datetime.fromisoformat(o["created_at"]).weekday() for o in business_orders)
        report_businesses[business_id] = {
            "orders": len(business_orders), "order_type_distribution": dict(Counter(o["type"] for o in business_orders)),
            "average_order_value_cents": round(sum(o["price_analysis"]["total_amount"] for o in business_orders) / len(business_orders), 2),
            "average_basket_size": round(sum(o["price_analysis"]["total_products"] for o in business_orders) / len(business_orders), 2),
            "monthly_order_distribution": dict(business_months),
            "daily_sales_distribution": dict(business_dates),
            "hourly_sales_distribution": dict(business_hours),
            "weekday_sales_distribution": dict(business_weekdays),
            "product_popularity": dict(product_counts), "category_popularity": dict(
                Counter(item["category"]["id"] for o in business_orders for item in o["items"])),
            "most_frequent_product_combinations": [
                {"products": list(key), "orders": count}
                for key, count in sorted(((k, v) for k, v in baskets[business_id].items() if len(k) > 1),
                                         key=lambda pair: pair[1], reverse=True)[:10]],
        }
    return {
        "businesses": len(business_ids), "products": len(products), "categories": len(category_ids),
        "orders": len(orders), "order_items": len(item_ids),
        "historical_date_range": [min(daily), max(daily)], "monthly_order_distribution": dict(monthly),
        "daily_sales_distribution": dict(daily), "hourly_sales_distribution": dict(hourly),
        "weekday_sales_distribution": dict(weekdays), "statistics_per_business": report_businesses,
        "product_sales_distribution": dict(quantities), "category_sales_distribution": dict(category_quantities),
        "percentage_of_products_with_zero_sales": round(100 * sum(not quantities[p["id"]] for p in products) / len(products), 2),
        "missing_dates": missing_dates, "missing_or_invalid_references": errors,
        "channel_supported": False, "channel_distribution": {},
    }


def generate(seed: int = DEFAULT_SEED, days: int = DEFAULT_DAYS,
             orders_per_day: int = DEFAULT_ORDERS_PER_DAY, output_dir: Path = OUT) -> dict[str, Any]:
    if days <= 0 or orders_per_day <= 0:
        raise ValueError("days and orders_per_day must be positive")
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, tzinfo=BUSINESS_TIMEZONE)
    products, orders = [], []
    order_offset = 0
    for profile in BUSINESSES:
        catalog = make_catalog(profile, start)
        products.extend([{key: value for key, value in product.items() if key != "_popularity"} for product in catalog])
        orders_for_business = make_orders(catalog, profile, start, days, orders_per_day, rng, order_offset)
        orders.extend(orders_for_business); order_offset += len(orders_for_business)
    orders.sort(key=lambda order: order["created_at"])
    stats = make_sales_stats(products, orders)
    report = validate(products, orders, stats, days)
    if report["missing_or_invalid_references"] or report["missing_dates"]:
        raise ValueError("generated data failed validation: " + "; ".join(
            report["missing_or_invalid_references"][:3] + [f"missing dates: {report['missing_dates'][:3]}"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in {"products.json": products, "orders.json": orders, "sales_stats.json": stats}.items():
        temporary = output_dir / f".{filename}.new"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_dir / filename)
    (output_dir / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--orders-per-day", type=int, default=DEFAULT_ORDERS_PER_DAY)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(generate(args.seed, args.days, args.orders_per_day, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
