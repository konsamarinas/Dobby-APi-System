"""Validated loading of the documented Dobby fixture files."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ORDER_TYPES

logger = logging.getLogger(__name__)


def read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Required dataset file is missing: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load {path.name}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise RuntimeError(f"{path.name} must contain a JSON array of objects")
    return value


def load_datasets(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    products = read_json(data_dir / "products.json")
    orders = read_json(data_dir / "orders.json")
    stats = read_json(data_dir / "sales_stats.json")
    _validate(products, orders, stats)
    return products, orders, stats


def _validate(products: list[dict[str, Any]], orders: list[dict[str, Any]],
              stats: list[dict[str, Any]]) -> None:
    product_ids = [product.get("id") for product in products]
    order_ids = [order.get("id") for order in orders]
    if any(not isinstance(identifier, str) or not identifier for identifier in product_ids):
        raise RuntimeError("products.json contains a product with a missing ID")
    if any(not isinstance(identifier, str) or not identifier for identifier in order_ids):
        raise RuntimeError("orders.json contains an order with a missing ID")
    if len(product_ids) != len(set(product_ids)):
        raise RuntimeError("products.json contains duplicate product IDs")
    if len(order_ids) != len(set(order_ids)):
        raise RuntimeError("orders.json contains duplicate order IDs")
    known_products = set(product_ids)
    products_by_id = {product["id"]: product for product in products}
    businesses = {product.get("business_id") for product in products}
    for product in products:
        product_id = product["id"]
        business_id = product.get("business_id")
        if not isinstance(business_id, str) or not business_id:
            raise RuntimeError(f"Product {product_id} field 'business_id' must be a non-empty string")
        if business_id not in businesses:
            raise RuntimeError(f"Product {product_id} has an unknown business")
        category = product.get("category")
        if not isinstance(category, dict) or not category.get("id"):
            raise RuntimeError(f"Product {product_id} field 'category.id' is required")
        if category.get("business_id") != business_id:
            raise RuntimeError(f"Product {product_id} category belongs to another business")
        if not isinstance(product.get("available"), bool):
            raise RuntimeError(f"Product {product_id} field 'available' must be boolean")
        if not isinstance(product.get("published"), bool):
            raise RuntimeError(f"Product {product_id} field 'published' must be boolean")
        order_types = product.get("order_types")
        if (not isinstance(order_types, list) or not order_types
                or any(order_type not in ORDER_TYPES for order_type in order_types)
                or len(order_types) != len(set(order_types))):
            raise RuntimeError(
                f"Product {product_id} field 'order_types' must be a non-empty list of unique valid order types"
            )
        for field in ("price_store", "price_delivery", "price_takeaway"):
            price = product.get(field)
            if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price < 0:
                raise RuntimeError(f"Product {product_id} field '{field}' must be a non-negative number")
    stat_ids = [stat.get("id") for stat in stats]
    if any(not isinstance(identifier, str) or not identifier for identifier in stat_ids):
        raise RuntimeError("sales_stats.json contains a record with a missing ID")
    if len(stat_ids) != len(set(stat_ids)):
        raise RuntimeError("sales_stats.json contains duplicate IDs")
    for stat in stats:
        product = products_by_id.get(stat["id"])
        if product is None:
            raise RuntimeError(f"Sales statistic {stat['id']} references an unknown product")
        stat_category = stat.get("category") or {}
        if stat_category.get("id") and stat_category["id"] != product.get("category", {}).get("id"):
            raise RuntimeError(f"Sales statistic {stat['id']} category does not match its product")
    item_ids: set[str] = set()
    for order in orders:
        order_id = order.get("id")
        for field in ("id", "business_id", "type", "created_at", "items"):
            if field not in order:
                raise RuntimeError(f"Order {order_id} is missing required field '{field}'")
        if order.get("business_id") not in businesses:
            raise RuntimeError(f"Order {order.get('id')} has an unknown business")
        if order.get("type") not in ORDER_TYPES:
            raise RuntimeError(f"Invalid order type in order {order.get('id')}")
        try:
            timestamp = datetime.fromisoformat(str(order["created_at"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid timestamp in order {order.get('id')}") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise RuntimeError(f"Order {order.get('id')} timestamp is timezone-naive")
        try:
            quantity_items = order["items"]
        except KeyError as exc:
            raise RuntimeError(f"Order {order.get('id')} has no items") from exc
        if not isinstance(quantity_items, list):
            raise RuntimeError(f"Order {order.get('id')} items must be a list")
        for item in quantity_items:
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise RuntimeError(f"Order {order_id} item is missing required field 'id'")
            if item_id in item_ids:
                raise RuntimeError(f"orders.json contains duplicate item ID {item.get('id')}")
            item_ids.add(item_id)
            if item.get("product_id") not in known_products:
                raise RuntimeError(f"Unknown product {item.get('product_id')} in order {order.get('id')}")
            product = products_by_id[item["product_id"]]
            if product.get("business_id") != order.get("business_id"):
                raise RuntimeError(
                    f"Order {order.get('id')} item {item.get('id')} references a product from another business"
                )
            item_category = (item.get("category") or {}).get("id")
            product_category = (product.get("category") or {}).get("id")
            if item_category and item_category != product_category:
                raise RuntimeError(
                    f"Order {order.get('id')} item {item.get('id')} category does not match product {item['product_id']}"
                )
            if not isinstance(item.get("quantity"), int) or item["quantity"] <= 0:
                raise RuntimeError(f"Invalid quantity in item {item.get('id')}")
