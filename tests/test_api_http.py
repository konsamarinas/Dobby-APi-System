import json
import unittest
from urllib.parse import urlsplit

from src.main import PRODUCTS_BY_BUSINESS, app, clear_metrics_cache, metrics


async def asgi_request(method: str, path: str, body: dict | None = None,
                       headers: list[tuple[bytes, bytes]] | None = None) -> tuple[int, dict, dict]:
    sent = []
    parsed_path = urlsplit(path)
    payload = json.dumps(body).encode() if body is not None else b""
    request_headers = headers or []
    if body is not None:
        request_headers = request_headers + [(b"content-type", b"application/json")]
    messages = [{"type": "http.request", "body": payload, "more_body": False}]

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    await app(
        {"type": "http", "method": method, "path": parsed_path.path,
         "query_string": parsed_path.query.encode(),
         "headers": request_headers, "scheme": "http", "server": ("testserver", 80),
         "client": ("testclient", 123), "root_path": ""},
        receive,
        send,
    )
    response_start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    try:
        decoded_body = json.loads(response_body or b"{}")
    except json.JSONDecodeError:
        decoded_body = response_body.decode()
    return response_start["status"], dict(response_start["headers"]), decoded_body


class HttpApiTests(unittest.IsolatedAsyncioTestCase):
    business_id = next(iter(PRODUCTS_BY_BUSINESS))
    product_id = next(iter(PRODUCTS_BY_BUSINESS[business_id]))

    async def test_health(self):
        status, _, body = await asgi_request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    async def test_recommendations(self):
        status, _, body = await asgi_request(
            "POST", "/recommendations",
            {"business_id": self.business_id, "order_type": "dine_in",
             "timestamp": "2026-10-31T09:20:00+02:00", "cart": [], "limit": 1},
        )
        self.assertEqual(status, 200)
        self.assertIn("recommendations", body)

    async def test_forecast(self):
        status, _, body = await asgi_request(
            "GET", f"/forecast?business_id={self.business_id}&level=product&id={self.product_id}&horizon=1",
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["forecast"]), 1)

    async def test_metrics_and_unknown_business(self):
        clear_metrics_cache()
        status, _, body = await asgi_request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn("recommendations", body)

        status, _, body = await asgi_request("GET", "/metrics?business_id=unknown")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")

    def test_metrics_cache_matches_after_invalidation_and_isolated_businesses(self):
        clear_metrics_cache()
        global_result = metrics()
        business_result = metrics(self.business_id)
        self.assertNotEqual(global_result["recommendations"]["n_baskets"],
                            business_result["recommendations"]["n_baskets"])
        global_result["recommendations"]["n_baskets"] = -1
        self.assertNotEqual(metrics()["recommendations"]["n_baskets"], -1)
        cached_global = metrics()
        clear_metrics_cache()
        uncached_global = metrics()
        self.assertEqual(cached_global, uncached_global)

    async def test_invalid_horizon(self):
        status, _, body = await asgi_request(
            "GET", f"/forecast?business_id={self.business_id}&level=product&id={self.product_id}&horizon=15",
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "unprocessable")

    async def test_invalid_request_body(self):
        status, _, body = await asgi_request("POST", "/recommendations", {"business_id": self.business_id})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_request")

    async def test_cors_preflight(self):
        status, headers, _ = await asgi_request(
            "OPTIONS", "/health",
            headers=[(b"origin", b"http://localhost:3000"),
                     (b"access-control-request-method", b"GET")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get(b"access-control-allow-origin"), b"http://localhost:3000")

    async def test_openapi_response_models(self):
        status, _, schema = await asgi_request("GET", "/openapi.json")
        self.assertEqual(status, 200)
        components = schema["components"]["schemas"]
        self.assertIn("HealthResponse", components)
        self.assertIn("MetricsResponse", components)
        self.assertEqual(
            schema["paths"]["/metrics"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/MetricsResponse",
        )


if __name__ == "__main__":
    unittest.main()
