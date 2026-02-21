# src/test_api.py
"""
Integration Tests for Reviews Sentiment API
=============================================
Run locally:
    pytest test_api.py -v

Run against container:
    BASE_URL=http://localhost:8000 API_KEY=your-key pytest test_api.py -v

Make sure the API server is running before executing tests.
"""

import os
import time
import pytest
import requests
from dotenv import load_dotenv

# ─── Configuration ──────────────────────────────────
load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY")

HEALTH_ENDPOINT = f"{BASE_URL}/health"
SENTIMENT_ENDPOINT = f"{BASE_URL}/api/v1/sentiment"

REQUEST_TIMEOUT = 10

# At least 3 product IDs that exist in your database
# Update these based on your actual data
TEST_PRODUCT_IDS = [1001, 1002, 1003]


# ═══════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def api_headers():
    """Return headers with the API key for authenticated endpoints."""
    if not API_KEY:
        pytest.fail(
            "❌ API_KEY not set. Add it to your .env file or pass via environment variable.\n"
            "   Example: API_KEY=sprints-secret-key-value pytest test_api.py -v"
        )
    return {"X-API-Key": API_KEY}


@pytest.fixture(scope="session", autouse=True)
def wait_for_api():
    """Wait for the API to become available before running tests."""
    max_retries = 15
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print(f"\n✅ API is ready at {BASE_URL}")
                return
        except requests.ConnectionError:
            pass

        if attempt < max_retries - 1:
            print(f"⏳ Waiting for API... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

    pytest.fail(
        f"❌ API at {BASE_URL} did not become available "
        f"after {max_retries * retry_delay}s. Is the server running?"
    )


# ═══════════════════════════════════════════════════════
#  Test Suite A: Health Check
# ═══════════════════════════════════════════════════════

class TestHealthCheck:
    """Tests for GET /health (no API key required)."""

    def test_health_returns_200(self):
        """A. Confirm the API is running via health check."""
        response = requests.get(HEALTH_ENDPOINT, timeout=REQUEST_TIMEOUT)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}"
        )

    def test_health_response_structure(self):
        """Validate the health check response has expected fields."""
        response = requests.get(HEALTH_ENDPOINT, timeout=REQUEST_TIMEOUT)
        data = response.json()

        assert "status" in data, "Missing 'status' field"
        assert "database" in data, "Missing 'database' field"

    def test_health_status_values(self):
        """Validate the health check reports OK status."""
        response = requests.get(HEALTH_ENDPOINT, timeout=REQUEST_TIMEOUT)
        data = response.json()

        assert data["status"] == "ok", f"Expected status 'ok', got '{data['status']}'"
        assert data["database"] == "connected", (
            f"Expected database 'connected', got '{data['database']}'"
        )


# ═══════════════════════════════════════════════════════
#  Test Suite B: Sentiment Endpoint — 3 Product IDs
# ═══════════════════════════════════════════════════════

class TestSentimentEndpoint:
    """Tests for GET /api/v1/sentiment/{product_id} (API key required)."""

    # ── Helper ──────────────────────────────────────
    def _validate_sentiment_response(self, data: dict, expected_product_id: int):
        """Validate response matches ProductSentimentSummary model."""

        # Check all required fields exist
        required_fields = [
            "product_id",
            "product_name",
            "latest_sentiment_score",
            "rolling_average_sentiment",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: '{field}'"

        # Validate types (mirrors Pydantic model)
        assert isinstance(data["product_id"], int), (
            f"product_id should be int, got {type(data['product_id']).__name__}"
        )
        assert isinstance(data["product_name"], str), (
            f"product_name should be str, got {type(data['product_name']).__name__}"
        )
        assert isinstance(data["latest_sentiment_score"], int), (
            f"latest_sentiment_score should be int, got {type(data['latest_sentiment_score']).__name__}"
        )
        assert isinstance(data["rolling_average_sentiment"], (int, float)), (
            f"rolling_average_sentiment should be numeric, got {type(data['rolling_average_sentiment']).__name__}"
        )

        # Validate product_id matches what was requested
        assert data["product_id"] == expected_product_id, (
            f"Expected product_id {expected_product_id}, got {data['product_id']}"
        )

        # Validate product_name is not empty
        assert len(data["product_name"].strip()) > 0, "product_name should not be empty"

    # ── Product 1 ───────────────────────────────────
    def test_sentiment_product_1(self, api_headers):
        """B. Query product ID 1 — validate structure and types."""
        product_id = TEST_PRODUCT_IDS[0]
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/{product_id}",
            headers=api_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200, (
            f"Expected 200 for product {product_id}, got {response.status_code}"
        )
        self._validate_sentiment_response(response.json(), product_id)

    # ── Product 2 ───────────────────────────────────
    def test_sentiment_product_2(self, api_headers):
        """B. Query product ID 2 — validate structure and types."""
        product_id = TEST_PRODUCT_IDS[1]
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/{product_id}",
            headers=api_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200, (
            f"Expected 200 for product {product_id}, got {response.status_code}"
        )
        self._validate_sentiment_response(response.json(), product_id)

    # ── Product 3 ───────────────────────────────────
    def test_sentiment_product_3(self, api_headers):
        """B. Query product ID 3 — validate structure and types."""
        product_id = TEST_PRODUCT_IDS[2]
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/{product_id}",
            headers=api_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200, (
            f"Expected 200 for product {product_id}, got {response.status_code}"
        )
        self._validate_sentiment_response(response.json(), product_id)

    # ── Rolling Average Specifically ────────────────
    def test_rolling_average_is_present_and_valid(self, api_headers):
        """Verify rolling_average_sentiment is a real calculated value."""
        product_id = TEST_PRODUCT_IDS[0]
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/{product_id}",
            headers=api_headers,
            timeout=REQUEST_TIMEOUT,
        )
        data = response.json()
        rolling_avg = data["rolling_average_sentiment"]

        assert rolling_avg is not None, "rolling_average_sentiment should not be None"
        assert isinstance(rolling_avg, float), (
            f"rolling_average_sentiment should be float, got {type(rolling_avg).__name__}"
        )


# ═══════════════════════════════════════════════════════
#  Test Suite C: Authentication & Error Handling
# ═══════════════════════════════════════════════════════

class TestAuthentication:
    """Tests for API key validation."""

    def test_missing_api_key_returns_403(self):
        """Request without X-API-Key header should fail."""
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/{TEST_PRODUCT_IDS[0]}",
            timeout=REQUEST_TIMEOUT,
            # No headers — no API key
        )
        assert response.status_code == 403, (
            f"Expected 403 without API key, got {response.status_code}"
        )

    def test_wrong_api_key_returns_403(self):
        """Request with an invalid API key should fail."""
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/{TEST_PRODUCT_IDS[0]}",
            headers={"X-API-Key": "totally-wrong-key-12345"},
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 403, (
            f"Expected 403 with wrong API key, got {response.status_code}"
        )

    def test_wrong_api_key_error_detail(self):
        """Verify the 403 response includes a meaningful message."""
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/{TEST_PRODUCT_IDS[0]}",
            headers={"X-API-Key": "wrong-key"},
            timeout=REQUEST_TIMEOUT,
        )
        data = response.json()
        assert "detail" in data, "403 response should include 'detail'"
        assert data["detail"] == "Invalid API Key"


class TestErrorHandling:
    """Tests for 404s and edge cases."""

    def test_nonexistent_product_returns_404(self, api_headers):
        """Querying a product that doesn't exist should return 404."""
        fake_id = 999999
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/{fake_id}",
            headers=api_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 404, (
            f"Expected 404 for fake product, got {response.status_code}"
        )

    def test_404_response_has_detail(self, api_headers):
        """Verify 404 includes a detail message."""
        fake_id = 999999
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/{fake_id}",
            headers=api_headers,
            timeout=REQUEST_TIMEOUT,
        )
        data = response.json()
        assert "detail" in data, "404 response should include 'detail'"

    def test_invalid_product_id_type_returns_422(self, api_headers):
        """Passing a string where int is expected should return 422."""
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/not-a-number",
            headers=api_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 422, (
            f"Expected 422 for invalid type, got {response.status_code}"
        )

    def test_response_time_under_threshold(self, api_headers):
        """API should respond within 2 seconds."""
        product_id = TEST_PRODUCT_IDS[0]
        start = time.time()
        response = requests.get(
            f"{SENTIMENT_ENDPOINT}/{product_id}",
            headers=api_headers,
            timeout=REQUEST_TIMEOUT,
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 2.0, f"Response took {elapsed:.2f}s (max allowed: 2.0s)"


# ═══════════════════════════════════════════════════════
#  Test Suite D: Swagger / OpenAPI Docs
# ═══════════════════════════════════════════════════════

class TestDocumentation:
    """Verify that auto-generated docs are accessible."""

    def test_swagger_ui_is_accessible(self):
        """Swagger UI should load at root (/)."""
        response = requests.get(f"{BASE_URL}/", timeout=REQUEST_TIMEOUT)
        assert response.status_code == 200, (
            f"Swagger UI not accessible, got {response.status_code}"
        )

    def test_openapi_json_is_accessible(self):
        """OpenAPI schema should be available at /openapi.json."""
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=REQUEST_TIMEOUT)
        assert response.status_code == 200

        schema = response.json()
        assert "paths" in schema, "OpenAPI schema missing 'paths'"
        assert "/health" in schema["paths"], "Missing /health in schema"
        assert "/api/v1/sentiment/{product_id}" in schema["paths"], (
            "Missing sentiment endpoint in schema"
        )


# ═══════════════════════════════════════════════════════
#  Run directly
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])