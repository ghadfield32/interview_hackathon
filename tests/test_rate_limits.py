#!/usr/bin/env python3
"""
Test script for rate limiting functionality.
Run this to verify that rate limits are working correctly.
"""

import asyncio
import httpx
import time
import os
from typing import Optional

class RateLimitTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=10.0)
        self.token: Optional[str] = None

    async def login(self) -> bool:
        """Login to get a JWT token."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/token",
                data={"username": "alice", "password": "supersecretvalue"}
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                print("✅ Login successful")
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False

    async def test_endpoint(self, endpoint: str, payload: dict, name: str, expected_limit: int):
        """Test rate limiting on a specific endpoint."""
        print(f"\n🔍 Testing {name} endpoint: {endpoint}")
        print(f"Expected limit: {expected_limit} requests per window")

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        success_count = 0
        rate_limited_count = 0

        for i in range(expected_limit + 5):  # Try a few extra requests
            try:
                response = await self.client.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    headers=headers
                )

                # Check rate limit headers
                remaining = response.headers.get("X-RateLimit-Remaining")
                limit = response.headers.get("X-RateLimit-Limit")

                if response.status_code == 200:
                    success_count += 1
                    print(f"  ✅ Request {i+1}: Success (Remaining: {remaining}/{limit})")
                elif response.status_code == 429:
                    rate_limited_count += 1
                    retry_after = response.headers.get("Retry-After", "unknown")
                    print(f"  🚫 Request {i+1}: Rate limited (Retry-After: {retry_after}s)")
                    break
                else:
                    print(f"  ❌ Request {i+1}: Error {response.status_code}")
                    break

            except Exception as e:
                print(f"  ❌ Request {i+1}: Exception {e}")
                break

        print(f"📊 Results: {success_count} successful, {rate_limited_count} rate limited")
        return success_count, rate_limited_count

    async def test_login_rate_limit(self):
        """Test login rate limiting."""
        print("\n🔍 Testing login rate limiting")

        rate_limited_count = 0
        for i in range(10):  # Try more than the limit
            try:
                response = await self.client.post(
                    f"{self.base_url}/api/v1/token",
                    data={"username": "alice", "password": "wrongpassword"}
                )

                if response.status_code == 401:
                    print(f"  ✅ Login attempt {i+1}: Expected 401 (invalid credentials)")
                elif response.status_code == 429:
                    rate_limited_count += 1
                    retry_after = response.headers.get("Retry-After", "unknown")
                    print(f"  🚫 Login attempt {i+1}: Rate limited (Retry-After: {retry_after}s)")
                    break
                else:
                    print(f"  ❌ Login attempt {i+1}: Unexpected {response.status_code}")
                    break

            except Exception as e:
                print(f"  ❌ Login attempt {i+1}: Exception {e}")
                break

        print(f"📊 Login rate limit results: {rate_limited_count} rate limited")
        return rate_limited_count > 0

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

async def main():
    """Run all rate limiting tests."""
    print("🚀 Starting rate limiting tests...")

    tester = RateLimitTester()

    try:
        # Test login rate limiting first
        login_rate_limited = await tester.test_login_rate_limit()

        # Login to get token for authenticated endpoints
        if not await tester.login():
            print("❌ Cannot proceed without login")
            return

        # Test iris prediction (light limit)
        iris_success, iris_rate_limited = await tester.test_endpoint(
            "/api/v1/iris/predict",
            {
                "model_type": "rf",
                "samples": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]
            },
            "Iris Prediction",
            120  # Should be 2x default limit
        )

        # Test cancer prediction (heavy limit)
        cancer_success, cancer_rate_limited = await tester.test_endpoint(
            "/api/v1/cancer/predict",
            {
                "model_type": "bayes",
                "samples": [{"mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8, "mean_area": 1001, "mean_smoothness": 0.1184, "mean_compactness": 0.2776, "mean_concavity": 0.3001, "mean_concave_points": 0.1471, "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871, "se_radius": 1.095, "se_texture": 0.9053, "se_perimeter": 8.589, "se_area": 153.4, "se_smoothness": 0.006399, "se_compactness": 0.04904, "se_concavity": 0.05373, "se_concave_points": 0.01587, "se_symmetry": 0.03003, "se_fractal_dimension": 0.006193, "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019, "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119, "worst_concave_points": 0.2654, "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189}]
            },
            "Cancer Prediction",
            30  # Should be cancer limit
        )

        # Test training endpoints (very limited)
        training_success, training_rate_limited = await tester.test_endpoint(
            "/api/v1/iris/train",
            {},
            "Iris Training",
            2  # Should be training limit
        )

        # Summary
        print("\n📋 Test Summary:")
        print(f"  Login rate limiting: {'✅ Working' if login_rate_limited else '❌ Not working'}")
        print(f"  Iris prediction rate limiting: {'✅ Working' if iris_rate_limited else '❌ Not working'}")
        print(f"  Cancer prediction rate limiting: {'✅ Working' if cancer_rate_limited else '❌ Not working'}")
        print(f"  Training rate limiting: {'✅ Working' if training_rate_limited else '❌ Not working'}")

        all_working = login_rate_limited and iris_rate_limited and cancer_rate_limited and training_rate_limited
        print(f"\n🎯 Overall: {'✅ All rate limits working' if all_working else '❌ Some rate limits not working'}")

    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main()) 
