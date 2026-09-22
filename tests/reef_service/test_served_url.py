"""The address a recipe's own evaluation calls reach the service at."""

from __future__ import annotations

import pytest

from reef.service.assembly import _served_url


@pytest.mark.unit
@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", "http://127.0.0.1:8900"),
        ("::", "http://127.0.0.1:8900"),
        ("", "http://127.0.0.1:8900"),
        ("127.0.0.1", "http://127.0.0.1:8900"),
        ("10.0.0.5", "http://10.0.0.5:8900"),
        ("::1", "http://[::1]:8900"),
        ("fe80::1", "http://[fe80::1]:8900"),
        ("reef.internal", "http://reef.internal:8900"),
    ],
)
def test_the_served_url_names_an_address_the_service_reaches_itself_at(host: str, expected: str) -> None:
    assert _served_url(host, 8900) == expected
