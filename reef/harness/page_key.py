"""The key a harness page link carries in place of the service token.

A request's page and a step's page are links a person opens in a browser,
which sends no Authorization header, so the link carries a credential in its
query. The service token there would reach whatever reads the link: the
model of a session that files the request and prints the link, the
provider behind it, Reef's own records. The page key is an HMAC of the
scenario keyed by the token's digest: it opens those two pages of that one
scenario and nothing else, and the token cannot be read back from it. The
service keeps only token digests, which is why the digest is the HMAC key.
"""

from __future__ import annotations

import hashlib
import hmac

#: What the HMAC covers before the scenario name, so the key is good for nothing but a page link.
PAGE_KEY_CONTEXT = b"reef-page\n"


def page_key_for_digest(digest: bytes, scenario: str) -> str:
    """The page key of ``scenario`` for the token whose sha256 is ``digest``, as hex."""
    return hmac.new(digest, PAGE_KEY_CONTEXT + scenario.encode("utf-8"), hashlib.sha256).hexdigest()


def page_key(token: str, scenario: str) -> str:
    """The page key of ``scenario`` for ``token``: HMAC sha256 keyed by sha256(token) over the context and the
    scenario, as hex."""
    return page_key_for_digest(hashlib.sha256(token.encode("utf-8")).digest(), scenario)


__all__ = ["PAGE_KEY_CONTEXT", "page_key", "page_key_for_digest"]
