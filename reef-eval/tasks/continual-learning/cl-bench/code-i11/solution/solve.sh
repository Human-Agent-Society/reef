#!/bin/bash
# The gold PR patch: makes FAIL_TO_PASS pass, so the oracle scores 1.0.
cd /testbed
echo ZGlmZiAtLWdpdCBhL3RlbmFjaXR5L19faW5pdF9fLnB5IGIvdGVuYWNpdHkvX19pbml0X18ucHkKLS0tIGEvdGVuYWNpdHkvX19pbml0X18ucHkKKysrIGIvdGVuYWNpdHkvX19pbml0X18ucHkKQEAgLTM1NCw2ICszNTQsNyBAQCBkZWYgYmVnaW4oc2VsZikgLT4gTm9uZToKICAgICAgICAgc2VsZi5zdGF0aXN0aWNzWyJzdGFydF90aW1lIl0gPSB0aW1lLm1vbm90b25pYygpCiAgICAgICAgIHNlbGYuc3RhdGlzdGljc1siYXR0ZW1wdF9udW1iZXIiXSA9IDEKICAgICAgICAgc2VsZi5zdGF0aXN0aWNzWyJpZGxlX2ZvciJdID0gMAorICAgICAgICBzZWxmLnN0YXRpc3RpY3NbImRlbGF5X3NpbmNlX2ZpcnN0X2F0dGVtcHQiXSA9IDAKIAogICAgIGRlZiBfYWRkX2FjdGlvbl9mdW5jKHNlbGYsIGZuOiB0LkNhbGxhYmxlWy4uLiwgdC5BbnldKSAtPiBOb25lOgogICAgICAgICBzZWxmLml0ZXJfc3RhdGUuYWN0aW9ucy5hcHBlbmQoZm4pCg== | base64 -d > /tmp/gold.patch
git apply --verbose /tmp/gold.patch \
  || git apply --verbose --reject /tmp/gold.patch \
  || patch --batch --fuzz=5 -p1 -i /tmp/gold.patch
