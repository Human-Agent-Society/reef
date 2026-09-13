"""Optional Tinker integration: remote LoRA training and immutable sampling.

Deployment/config discovery does not import the SDK. ``client`` owns the SDK
boundary, ``losses`` shapes exact token data, and ``runtime`` owns publication.
"""
