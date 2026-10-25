"""Tinker's SDK pieces that the Tinker training component and the Tinker serving runtime share.

``sampling`` renders and samples text chat on Tinker, and shapes the answer with Reef's training capture.
The training component (``reef.train.tinker_backend``) samples its rollouts with it, and the serving runtime
(``reef.inference.tinker``) serves requests with it. Neither side imports the other.
"""
