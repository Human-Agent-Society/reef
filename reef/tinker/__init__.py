"""Tinker's SDK pieces that the Tinker training component and the Tinker serving runtime share.

``checkpoint`` is the manifest of a Tinker checkpoint (``TinkerCheckpoint``): the training component writes it, and
the serving runtime reads it to find the sampler of a release. ``service`` opens and closes SDK sessions on the
configured Tinker-compatible service. ``sampling`` renders and samples text chat on Tinker, and shapes the answer
with Reef's training capture. The training component (``reef.train.tinker_backend``) samples its rollouts with it,
and the serving runtime (``reef.inference.tinker``) serves requests with it. Neither side imports the other.
"""
