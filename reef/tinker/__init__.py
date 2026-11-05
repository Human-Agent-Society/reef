"""Tinker's SDK pieces that the Tinker training component and the Tinker serving runtime share.

``checkpoint`` names the manifest of a Tinker artifact and reads the sampler it names (``TinkerArtifact``): the training
component writes the whole manifest (``reef.train.tinker_backend.checkpoint``), and the serving runtime reads it to
find the sampler of a snapshot. ``service`` opens and closes SDK sessions on the
configured Tinker-compatible service. ``sampling`` renders and samples text chat on Tinker, and shapes the answer
with Reef's training capture. The training component (``reef.train.tinker_backend``) samples its rollouts with it,
and the serving runtime (``reef.inference.tinker``) serves requests with it. Neither side imports the other.
"""
