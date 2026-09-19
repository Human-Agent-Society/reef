"""SGLang inference launch, capture and engine control, independent of training backends.

GPU libraries are imported only inside their owning native adapter. Deployment
assembly supplies configuration and borrowed placement; training backends do
not own engines. This integration depends on Reef runtime contracts and never
imports Slime or another training implementation.
"""
