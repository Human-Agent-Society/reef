"""SGLang inference launch, capture and engine control, independent of training backends.

GPU libraries are imported only inside their owning native adapter. Training
backends supply configuration and borrowed placement; they do not own engines.
"""
