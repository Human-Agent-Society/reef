Published TTT-Discover TriMul seed. It normalizes the pair representation in a
Triton kernel, computes the projection and gating paths, performs the outgoing
triangle contraction with mixed-precision matrix multiplication, and fuses the
post-contraction normalization, output gate, and final projection. It preserves
float32 output and supports masked and unmasked benchmark shapes.
