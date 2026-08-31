diff --git a/src/diffusion/gaussian_diffusion.py b/src/diffusion/gaussian_diffusion.py
index f0c799a..3726b95 100644
--- a/src/diffusion/gaussian_diffusion.py
+++ b/src/diffusion/gaussian_diffusion.py
@@ -838,7 +838,7 @@ class GaussianDiffusion:
             # Backup for context restoration
             img_prev = img.clone()
             
-            with th.no_grad():
+            with th.no_grad(), th.autocast("cuda", dtype=th.bfloat16):  # [speedup-ref] bf16 sampling
                 out = self.dfot_ddim_sample(
                     model,
                     img,
