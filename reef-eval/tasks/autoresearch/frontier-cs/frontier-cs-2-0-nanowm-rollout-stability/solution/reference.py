diff --git a/src/diffusion/df_sample.py b/src/diffusion/df_sample.py
index 43a298a..9f23e70 100644
--- a/src/diffusion/df_sample.py
+++ b/src/diffusion/df_sample.py
@@ -287,6 +287,7 @@ def dfot_sample(
             truncate_at = clean_rows[0].item() + 1
             scheduling_matrix = scheduling_matrix[:truncate_at]
 
+    history_stabilization_level = 0.20  # [stability-ref] stronger history stabilization reduces long-horizon drift
     if history_stabilization_level > 0.0 and n_context_frames > 0 and context is not None:
         assert 0.0 < history_stabilization_level < 1.0
         t_stab = int(round(history_stabilization_level * (diffusion.num_timesteps - 1)))
