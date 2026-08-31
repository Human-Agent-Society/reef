diff --git a/bot.py b/bot.py
index d611e9b..cd85a3a 100644
--- a/bot.py
+++ b/bot.py
@@ -2,6 +2,7 @@ from generals.agents import ExpanderAgent
 
 
 class FrontierAgent(ExpanderAgent):
+    # Reference patch keeps the public baseline behavior.
     """Baseline expanding bot.
 
     Improve this class or replace it with your own implementation. The judge
