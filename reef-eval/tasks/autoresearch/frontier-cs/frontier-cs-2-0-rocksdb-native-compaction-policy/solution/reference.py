diff --git a/db/compaction/compaction_picker.cc b/db/compaction/compaction_picker.cc
index b92a507..d885b70 100644
--- a/db/compaction/compaction_picker.cc
+++ b/db/compaction/compaction_picker.cc
@@ -551,9 +551,54 @@ bool CompactionPicker::SetupOtherInputs(
       assert(!expanded_output_level_inputs.empty());
       if (!AreFilesInCompaction(expanded_output_level_inputs.files) &&
           ExpandInputsToCleanCut(cf_name, vstorage,
-                                 &expanded_output_level_inputs) &&
-          expanded_output_level_inputs.size() == output_level_inputs->size()) {
-        expand_inputs = true;
+                                 &expanded_output_level_inputs)) {
+        expanded_inputs_size = TotalFileSize(expanded_inputs.files);
+        if (expanded_output_level_inputs.size() ==
+            output_level_inputs->size()) {
+          expand_inputs = true;
+        } else if (input_level + 2 == vstorage->num_levels()) {
+          CompactionInputFiles added_inputs;
+          added_inputs.level = input_level;
+          for (auto* file : expanded_inputs.files) {
+            if (std::find(inputs->files.begin(), inputs->files.end(), file) ==
+                inputs->files.end()) {
+              added_inputs.files.push_back(file);
+            }
+          }
+          if (!added_inputs.empty()) {
+            InternalKey added_start, added_limit;
+            GetRange(added_inputs, &added_start, &added_limit);
+            CompactionInputFiles added_output_inputs;
+            added_output_inputs.level = output_level;
+            vstorage->GetOverlappingInputs(output_level, &added_start,
+                                           &added_limit,
+                                           &added_output_inputs.files);
+            if (!added_output_inputs.empty() &&
+                !AreFilesInCompaction(added_output_inputs.files) &&
+                ExpandInputsToCleanCut(cf_name, vstorage,
+                                       &added_output_inputs)) {
+              const uint64_t added_inputs_size =
+                  TotalFileSize(added_inputs.files);
+              const uint64_t added_output_size =
+                  TotalFileSize(added_output_inputs.files);
+              const uint64_t expanded_output_size =
+                  TotalFileSize(expanded_output_level_inputs.files);
+              const long double separate_write_bytes =
+                  static_cast<long double>(inputs_size) +
+                  output_level_inputs_size + added_inputs_size +
+                  added_output_size;
+              const long double combined_write_bytes =
+                  static_cast<long double>(expanded_inputs_size) +
+                  expanded_output_size;
+              if (combined_write_bytes * 10.0L <= separate_write_bytes * 9.0L &&
+                  expanded_output_size <=
+                      MultiplyCheckOverflow(expanded_inputs_size, 2.0)) {
+                expand_inputs = true;
+                output_level_inputs->files = expanded_output_level_inputs.files;
+              }
+            }
+          }
+        }
       }
     }
     if (!expand_inputs) {
