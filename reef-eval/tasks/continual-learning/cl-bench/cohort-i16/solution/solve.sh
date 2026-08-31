#!/bin/bash
# Population truth as the report: scores exactly the per-instance
# ceiling, 0.09868 bits/cohort (task.toml: oracle_score).
cat > /app/report.json <<'EOF'
{
 "estimates": [
  {
   "cohort_id": "voc_3_6",
   "estimated_survival_12m": 0.651914,
   "estimated_survival_24m": 0.317242,
   "estimated_survival_36m": 0.12032
  },
  {
   "cohort_id": "acron_lt2",
   "estimated_survival_12m": 0.610621,
   "estimated_survival_24m": 0.268785,
   "estimated_survival_36m": 0.089277
  },
  {
   "cohort_id": "family_history_yes",
   "estimated_survival_12m": 0.722241,
   "estimated_survival_24m": 0.401429,
   "estimated_survival_36m": 0.187041
  },
  {
   "cohort_id": "hhbp_gte8",
   "estimated_survival_12m": 0.509227,
   "estimated_survival_24m": 0.250566,
   "estimated_survival_36m": 0.109859
  },
  {
   "cohort_id": "avoc_high_prb1_lt25",
   "estimated_survival_12m": 0.72526,
   "estimated_survival_24m": 0.498387,
   "estimated_survival_36m": 0.317575
  },
  {
   "cohort_id": "avoc_high_education_gte16",
   "estimated_survival_12m": 0.767928,
   "estimated_survival_24m": 0.397977,
   "estimated_survival_36m": 0.157325
  },
  {
   "cohort_id": "avoc_gte15_gg1",
   "estimated_survival_12m": 0.788332,
   "estimated_survival_24m": 0.445292,
   "estimated_survival_36m": 0.223256
  },
  {
   "cohort_id": "family_history_gg3",
   "estimated_survival_12m": 0.652386,
   "estimated_survival_24m": 0.359387,
   "estimated_survival_36m": 0.163281
  },
  {
   "cohort_id": "family_history_prb1_gte28",
   "estimated_survival_12m": 0.757061,
   "estimated_survival_24m": 0.391511,
   "estimated_survival_36m": 0.154454
  },
  {
   "cohort_id": "education_gte16_cognitive_lt050",
   "estimated_survival_12m": 0.424119,
   "estimated_survival_24m": 0.24153,
   "estimated_survival_36m": 0.160836
  },
  {
   "cohort_id": "education_gte16_gg3",
   "estimated_survival_12m": 0.745005,
   "estimated_survival_24m": 0.424754,
   "estimated_survival_36m": 0.191681
  },
  {
   "cohort_id": "education_lt13_gg1",
   "estimated_survival_12m": 0.848356,
   "estimated_survival_24m": 0.644132,
   "estimated_survival_36m": 0.446374
  },
  {
   "cohort_id": "acron_lt2_prb1_24_28",
   "estimated_survival_12m": 0.50889,
   "estimated_survival_24m": 0.219278,
   "estimated_survival_36m": 0.079586
  },
  {
   "cohort_id": "prb1_gte28_acron_gte4",
   "estimated_survival_12m": 0.799999,
   "estimated_survival_24m": 0.509918,
   "estimated_survival_36m": 0.268161
  },
  {
   "cohort_id": "voc_3_6_prb1_lt25",
   "estimated_survival_12m": 0.45503,
   "estimated_survival_24m": 0.207537,
   "estimated_survival_36m": 0.087518
  },
  {
   "cohort_id": "gg3_voc_lt3",
   "estimated_survival_12m": 0.767047,
   "estimated_survival_24m": 0.469862,
   "estimated_survival_36m": 0.23692
  },
  {
   "cohort_id": "age_gte55_acron_gte4",
   "estimated_survival_12m": 0.773588,
   "estimated_survival_24m": 0.467712,
   "estimated_survival_36m": 0.224198
  },
  {
   "cohort_id": "cognitive_lt050_age_lt55",
   "estimated_survival_12m": 0.570735,
   "estimated_survival_24m": 0.413006,
   "estimated_survival_36m": 0.300691
  },
  {
   "cohort_id": "hhbp_gte8_prb1_lt25",
   "estimated_survival_12m": 0.322916,
   "estimated_survival_24m": 0.128395,
   "estimated_survival_36m": 0.061342
  },
  {
   "cohort_id": "age_40_60_voc_gte5",
   "estimated_survival_12m": 0.401318,
   "estimated_survival_24m": 0.129698,
   "estimated_survival_36m": 0.030788
  },
  {
   "cohort_id": "avoc_high_family_history_prb1_gte28",
   "estimated_survival_12m": 0.763881,
   "estimated_survival_24m": 0.377901,
   "estimated_survival_36m": 0.137905
  },
  {
   "cohort_id": "education_high_voc_high_acron_low",
   "estimated_survival_12m": 0.660571,
   "estimated_survival_24m": 0.263049,
   "estimated_survival_36m": 0.062903
  },
  {
   "cohort_id": "family_history_gg3_voc_gte5",
   "estimated_survival_12m": 0.360519,
   "estimated_survival_24m": 0.129567,
   "estimated_survival_36m": 0.046532
  },
  {
   "cohort_id": "avoc_high_cognitive_low_family_history",
   "estimated_survival_12m": 0.558169,
   "estimated_survival_24m": 0.380909,
   "estimated_survival_36m": 0.266783
  },
  {
   "cohort_id": "education_low_prb1_gte28_gg3",
   "estimated_survival_12m": 0.463468,
   "estimated_survival_24m": 0.21502,
   "estimated_survival_36m": 0.092966
  },
  {
   "cohort_id": "acron_high_hhbp_high_prb1_lt25",
   "estimated_survival_12m": 0.487321,
   "estimated_survival_24m": 0.267666,
   "estimated_survival_36m": 0.14294
  },
  {
   "cohort_id": "family_history_education_gte16_prb1_lt25",
   "estimated_survival_12m": 0.751334,
   "estimated_survival_24m": 0.436123,
   "estimated_survival_36m": 0.205592
  },
  {
   "cohort_id": "avoc_low_voc_high_age_40_60",
   "estimated_survival_12m": 0.34272,
   "estimated_survival_24m": 0.100878,
   "estimated_survival_36m": 0.024846
  },
  {
   "cohort_id": "age_gte60_gg3_education_gte16",
   "estimated_survival_12m": 0.798305,
   "estimated_survival_24m": 0.475095,
   "estimated_survival_36m": 0.220789
  },
  {
   "cohort_id": "prb1_lt25_voc_gte5_acron_lt2",
   "estimated_survival_12m": 0.238582,
   "estimated_survival_24m": 0.048246,
   "estimated_survival_36m": 0.012876
  },
  {
   "cohort_id": "gg1_family_history_cognitive_lt050",
   "estimated_survival_12m": 0.801991,
   "estimated_survival_24m": 0.625943,
   "estimated_survival_36m": 0.448804
  },
  {
   "cohort_id": "prb1_gte30_acron_gte4_voc_lt3",
   "estimated_survival_12m": 0.828688,
   "estimated_survival_24m": 0.524643,
   "estimated_survival_36m": 0.267884
  },
  {
   "cohort_id": "avoc_gte15_gg1_age_gte55",
   "estimated_survival_12m": 0.746634,
   "estimated_survival_24m": 0.341011,
   "estimated_survival_36m": 0.107363
  },
  {
   "cohort_id": "cognitive_gte075_age_gte60_gg3",
   "estimated_survival_12m": 0.81738,
   "estimated_survival_24m": 0.494768,
   "estimated_survival_36m": 0.232987
  },
  {
   "cohort_id": "cognitive_lt050_prb1_gte28_acron_lt2",
   "estimated_survival_12m": 0.277984,
   "estimated_survival_24m": 0.068908,
   "estimated_survival_36m": 0.019287
  },
  {
   "cohort_id": "gg3_education_gte16_voc_3_6",
   "estimated_survival_12m": 0.753382,
   "estimated_survival_24m": 0.424751,
   "estimated_survival_36m": 0.18806
  }
 ]
}
EOF
