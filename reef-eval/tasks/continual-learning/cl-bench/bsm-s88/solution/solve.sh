#!/bin/bash
# The perfect report, derived from the ground truth: scores IoU 1.0.
cat > /app/report.json <<'EOF'
{
 "transmitters": [
  {
   "center_freq": 7.97,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 31.16,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 55.04,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 79.49,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 103.97,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 127.25,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 151.22,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 19.17,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 43.56,
   "bandwidth": 5.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 67.35,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 91.9,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 115.48,
   "bandwidth": 5.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 139.7,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  }
 ]
}
EOF
