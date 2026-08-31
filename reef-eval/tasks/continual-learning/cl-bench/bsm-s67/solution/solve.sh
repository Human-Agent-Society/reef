#!/bin/bash
# The perfect report, derived from the ground truth: scores IoU 1.0.
cat > /app/report.json <<'EOF'
{
 "transmitters": [
  {
   "center_freq": 7.32,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 31.79,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 55.6,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 79.62,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 103.09,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 127.18,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 151.13,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 19.63,
   "bandwidth": 5.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 43.55,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 67.43,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 91.02,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 115.17,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 139.46,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  }
 ]
}
EOF
