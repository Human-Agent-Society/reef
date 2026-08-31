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
   "center_freq": 31.13,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 55.13,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 79.13,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 103.5,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 127.69,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 151.15,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 19.0,
   "bandwidth": 5.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 43.63,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 67.95,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 91.51,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 115.22,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 139.16,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  }
 ]
}
EOF
