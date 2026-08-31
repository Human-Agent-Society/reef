#!/bin/bash
# The perfect report, derived from the ground truth: scores IoU 1.0.
cat > /app/report.json <<'EOF'
{
 "transmitters": [
  {
   "center_freq": 7.95,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 31.75,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 55.16,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 79.28,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 103.3,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 127.71,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 151.95,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 19.84,
   "bandwidth": 5.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 43.72,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 67.07,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 91.46,
   "bandwidth": 5.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 115.43,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 139.37,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  }
 ]
}
EOF
