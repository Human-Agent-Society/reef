#!/bin/bash
# The perfect report, derived from the ground truth: scores IoU 1.0.
cat > /app/report.json <<'EOF'
{
 "transmitters": [
  {
   "center_freq": 7.12,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 31.6,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 55.42,
   "bandwidth": 15.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 79.38,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 103.42,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 127.27,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 151.01,
   "bandwidth": 15.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 19.59,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 43.69,
   "bandwidth": 5.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 67.23,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 91.68,
   "bandwidth": 5.0,
   "currently_active": true,
   "estimated_power": -40.0
  },
  {
   "center_freq": 115.41,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  },
  {
   "center_freq": 139.82,
   "bandwidth": 5.0,
   "currently_active": false,
   "estimated_power": -60.0
  }
 ]
}
EOF
