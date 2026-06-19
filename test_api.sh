#!/usr/bin/env bash
# Usage: ./test_api.sh [url]
#   url defaults to http://localhost:8000
#   Example: ./test_api.sh https://my-api.com

BASE_URL="${1:-http://localhost:8000}"

echo "=== Testing API at: $BASE_URL ==="

echo ""
echo ">>> HEALTH CHECK <<<"
curl -s "$BASE_URL/health" | python3 -m json.tool

echo ""
echo ">>> MODEL INFO <<<"
curl -s "$BASE_URL/model/info" | python3 -m json.tool

echo ""
echo ">>> MODEL HISTORY <<<"
curl -s "$BASE_URL/model/history?limit=3" | python3 -m json.tool

# --- Normal transaction (Class=0 from creditcard.csv) ---
echo ""
echo ">>> NORMAL TRANSACTION (POST /predict) <<<"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "V1": -1.359807134, "V2": -0.072781173, "V3": 2.536346738, "V4": 1.378155224,
      "V5": -0.33832077, "V6": 0.462387778, "V7": 0.239598554, "V8": 0.098697901,
      "V9": 0.36378697, "V10": 0.090794172, "V11": -0.551599533, "V12": -0.617800856,
      "V13": -0.991389847, "V14": -0.311169354, "V15": 1.468176972, "V16": -0.470400525,
      "V17": 0.207971242, "V18": 0.02579058, "V19": 0.40399296, "V20": 0.251412098,
      "V21": -0.018306778, "V22": 0.277837576, "V23": -0.11047391, "V24": 0.066928075,
      "V25": 0.128539358, "V26": -0.189114844, "V27": 0.133558377, "V28": -0.021053053,
      "Amount": 149.62, "Time": 0.0
    }
  }' | python3 -m json.tool

# --- Fraud transaction (Class=1 from creditcard.csv) ---
echo ""
echo ">>> FRAUD TRANSACTION (POST /predict) <<<"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "V1": -2.312226542, "V2": 1.951992011, "V3": -1.609850732, "V4": 3.997905588,
      "V5": -0.522187865, "V6": -1.426545319, "V7": -2.537387306, "V8": 1.391657248,
      "V9": -2.770089277, "V10": -2.772272145, "V11": 3.202033207, "V12": -2.899907388,
      "V13": -0.595221881, "V14": -4.289253782, "V15": 0.38972412, "V16": -1.14074718,
      "V17": -2.830055675, "V18": -0.016822468, "V19": 0.416955705, "V20": 0.126910559,
      "V21": 0.517232371, "V22": -0.035049369, "V23": -0.465211076, "V24": 0.320198199,
      "V25": 0.044519167, "V26": 0.177839798, "V27": 0.261145003, "V28": -0.143275875,
      "Amount": 0.0, "Time": 406.0
    }
  }' | python3 -m json.tool

echo ""
echo "=== Done ==="
