#!/bin/bash
set -e

echo "========================================================"
echo "1. Testing Package Import"
echo "========================================================"
python -c "import ds_msp; print('DS-MSP package loaded successfully')"

echo ""
echo "========================================================"
echo "2. Running Calibration + Reprojection Check (tests/verify_calibration.py)"
echo "========================================================"
python tests/verify_calibration.py

echo ""
echo "========================================================"
echo "3. Running Visualization (visualize.py)"
echo "========================================================"
python visualize.py fov
python visualize.py undistort

echo ""
echo "========================================================"
echo "4. Running Unit Tests (tests/test_ds_camera_cv.py)"
echo "========================================================"
python tests/test_ds_camera_cv.py

echo ""
echo "========================================================"
echo "5. Running K-Inverse & 3D Reconstruction Verification"
echo "========================================================"
python tests/verify_k_inverse.py
python tests/verify_3d_reconstruction.py

echo ""
echo "========================================================"
echo "ALL CHECKS PASSED!"
echo "========================================================"
