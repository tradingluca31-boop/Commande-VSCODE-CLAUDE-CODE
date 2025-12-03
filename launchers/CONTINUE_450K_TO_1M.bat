@echo off
echo ================================================================================
echo CONTINUE TRAINING FROM 450K to 1M STEPS
echo ================================================================================
echo.
echo [INFO] Loading checkpoint 450K (Win Rate: 70.05%%, ROI: 12.72%%)
echo [INFO] Training 450K -^> 1M (550K additional steps, ~3h20min)
echo [INFO] Checkpoints every 50K (450K, 500K, 550K, ..., 1M)
echo.
echo ================================================================================

cd /d "%~dp0\..\training"
python continue_from_450k_to_1M.py

echo.
echo ================================================================================
pause