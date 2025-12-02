@echo off
echo ================================================================================
echo SMOKE TEST MINI - AGENT 7 V2.1
echo ================================================================================
echo.
echo Ultra-fast test (~1 minute, 100 steps)
echo.
echo Verifies:
echo   [1] Model loads
echo   [2] 3 actions (SELL, HOLD, BUY)
echo   [3] Opens AND closes positions
echo   [4] No mode collapse
echo.
echo ================================================================================

REM Use absolute paths to avoid space issues
set "BASE_DIR=C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"
set "PYTHONPATH=%PYTHONPATH%;C:\Users\lbye3\Desktop\GoldRL;C:\Users\lbye3\Desktop\GoldRL\src;C:\Users\lbye3\Desktop\GoldRL\AGENT_V2;%BASE_DIR%"

cd /d "C:\Users\lbye3\Desktop\GoldRL"

python "%BASE_DIR%\tests\smoke_test_MINI.py"

pause
