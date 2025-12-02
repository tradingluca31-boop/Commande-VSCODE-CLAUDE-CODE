@echo off
echo ================================================================================
echo QUICK TEST 10K - AGENT 7 V2.1 (VERIFY TRADING)
echo ================================================================================
echo.
echo Quick test (~5 minutes, 10K steps) to verify:
echo   [1] Agent opens positions
echo   [2] Agent closes positions
echo   [3] Actions are balanced (not 100%% BUY/SELL/HOLD)
echo   [4] Trades are executed
echo.
echo ================================================================================

REM Use absolute paths to avoid space issues
set "BASE_DIR=C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"
set "PYTHONPATH=%PYTHONPATH%;C:\Users\lbye3\Desktop\GoldRL;C:\Users\lbye3\Desktop\GoldRL\src;C:\Users\lbye3\Desktop\GoldRL\AGENT_V2;%BASE_DIR%"

cd /d "C:\Users\lbye3\Desktop\GoldRL"

python "%BASE_DIR%\tests\quick_test_10k.py"

pause
