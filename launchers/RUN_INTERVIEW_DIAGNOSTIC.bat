@echo off
echo ================================================================================
echo INTERVIEW DIAGNOSTIC - AGENT 7 V2.1
echo ================================================================================
echo.
echo 10 Questions pour comprendre pourquoi l'agent n'ouvre pas de positions
echo.
echo ================================================================================

REM Use absolute paths to avoid space issues
set "BASE_DIR=C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"
set "PYTHONPATH=%PYTHONPATH%;C:\Users\lbye3\Desktop\GoldRL;C:\Users\lbye3\Desktop\GoldRL\src;C:\Users\lbye3\Desktop\GoldRL\AGENT_V2;%BASE_DIR%"

cd /d "C:\Users\lbye3\Desktop\GoldRL"

python "%BASE_DIR%\analysis\interview_agent7_diagnostic.py"

pause
