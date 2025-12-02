@echo off
echo ================================================================================
echo L'AGENT 7 PARLE - Interview Automatique
echo ================================================================================
echo.
echo L'agent repond AUTOMATIQUEMENT a toutes les questions.
echo Pas besoin d'interaction - il s'explique tout seul!
echo.
echo ================================================================================
echo.

set "BASE_DIR=C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"
set "PYTHONPATH=%PYTHONPATH%;C:\Users\lbye3\Desktop\GoldRL;C:\Users\lbye3\Desktop\GoldRL\src;C:\Users\lbye3\Desktop\GoldRL\AGENT_V2;%BASE_DIR%"

cd /d "C:\Users\lbye3\Desktop\GoldRL"

python "%BASE_DIR%\analysis\agent7_parle.py"

pause
