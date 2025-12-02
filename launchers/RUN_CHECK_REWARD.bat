@echo off
echo ================================================================================
echo CHECK REWARD FUNCTION - Agent 7 V2.1
echo ================================================================================
echo.
echo CHECKLIST DE DIAGNOSTIC:
echo   - HOLD donne-t-il un reward?
echo   - Les trades perdants sont-ils trop penalises?
echo   - Y a-t-il un reward uniquement a la cloture?
echo.
echo PIEGE: Si HOLD = safe et TRADE = risque, l'agent choisit HOLD!
echo.
echo ================================================================================
echo.

set "BASE_DIR=C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"
set "PYTHONPATH=%PYTHONPATH%;C:\Users\lbye3\Desktop\GoldRL;C:\Users\lbye3\Desktop\GoldRL\src;C:\Users\lbye3\Desktop\GoldRL\AGENT_V2;%BASE_DIR%"

cd /d "C:\Users\lbye3\Desktop\GoldRL"

python "%BASE_DIR%\analysis\check_reward_function.py"

pause
