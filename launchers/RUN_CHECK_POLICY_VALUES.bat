@echo off
echo ================================================================================
echo CHECK POLICY VALUES - Agent 7 V2.1
echo ================================================================================
echo.
echo Regarde ce que l'agent PENSE vraiment:
echo   - P(SELL), P(HOLD), P(BUY) = Probabilites d'action
echo   - V(s) = Value Function du Critic
echo.
echo Si P(HOLD) ^>^> P(BUY) et P(SELL) = L'agent a appris que HOLD est "safe"
echo Si V(s) est plat = Le Critic ne distingue pas les etats
echo.
echo ================================================================================
echo.

set "BASE_DIR=C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"
set "PYTHONPATH=%PYTHONPATH%;C:\Users\lbye3\Desktop\GoldRL;C:\Users\lbye3\Desktop\GoldRL\src;C:\Users\lbye3\Desktop\GoldRL\AGENT_V2;%BASE_DIR%"

cd /d "C:\Users\lbye3\Desktop\GoldRL"

python "%BASE_DIR%\analysis\check_policy_values.py"

pause
