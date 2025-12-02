@echo off
echo ================================================================================
echo INTERVIEW INTERACTIF - AGENT 7 V2.1
echo ================================================================================
echo.
echo Pose des questions directement a l'agent!
echo.
echo Questions disponibles:
echo   [1] Quelle action vas-tu prendre?
echo   [2] Pourquoi cette decision?
echo   [3] As-tu peur de perdre?
echo   [4] Quelle est ta position actuelle?
echo   [5] Que retiens-tu de tes trades?
echo   [6] Montre-moi un step complet
echo   [7] Qu'est-ce qui t'empeche de trader?
echo   [8] Comment t'aider a trader plus?
echo.
echo ================================================================================
echo.

REM Use absolute paths to avoid space issues
set "BASE_DIR=C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"
set "PYTHONPATH=%PYTHONPATH%;C:\Users\lbye3\Desktop\GoldRL;C:\Users\lbye3\Desktop\GoldRL\src;C:\Users\lbye3\Desktop\GoldRL\AGENT_V2;%BASE_DIR%"

cd /d "C:\Users\lbye3\Desktop\GoldRL"

python "%BASE_DIR%\analysis\interview_agent7_interactif.py"

pause
