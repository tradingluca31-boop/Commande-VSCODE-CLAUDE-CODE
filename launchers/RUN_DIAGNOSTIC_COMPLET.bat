@echo off
echo ================================================================================
echo DIAGNOSTIC COMPLET - AGENT 7 V2.1
echo ================================================================================
echo.
echo 10 Questions pour comprendre pourquoi l'agent n'ouvre pas de positions:
echo.
echo   Q1. L'agent a-t-il PEUR de l'echec?
echo   Q2. Prefere-t-il HOLD par securite?
echo   Q3. Ferme-t-il correctement les positions?
echo   Q4. Comment reagit-il step par step?
echo   Q5. Quels BLOCAGES l'empechent d'ouvrir?
echo   Q6. Que voit-il dans les features?
echo   Q7. Son Critic distingue-t-il les bons/mauvais etats?
echo   Q8. Sa memoire LSTM fonctionne-t-elle?
echo   Q9. Le reward l'encourage-t-il a trader?
echo   Q10. SOLUTIONS pour le faire trader
echo.
echo Duree estimee: ~5 minutes
echo ================================================================================
echo.

REM Use absolute paths to avoid space issues
set "BASE_DIR=C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"
set "PYTHONPATH=%PYTHONPATH%;C:\Users\lbye3\Desktop\GoldRL;C:\Users\lbye3\Desktop\GoldRL\src;C:\Users\lbye3\Desktop\GoldRL\AGENT_V2;%BASE_DIR%"

cd /d "C:\Users\lbye3\Desktop\GoldRL"

python "%BASE_DIR%\analysis\diagnostic_complet_agent7.py"

echo.
echo ================================================================================
echo DIAGNOSTIC TERMINE
echo ================================================================================
echo.
pause
