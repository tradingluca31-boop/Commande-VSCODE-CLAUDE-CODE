@echo off
echo ================================================================================
echo FIND BLOCKER - Pourquoi les trades ne s'ouvrent pas?
echo ================================================================================
echo.
echo L'agent choisit BUY/SELL mais AUCUN trade ne s'ouvre!
echo Ce script trouve exactement CE QUI BLOQUE.
echo.
echo ================================================================================
echo.

set "BASE_DIR=C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7"
set "PYTHONPATH=%PYTHONPATH%;C:\Users\lbye3\Desktop\GoldRL;C:\Users\lbye3\Desktop\GoldRL\src;C:\Users\lbye3\Desktop\GoldRL\AGENT_V2;%BASE_DIR%"

cd /d "C:\Users\lbye3\Desktop\GoldRL"

python "%BASE_DIR%\analysis\find_blocker.py"

pause
