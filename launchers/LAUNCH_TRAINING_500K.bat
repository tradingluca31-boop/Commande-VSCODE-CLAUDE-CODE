@echo off
echo ================================================================================
echo AGENT 7 V2.1 CRITIC BOOST + LSTM - TRAINING 500K STEPS
echo ================================================================================
echo.
echo Training Configuration:
echo   Algorithm:     RecurrentPPO + LSTM (256 neurons, 16 steps)
echo   Features:      229 (209 base + 20 RL+MEMORY)
echo   vf_coef:       1.0 (MAXIMUM Critic learning)
echo   n_epochs:      25
echo   Architecture:  Separate Actor[256,256] / Critic[256,256]
echo.
echo   Total Steps:   500,000
echo   Duration:      ~2 hours
echo   Checkpoints:   Every 50K steps
echo.
echo Expected Performance:
echo   Sharpe:        2.5+
echo   Max DD:        ^<6%%
echo   ROI:           18-22%%
echo   Critic Std:    ^>1.0 (HEALTHY)
echo.
echo ================================================================================
echo.

REM Change to project root
cd /d "C:\Users\lbye3\Desktop\GoldRL"

REM Set Python path to include necessary directories
set PYTHONPATH=%PYTHONPATH%;C:\Users\lbye3\Desktop\GoldRL;C:\Users\lbye3\Desktop\GoldRL\src;C:\Users\lbye3\Desktop\GoldRL\AGENT_V2

REM Run training from the organized structure
echo [START] Launching training...
echo.
python "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\FICHIER IMPORTANT AGENT 7\training\train_CRITIC_BOOST_LSTM.py"

echo.
echo ================================================================================
echo TRAINING COMPLETE
echo ================================================================================
echo.
echo Check results in:
echo   - Models: C:\Users\lbye3\Desktop\GoldRL\AGENT_V2\AGENT 7 V2\models\
echo   - Logs:   C:\Users\lbye3\Desktop\GoldRL\output\logs\agent7_critic_boost_lstm\
echo.
echo Next steps:
echo   1. Check TensorBoard: tensorboard --logdir C:\Users\lbye3\Desktop\GoldRL\output\logs\agent7_critic_boost_lstm
echo   2. Run smoke test: RUN_SMOKE_TEST_MINI.bat
echo   3. Run SHAP analysis: python explain_shap_agent7.py
echo.
pause
