@echo off
setlocal enabledelayedexpansion

:: Reproduction Batch Script for MMF-EMSNet Scenarios
:: This script trains and evaluates MMF-EMSNet models under the three specific configurations
:: retrieved from HPO best results.

:: Configuration Variables (Edit as needed)
set PYTHON_PATH=C:\Users\awan2\miniconda3\envs\gizviz\python
set EPOCHS=100
set NUM_SAMPLES_PER_CLASS=5
set DATASET=Dataset/NPZ/dataset_16.npz

echo =======================================================================
echo Starting Reproduction Run for MMF-EMSNet Scenarios
echo Python Executable: %PYTHON_PATH%
echo Epochs:            %EPOCHS%
echo Dataset:           %DATASET%
echo =======================================================================

:: -------------------------------------------------------------------------
:: Scenario 1: Binary (Class 0 vs 4)
:: Parameters: Residual=True, mcmaf, dsm_only, RMSprop, lr=0.001, bs=64
:: -------------------------------------------------------------------------
echo.
echo [1/3] Running Scenario 1 Reproduction...
echo Parameters: Residual=True, mcmaf, dsm_only, RMSprop, lr=0.001, bs=64
%PYTHON_PATH% run_custom_train_infer.py ^
    --dataset "%DATASET%" ^
    --scenario 1 ^
    --residual True ^
    --dsm-mode dsm_only ^
    --optimizer RMSprop ^
    --lr 0.001 ^
    --batch-size 64 ^
    --epochs %EPOCHS% ^
    --num-samples-per-class %NUM_SAMPLES_PER_CLASS% ^
    --output-dir results_reproduced/scenario_1

if %errorlevel% neq 0 (
    echo [ERROR] Scenario 1 reproduction failed. Exiting...
    exit /b %errorlevel%
)

:: -------------------------------------------------------------------------
:: Scenario 2: Binary (Class 0-3 vs 4)
:: Parameters: Residual=False, mcmaf, dsm_uncertainty, RMSprop, lr=0.0001, bs=256
:: -------------------------------------------------------------------------
echo.
echo [2/3] Running Scenario 2 Reproduction...
echo Parameters: Residual=False, mcmaf, dsm_uncertainty, RMSprop, lr=0.0001, bs=256
%PYTHON_PATH% run_custom_train_infer.py ^
    --dataset "%DATASET%" ^
    --scenario 2 ^
    --residual False ^
    --dsm-mode dsm_uncertainty ^
    --optimizer RMSprop ^
    --lr 0.0001 ^
    --batch-size 256 ^
    --epochs %EPOCHS% ^
    --num-samples-per-class %NUM_SAMPLES_PER_CLASS% ^
    --output-dir results_reproduced/scenario_2

if %errorlevel% neq 0 (
    echo [ERROR] Scenario 2 reproduction failed. Exiting...
    exit /b %errorlevel%
)

:: -------------------------------------------------------------------------
:: Scenario 3: Multiclass (All 5 classes)
:: Parameters: Residual=True, mcmaf, dsm_uncertainty, Nadam, lr=0.001, bs=128
:: -------------------------------------------------------------------------
echo.
echo [3/3] Running Scenario 3 Reproduction...
echo Parameters: Residual=True, mcmaf, dsm_uncertainty, Nadam, lr=0.001, bs=128
%PYTHON_PATH% run_custom_train_infer.py ^
    --dataset "%DATASET%" ^
    --scenario 3 ^
    --residual True ^
    --dsm-mode dsm_uncertainty ^
    --optimizer Nadam ^
    --lr 0.001 ^
    --batch-size 128 ^
    --epochs %EPOCHS% ^
    --num-samples-per-class %NUM_SAMPLES_PER_CLASS% ^
    --output-dir results_reproduced/scenario_3

if %errorlevel% neq 0 (
    echo [ERROR] Scenario 3 reproduction failed. Exiting...
    exit /b %errorlevel%
)

echo.
echo =======================================================================
echo Reproduction Runs Completed Successfully.
echo Outputs saved in: results_reproduced/
echo =======================================================================
pause
