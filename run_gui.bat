@echo off
rem Запуск Pain Axis Workbench с интерпретатором, в котором есть torch и transformers.
rem Порядок поиска: переменная PAIN_AXIS_PYTHON, py -3.10, python из PATH, типовые пути.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY=%PAIN_AXIS_PYTHON%"
if defined PY if exist "%PY%" goto :run

for /f "delims=" %%i in ('py -3.10 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%i"
if defined PY if exist "%PY%" goto :run

for /f "delims=" %%i in ('python -c "import torch,sys;print(sys.executable)" 2^>nul') do set "PY=%%i"
if defined PY if exist "%PY%" goto :run

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if defined PY if exist "%PY%" goto :run
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if defined PY if exist "%PY%" goto :run
if exist "C:\Program Files\Python310\python.exe" set "PY=C:\Program Files\Python310\python.exe"
if defined PY if exist "%PY%" goto :run

echo Не найден Python с torch. Укажите интерпретатор вручную:
echo    set PAIN_AXIS_PYTHON=C:\путь\к\python.exe
echo    run_gui.bat
echo.
echo Для одного лишь демо-стенда (без torch) подойдёт любой Python 3.10+:
echo    python run_gui.py
pause
exit /b 1

:run
echo Интерпретатор: %PY%
"%PY%" run_gui.py %*
if errorlevel 1 pause
endlocal
