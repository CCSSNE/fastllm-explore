@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PY_CMD=python"
where python >nul 2>nul
if errorlevel 1 (
    set "PY_CMD=py -3"
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python was not found in PATH.
        echo Please run this from a terminal that can find python, or install the Python launcher.
        echo.
        pause
        exit /b 1
    )
)

if "%~1"=="" (
    %PY_CMD% "%~dp0call_openai_api.py" --timeout 7200
) else (
    %PY_CMD% "%~dp0call_openai_api.py" %*
)

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Exit code: %EXIT_CODE%
echo.
pause
exit /b %EXIT_CODE%
