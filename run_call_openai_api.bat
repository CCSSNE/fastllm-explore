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
    echo Running default local API call: max_tokens=16 timeout=7200
    echo You can pass call_openai_api.py arguments to this bat file.
    echo.
    %PY_CMD% "%~dp0call_openai_api.py" --max-tokens 16 --timeout 7200
) else (
    %PY_CMD% "%~dp0call_openai_api.py" %*
)

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo call_openai_api.py exit code: %EXIT_CODE%
echo.
pause
exit /b %EXIT_CODE%
