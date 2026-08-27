@echo off
REM Conductor kohya discovery job — double-click this, or run it from any shell.
REM Works from PowerShell, cmd.exe, Git Bash and Explorer; no venv activation,
REM no path juggling, no PowerShell-vs-bash quoting differences.
REM
REM Pass --dry-run to build the job without submitting, or --job <jid> to fetch
REM results for a job that was already submitted.

setlocal
set "HERE=%~dp0"
"%HERE%.venv\Scripts\python.exe" "%HERE%run_discovery.py" %*
set "RC=%ERRORLEVEL%"

REM Keep the window open when launched by double-click, so the output is readable.
echo.
echo (exit code %RC%)
if "%~1"=="" pause
exit /b %RC%
