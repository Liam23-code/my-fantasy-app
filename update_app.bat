@echo off
REM ---------------------------------------------------------------------------
REM update_app.bat -- stage everything, commit, and push so Render redeploys.
REM
REM Usage:  update_app.bat "your commit message"
REM ---------------------------------------------------------------------------
setlocal

if "%~1"=="" (
    echo ERROR: a commit message is required.
    echo.
    echo   Usage:  update_app.bat "your commit message"
    exit /b 1
)

cd /d "%~dp0"

echo [1/3] Staging changes...
git add .
if errorlevel 1 exit /b 1

REM `git commit` exits non-zero when there is nothing staged. That is a normal
REM "already up to date" case, not a failure, so fall through to the push
REM instead of aborting -- a previous run may have committed but failed to push.
echo [2/3] Committing...
git commit -m "%~1"
if errorlevel 1 echo    (nothing new to commit -- pushing anyway)

echo [3/3] Pushing...
git push
if errorlevel 1 (
    echo.
    echo ERROR: push failed. Resolve the problem above and re-run.
    exit /b 1
)

echo.
echo Done. Render will pick up the new commit and redeploy.
endlocal
