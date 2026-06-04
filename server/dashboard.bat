@echo off
REM Double-click this file (or run it) to launch the Minesweeper AI web dashboard.
REM It uses the project's virtual env and opens the browser when the server is ready.
cd /d "%~dp0"
echo Starting Minesweeper AI dashboard...
".venv\Scripts\python.exe" -m trainer.dashboard --tag beginner %*
echo.
echo Dashboard stopped. Press any key to close this window.
pause >nul
