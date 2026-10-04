@echo off
rem Turns the Snake screensaver off and deletes its files and saved progress.
reg delete "HKCU\Control Panel\Desktop" /v SCRNSAVE.EXE /f >nul 2>&1
taskkill /f /im SnakeSaver.scr >nul 2>&1
rmdir /s /q "%LOCALAPPDATA%\SnakeSaver" 2>nul
echo Snake screensaver removed.
