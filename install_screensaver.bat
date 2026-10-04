@echo off
rem Installs the Snake screensaver for the current user (no admin rights, no Python needed).
rem Copies the files it needs to %LOCALAPPDATA%\SnakeSaver\app and points Windows at that copy.
rem Usage: install_screensaver.bat [idle-minutes] [classic|cartoon]      (defaults: 5 classic)
rem   classic = black and green 2D retro monitor look (works offline)
rem   cartoon = 3D cartoon look (needs internet access to load the three.js library)
setlocal
set "SRC=%~dp0"
set "APP=%LOCALAPPDATA%\SnakeSaver\app"
set "MIN=%~1"
if "%MIN%"=="" set "MIN=5"
set "STYLE=%~2"
if "%STYLE%"=="" set "STYLE=classic"
set /a SECS=%MIN%*60
if /i "%STYLE%"=="cartoon" (set "SCR=SnakeCartoon.scr") else (set "SCR=SnakeSaver.scr")

for %%F in ("screensaver\SnakeSaver.scr" "screensaver\SnakeCartoon.scr" "snake_classic.html" "snake3d.html" "replays.js" "replays_wins.js") do (
  if not exist "%SRC%%%~F" (echo Missing file: %SRC%%%~F & exit /b 1)
)
if not exist "%APP%" mkdir "%APP%"
for %%F in ("screensaver\SnakeSaver.scr" "screensaver\SnakeCartoon.scr" "snake_classic.html" "snake3d.html" "replays.js" "replays_wins.js") do copy /y "%SRC%%%~F" "%APP%\" >nul

reg add "HKCU\Control Panel\Desktop" /v SCRNSAVE.EXE /t REG_SZ /d "%APP%\%SCR%" /f >nul
reg add "HKCU\Control Panel\Desktop" /v ScreenSaveActive /t REG_SZ /d 1 /f >nul
reg add "HKCU\Control Panel\Desktop" /v ScreenSaveTimeOut /t REG_SZ /d %SECS% /f >nul
reg add "HKCU\Control Panel\Desktop" /v ScreenSaverIsSecure /t REG_SZ /d 0 /f >nul

echo Installed (%STYLE%) to %APP%
echo Screensaver starts after %MIN% idle minute(s). Change it in Settings ^> Personalization ^> Lock screen ^> Screen saver
echo (both "SnakeSaver" and "SnakeCartoon" are listed there).
echo Preview now:  "%APP%\%SCR%" /s
endlocal
