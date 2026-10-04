@echo off
rem Installs the Snake screensaver for the current user (no admin rights, no Python needed).
rem Copies the three files it needs to %LOCALAPPDATA%\SnakeSaver\app and points Windows at that copy.
rem Usage: install_screensaver.bat [idle-minutes]      (default 5)
setlocal
set "SRC=%~dp0"
set "APP=%LOCALAPPDATA%\SnakeSaver\app"
set "MIN=%~1"
if "%MIN%"=="" set "MIN=5"
set /a SECS=%MIN%*60

for %%F in ("screensaver\SnakeSaver.scr" "snake_classic.html" "replays_wins.js") do (
  if not exist "%SRC%%%~F" (echo Missing file: %SRC%%%~F & exit /b 1)
)
if not exist "%APP%" mkdir "%APP%"
copy /y "%SRC%screensaver\SnakeSaver.scr" "%APP%\" >nul
copy /y "%SRC%snake_classic.html" "%APP%\" >nul
copy /y "%SRC%replays_wins.js" "%APP%\" >nul

reg add "HKCU\Control Panel\Desktop" /v SCRNSAVE.EXE /t REG_SZ /d "%APP%\SnakeSaver.scr" /f >nul
reg add "HKCU\Control Panel\Desktop" /v ScreenSaveActive /t REG_SZ /d 1 /f >nul
reg add "HKCU\Control Panel\Desktop" /v ScreenSaveTimeOut /t REG_SZ /d %SECS% /f >nul
reg add "HKCU\Control Panel\Desktop" /v ScreenSaverIsSecure /t REG_SZ /d 0 /f >nul

echo Installed to %APP%
echo Screensaver starts after %MIN% idle minute(s). Change it in Settings ^> Personalization ^> Lock screen ^> Screen saver.
echo Preview now:  "%APP%\SnakeSaver.scr" /s
endlocal
