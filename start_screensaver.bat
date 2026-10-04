@echo off
rem Opens the snake screensaver full screen in Microsoft Edge (kiosk mode).
rem Close it with Alt+F4. Needs an internet connection for the 3D library (three.js).
set "PAGE=%~dp0snake3d.html"
start "" msedge --kiosk "file:///%PAGE:\=/%#saver" --edge-kiosk-type=fullscreen --no-first-run
