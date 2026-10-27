@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
if not exist "outputs\launch" mkdir "outputs\launch"
set "WATCH_UV="
for /f "delims=" %%U in ('where.exe uv.exe 2^>nul') do if not defined WATCH_UV set "WATCH_UV=%%U"
if not defined WATCH_UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "WATCH_UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined WATCH_UV goto no_uv
echo NPU live subtitles
echo.
set "WATCH_PLAYER=mpv"
choice /c 12 /n /m "Player: [1] mpv  [2] MPC-HC: "
if errorlevel 3 goto cancelled
if not errorlevel 1 goto cancelled
if errorlevel 2 set "WATCH_PLAYER=mpc-hc"
set "WATCH_ASR=moonshine"
set "WATCH_LANGUAGE=en"
choice /c 12 /n /m "Audio language: [1] English  [2] Japanese: "
if errorlevel 3 goto cancelled
if not errorlevel 1 goto cancelled
if errorlevel 2 (
    set "WATCH_ASR=granite"
    set "WATCH_LANGUAGE=ja"
)
if "%~1"=="" goto choose
"%WATCH_UV%" run --locked watch.py --player "%WATCH_PLAYER%" --asr "%WATCH_ASR%" --language "%WATCH_LANGUAGE%" -- "%~1" 2>>"outputs\launch\uv-startup.log"
goto finished
:choose
"%WATCH_UV%" run --locked watch.py --player "%WATCH_PLAYER%" --asr "%WATCH_ASR%" --language "%WATCH_LANGUAGE%" 2>>"outputs\launch\uv-startup.log"
:finished
set "WATCH_RESULT=%ERRORLEVEL%"
if "%WATCH_RESULT%"=="0" goto done
if "%WATCH_RESULT%"=="130" goto done
if "%WATCH_RESULT%"=="3" goto done
goto uv_error
:cancelled
set "WATCH_RESULT=0"
goto done
:no_uv
set "WATCH_RESULT=1"
echo uv.exe was not found in PATH or the user .local\bin directory.>>"outputs\launch\uv-startup.log"
:uv_error
type "outputs\launch\uv-startup.log"
powershell.exe -NoProfile -NonInteractive -Command "Add-Type -AssemblyName PresentationFramework; [void][System.Windows.MessageBox]::Show('The uv environment could not start. See outputs\launch\uv-startup.log. Install uv if it was not found.','NPU live captions')"
:done
popd
exit /b %WATCH_RESULT%
