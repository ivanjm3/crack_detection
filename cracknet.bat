@echo off
setlocal EnableDelayedExpansion
REM ============================================================================
REM  cracknet.bat - drive the Jetson from this Windows machine.
REM
REM  Double-click for a menu, or run with a command:
REM
REM     cracknet              menu
REM     cracknet live         start the 3-camera live preview (~10 fps)
REM     cracknet scan         start the 3-camera survey scan (~6.5 s/position)
REM     cracknet stop         stop whatever is running
REM     cracknet status       what is running, and is it healthy
REM     cracknet logs         last 40 lines of the server log
REM     cracknet check        preflight only - changes nothing
REM     cracknet identify     label each camera so rig.json can be written
REM     cracknet open         open the dashboard in a browser
REM     cracknet shell "cmd"  run one command on the Jetson
REM
REM  Link management (no USB cable needed):
REM
REM     cracknet join         join the Jetson's Wi-Fi AP - one time per PC
REM     cracknet link         which route is live, is SSH up
REM     cracknet leave        drop the AP, go back to the usual Wi-Fi
REM     cracknet netmode ap   tell the Jetson to host the AP (default)
REM     cracknet netmode wifi tell the Jetson to rejoin the campus Wi-Fi
REM
REM  No SSH session to open and nothing to type on the Jetson. Commands go over
REM  the network through tools\jssh.py, which uses paramiko and reads the user
REM  and password from tools\.jetson.env (gitignored, never committed).
REM
REM  JETSON_HOSTS in that file lists the routes to try in order - the Jetson's
REM  own AP (10.42.0.1) first, the USB link (192.168.55.1) as a fallback. The
REM  first one that answers on port 22 is used, so unplugging USB changes
REM  nothing here.
REM ============================================================================

cd /d "%~dp0"

set "PY=python"
set "JSSH=tools\jssh.py"
set "ENVFILE=tools\.jetson.env"
set "REMOTE=/home/sarah/cracknet"

REM ---- preflight: fail with a fixable message, not a stack trace -------------
where %PY% >nul 2>&1 || (
    echo [X] Python is not on PATH. Install it, or edit PY= at the top of this file.
    goto :end
)
if not exist "%JSSH%" (
    echo [X] %JSSH% not found. Run this from the project folder.
    goto :end
)
if not exist "%ENVFILE%" (
    echo [X] %ENVFILE% not found. Create it with:
    echo.
    echo        JETSON_HOST=192.168.55.1
    echo        JETSON_USER=sarah
    echo        JETSON_PASS=your-password
    echo.
    goto :end
)

REM Ask which route is actually live, rather than trusting one address: the
REM board answers on its own AP or over USB depending on how it is plugged in,
REM and the dashboard URL has to match whichever it is.
set "HOST="
REM stderr is left visible on purpose: when the Wi-Fi link has dropped this
REM prints "reconnecting to CrackNet", which explains the pause.
for /f "usebackq delims=" %%h in (`%PY% tools\jlink.py host`) do set "HOST=%%h"
if "%HOST%"=="" set "HOST=10.42.0.1"

set "CMD=%~1"
if "%CMD%"=="" goto :menu
goto :dispatch

REM ---------------------------------------------------------------- menu ----
:menu
echo.
echo   CrackNet - Jetson at %HOST%
echo   ---------------------------------------------
echo     1  Start LIVE preview    3 cameras, ~10 fps
echo     2  Start SCAN survey     3 cameras, 1080p
echo     3  Stop
echo     4  Status
echo     5  Logs
echo     6  Preflight check
echo     7  Identify cameras
echo     8  Open dashboard
echo     ---------------------------------------------
echo     9  Join the Jetson Wi-Fi    (no USB needed)
echo     L  Link status
echo     0  Exit
echo.
set "choice="
set /p "choice=  Choose: "
if "%choice%"=="1" set "CMD=live"
if "%choice%"=="2" set "CMD=scan"
if "%choice%"=="3" set "CMD=stop"
if "%choice%"=="4" set "CMD=status"
if "%choice%"=="5" set "CMD=logs"
if "%choice%"=="6" set "CMD=check"
if "%choice%"=="7" set "CMD=identify"
if "%choice%"=="8" set "CMD=open"
if "%choice%"=="9" set "CMD=join"
if /i "%choice%"=="L" set "CMD=link"
if "%choice%"=="0" goto :end
if "%CMD%"=="" (
    echo   Not a choice.
    goto :menu
)

REM ------------------------------------------------------------ dispatch ----
:dispatch
if /i "%CMD%"=="live"     goto :start_live
if /i "%CMD%"=="scan"     goto :start_scan
if /i "%CMD%"=="stop"     goto :stop
if /i "%CMD%"=="status"   goto :status
if /i "%CMD%"=="logs"     goto :logs
if /i "%CMD%"=="check"    goto :check
if /i "%CMD%"=="identify" goto :identify
if /i "%CMD%"=="open"     goto :open
if /i "%CMD%"=="shell"    goto :shell
if /i "%CMD%"=="join"     goto :join
if /i "%CMD%"=="leave"    goto :leave
if /i "%CMD%"=="link"     goto :link
if /i "%CMD%"=="netmode"  goto :netmode
echo [X] Unknown command "%CMD%". Run cracknet with no arguments for the menu.
goto :end

:start_live
echo Starting LIVE preview on %HOST% ...
REM restart, not start: start refuses when something is already up, which during
REM a demo is the moment you least want to debug a stale process.
call :run "cd %REMOTE% && CRACKNET_MODE=live ./cracknet.sh restart"
goto :after_start

:start_scan
echo Starting SCAN survey on %HOST% ...
call :run "cd %REMOTE% && CRACKNET_MODE=scan ./cracknet.sh restart"
goto :after_start

:after_start
echo.
echo   Dashboard: http://%HOST%:8081/
set "ans="
set /p "ans=  Open it now? [Y/n] "
if /i "!ans!"=="n" goto :end
start "" "http://%HOST%:8081/"
goto :end

:stop
REM Both modes share a port, so stop both rather than making the user remember
REM which one was started.
call :run "cd %REMOTE% && CRACKNET_MODE=live ./cracknet.sh stop; CRACKNET_MODE=scan ./cracknet.sh stop"
goto :end

:status
call :run "cd %REMOTE% && CRACKNET_MODE=live ./cracknet.sh status; echo; CRACKNET_MODE=scan ./cracknet.sh status"
goto :end

:logs
call :run "cd %REMOTE% && ls serve-*.log >/dev/null 2>&1 && tail -n 40 serve-*.log || echo 'no logs yet - nothing has been started'"
goto :end

:check
call :run "cd %REMOTE% && CRACKNET_MODE=live ./cracknet.sh check"
goto :end

:identify
echo Capturing one labelled frame per camera ...
call :run "cd %REMOTE% && python3 capture.py identify --out scan"
echo.
echo Look at the images, then write the mapping into %REMOTE%/rig.json
goto :end

:open
start "" "http://%HOST%:8081/"
goto :end

:shell
if "%~2"=="" (
    echo Usage: cracknet shell "command to run on the jetson"
    goto :end
)
call :run "%~2"
goto :end

:join
REM One time per PC. The AP password is generated on the Jetson and fetched
REM over whatever route works now, so nobody ever types it.
%PY% tools\jlink.py join
goto :end

:leave
%PY% tools\jlink.py leave
goto :end

:link
%PY% tools\jlink.py status
goto :end

:netmode
if "%~2"=="" (
    echo Usage: cracknet netmode ap^|wifi
    echo.
    echo   ap     the Jetson hosts its own Wi-Fi - reachable with no cables,
    echo          but it has no internet in this mode
    echo   wifi   the Jetson rejoins the campus Wi-Fi - it gets internet, but
    echo          client isolation there means this PC cannot reach it
    goto :end
)
REM Switching the radio cannot be done over the radio: the command that tears
REM the AP down is also the command whose reply has to travel across it. Done
REM over USB when USB is present, and warned about when it is not.
%PY% -c "import sys;sys.path.insert(0,'tools');import jssh;sys.exit(0 if jssh.reachable('192.168.55.1',timeout=2) else 1)"
if errorlevel 1 (
    echo [!] The USB link is not up, so this command will be sent over the
    echo     very Wi-Fi link it changes. Expect it to cut off mid-reply.
    echo     "netmode wifi" will then leave the board reachable only by USB
    echo     or a monitor. Plug the USB cable in first if you can.
    echo.
    set "go="
    set /p "go=    Continue anyway? [y/N] "
    if /i not "!go!"=="y" goto :end
)
echo Setting the Jetson link mode to %~2 ...
%PY% "%JSSH%" --sudo --timeout 180 "export NO_COLOR=1; cd %REMOTE% && ./netlink.sh %~2"
goto :end

REM ------------------------------------------------------------- helper -----
:run
REM %~1 is the remote command. Quoting matters here: it is passed through cmd,
REM then Python, then bash, so it is sent as ONE argument and left alone.
%PY% "%JSSH%" "export NO_COLOR=1; %~1"
if errorlevel 1 (
    echo.
    echo [!] Exited non-zero.
    echo     If something printed above, that IS the answer - a failed preflight
    echo     exits non-zero on purpose, so fix what it listed.
    echo     If nothing printed, it could not reach the Jetson:
    echo       - powered on?
    echo       - cracknet link      which route is live
    echo       - cracknet join      rejoin the Jetson Wi-Fi
    echo       - ping %HOST%
    echo       - credentials in %ENVFILE%
)
exit /b

:end
if "%~1"=="" (
    echo.
    pause
)
endlocal
