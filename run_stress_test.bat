@echo off
REM Windows batch script to run Locust stress tests
REM Usage: run_stress_test.bat [users] [spawn_rate] [duration]

set USERS=500
set SPAWN_RATE=50
set DURATION=5m
set HOST=http://localhost:8000

if not "%1"=="" set USERS=%1
if not "%2"=="" set SPAWN_RATE=%2
if not "%3"=="" set DURATION=%3

echo ======================================================================
echo Running Locust Stress Test
echo ======================================================================
echo Users: %USERS%
echo Spawn Rate: %SPAWN_RATE% users/second
echo Duration: %DURATION%
echo Host: %HOST%
echo ======================================================================
echo.

REM Create reports directory
if not exist "reports" mkdir reports

REM Run Locust
locust -f locustfile.py --host=%HOST% --headless -u %USERS% -r %SPAWN_RATE% -t %DURATION% --html=reports\stress_test_%USERS%.html --csv=reports\stress_test_%USERS%

echo.
echo ======================================================================
echo Test completed! Check reports\stress_test_%USERS%.html for results
echo ======================================================================
pause

