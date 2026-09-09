@echo off
REM Gurobi parameter test on RDS. Re-solves 4 pilot runs under settings B, C, D
REM (barrier root, focus/cuts, NoRel) and F (6 threads) and compares with the
REM pilot's own solve. 14 parallel jobs (13x3 + 1x6 = 45 of 48 cores), 2.5 h time limit each -> about 3 h.
REM
REM Edit SOURCE if the pilot folder name differs. Run from any location.

cd /d %~dp0
set SOURCE=results\parallel_creation_test_20260909_145257

python solver_param_test.py --source %SOURCE% --workers 14 --dry-run
echo.
echo Starting in 10 s (Ctrl+C to abort) ...
timeout /t 10 >nul

python solver_param_test.py --source %SOURCE% --workers 14

echo.
echo Done. Report: results\solver_param_test_*\report.txt
pause
