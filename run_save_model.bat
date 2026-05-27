@echo off
cd /d "%~dp0"
echo Trying active Jupyter kernel...
python scripts\export_from_kernel.py
if %ERRORLEVEL%==0 goto done
echo.
echo If that failed: open notebooks\legacy\Epilepsy.ipynb, run the SMOTE cell, then run the last cell "Save model".
pause
:done
