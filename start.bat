@echo off

:main
conda info --envs
set /p select=Select your anaconda env : 
goto excute

:excute
cls
color B
echo C:\Users\jjs00\Anaconda3\envs\%select%\python
color A
echo ====================================
echo            Server Excute!
echo ====================================
call C:\Users\jjs00\Anaconda3\Scripts\activate.bat %select%
python manage.py runserver