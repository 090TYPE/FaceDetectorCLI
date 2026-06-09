@echo off
REM Запуск Face Detector CLI на Windows.
REM Создаёт venv при первом запуске, ставит зависимости, прокидывает аргументы.
REM Сам находит интерпретатор: сначала py (Python Launcher), потом python.

setlocal
cd /d "%~dp0"

REM Выбор интерпретатора для создания venv: предпочитаем py
set "BOOT=py"
where py >nul 2>nul || set "BOOT=python"

if not exist venv (
    echo [run] Создаю виртуальное окружение через %BOOT% ...
    %BOOT% -m venv venv
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
)

REM RTSP стабильнее по TCP (камеры Topotek и др.)
set "OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp"

venv\Scripts\python.exe main.py %*
