#!/usr/bin/env bash
# Запуск Face Detector CLI на Linux / Raspberry Pi.
# Создаёт venv при первом запуске, ставит зависимости, прокидывает аргументы.
set -e
cd "$(dirname "$0")"

# На Raspberry Pi используем requirements-pi.txt (headless OpenCV)
REQ=requirements.txt
if [ -f /proc/device-tree/model ] && grep -qi raspberry /proc/device-tree/model; then
    REQ=requirements-pi.txt
    echo "[run] Обнаружен Raspberry Pi → $REQ"
fi

if [ ! -d venv ]; then
    echo "[run] Создаю виртуальное окружение..."
    python3 -m venv venv --system-site-packages   # доступ к системному picamera2
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r "$REQ"
fi

exec ./venv/bin/python main.py "$@"
