# Face Detector CLI

[![CI](https://github.com/090TYPE/FaceDetectorCLI/actions/workflows/ci.yml/badge.svg)](https://github.com/090TYPE/FaceDetectorCLI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
Данный детектор был реализован для камеры от сибиричка(БпЛА коптерного типа) поставленную на обычный распечатанный пьедестал на 3d принтере
Если вдруг будет интересно как всё утроенно задавайте вопросы я отвечу
Консольное приложение для **детекции лиц и людей** и анализа **возраста / пола / эмоций**
(плюс опционально — рук) из командной строки. Без веб-интерфейса.
Умеет **поворачивать камеру за человеком** (pan-tilt сервоприводы на Raspberry Pi).
Работает на ПК (Windows / Linux / macOS) и на **Raspberry Pi** (в т.ч. headless, без монитора).

Основано на том же движке, что и веб-версия FaceDetector: OpenCV DNN (детекция лиц,
ResNet-SSD) + Caffe-модели Levi-Hassner (возраст/пол) + ONNX FER+ (эмоции) +
MediaPipe (руки). TensorFlow не требуется.

---

## Установка

### Windows / Linux / macOS (ПК)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```
Или просто запустите `run.bat` (Windows) / `./run.sh` (Linux) — окружение создастся само.

### Raspberry Pi
```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2 libatlas-base-dev

git clone <repo> FaceDetectorCLI && cd FaceDetectorCLI
chmod +x run.sh
./run.sh webcam --picamera --no-display --save-dir snaps
```
`run.sh` сам определяет Raspberry Pi и ставит облегчённый `requirements-pi.txt`
(`opencv-python-headless` вместо полного OpenCV).

> Модели (~100 МБ) скачиваются автоматически при первом запуске в `models_cv/`.
> Нужен интернет на первом запуске; дальше работает офлайн.

---

## Использование

```bash
# Фото
python main.py image photo.jpg
python main.py image photo.jpg -o out.jpg --people     # + рамки людей целиком
python main.py image photo.jpg -o out.jpg --hands

# Видеофайл
python main.py video clip.mp4 -o annotated.mp4
python main.py video clip.mp4 --no-display          # без окна, только запись

# Камера (USB-вебка)
python main.py webcam --camera 0

# Сетевая IP-камера (RTSP-поток по Ethernet)
python main.py webcam --rtsp rtsp://user:pass@192.168.1.10:554/stream --no-display

# Камера Raspberry Pi (модуль камеры), headless
python main.py webcam --picamera --no-display --save-dir snaps --every 15

# Распознавать людей + ПОВОРАЧИВАТЬ камеру за человеком (сервоприводы pan-tilt)
python main.py webcam --picamera --people --track --no-display

# Проверить логику слежения на ПК без сервоприводов (печатает углы)
python main.py webcam --camera 0 --track --servo sim
```

---

## Сетевая IP-камера (`--rtsp`)

Если камера отдаёт видео по сети (Ethernet, RTSP/H.264) — указываете её URL,
USB-захват не нужен. Есть автопереподключение при обрыве потока.
```bash
python main.py webcam --rtsp rtsp://admin:1234@192.168.1.10:554/stream --people --no-display
```
Формат URL зависит от камеры, типовые пути: `/stream`, `/live`, `/h264`,
`/onvif1`, `/cam/realmonitor?channel=1&subtype=0`. Узнать точный путь помогает
ONVIF Device Manager или документация на модуль камеры.

---

## Распознавание людей (`--people`)

Флаг `--people` включает детектор людей целиком (а не только лиц) — модель
**MobileNet-SSD** (OpenCV DNN, класс `person`). Найденные люди обводятся отдельной
рамкой с подписью «Человек». Модель (~23 МБ) скачивается автоматически при первом
использовании. Работает в `image`, `video` и `webcam`.

---

## Поворот камеры за человеком (`--track`)

На Raspberry Pi камера ставится на **поворотный механизм pan-tilt** с двумя
сервоприводами (горизонталь + вертикаль). При включённом `--track` приложение
находит человека, вычисляет смещение его центра от центра кадра и плавно
доворачивает камеру (пропорциональный регулятор), удерживая человека по центру.

### Поддерживаемое железо (`--servo`)
| Привод        | Что это                          | Установка                                            |
|---------------|----------------------------------|------------------------------------------------------|
| `pantilthat`  | Pimoroni Pan-Tilt HAT            | `pip install pantilthat`                             |
| `servokit`    | Adafruit PCA9685 / ServoKit (I2C)| `pip install adafruit-circuitpython-servokit`        |
| `gpiozero`    | 2 обычных серво напрямую на GPIO | встроено в Raspberry Pi OS                            |
| `sim`         | без железа — печатает углы (отладка на ПК) | —                                          |
| `auto`        | автоопределение (по умолчанию)   | пробует pantilthat → servokit → gpiozero → sim       |

### Подключение обычных серво (gpiozero)
По умолчанию: серво **поворота** → GPIO **17**, серво **наклона** → GPIO **18**
(меняется флагами `--pan-pin` / `--tilt-pin`). Серво питать от внешних 5 В,
землю объединить с Raspberry Pi.
```bash
python main.py webcam --picamera --track --servo gpiozero --pan-pin 17 --tilt-pin 18 --no-display
```

### Настройка слежения
| Флаг               | Назначение                                                        |
|--------------------|-------------------------------------------------------------------|
| `--track`          | Включить слежение                                                 |
| `--track-target`   | За чем следить: `person` / `face` / `auto` (по умолч. auto)        |
| `--servo`          | Тип привода (см. таблицу выше)                                     |
| `--track-gain N`   | Скорость доворота (градусов/кадр; меньше = плавнее, по умолч. 18)  |
| `--invert-pan`     | Если камера крутится в другую сторону по горизонтали               |
| `--invert-tilt`    | То же для вертикали                                                |

> Если камера уезжает **от** человека вместо слежения — добавьте `--invert-pan`
> и/или `--invert-tilt` (зависит от того, как смонтированы серво).

### Команды
| Команда  | Что делает                                              |
|----------|---------------------------------------------------------|
| `image`  | Анализ одного изображения, сохраняет размеченную копию  |
| `video`  | Анализ видеофайла, пишет размеченный `.mp4`             |
| `webcam` | Поток с камеры в реальном времени (USB или Pi Camera)   |

### Полезные флаги
| Флаг            | Назначение                                              |
|-----------------|---------------------------------------------------------|
| `--no-display`  | Не открывать окно (**обязательно на headless Pi**)      |
| `--picamera`    | Использовать модуль камеры Raspberry Pi (picamera2)     |
| `--camera N`    | Индекс USB-камеры (по умолчанию 0)                      |
| `--rtsp URL`    | Сетевая IP-камера (RTSP/HTTP поток по Ethernet)         |
| `--people`      | Распознавать людей целиком (MobileNet-SSD)              |
| `--track`       | Поворачивать камеру за человеком (см. раздел ниже)      |
| `--hands`       | Включить детекцию рук (нужен `mediapipe`)               |
| `--every N`     | Анализировать каждый N-й кадр (больше = легче для CPU)  |
| `--save-dir D`  | Сохранять кадры с найденными лицами в папку D           |
| `--max-frames N`| Остановить webcam после N кадров                        |
| `--conf X`      | Порог уверенности детекции лиц (0–1, по умолч. 0.5)     |
| `--no-age` / `--no-gender` / `--no-emotion` | Отключить отдельные атрибуты (быстрее) |

---

## Советы по производительности на Raspberry Pi
- Используйте `--no-display` (без X-сервера экономит ресурсы).
- Поднимите `--every` (например `--every 20`) — реже гонять тяжёлый анализ.
- Уменьшите разрешение: `--width 480 --height 360`.
- `--no-emotion` / `--no-age` ускоряют обработку, если эти атрибуты не нужны.
- Детекцию рук (`--hands` / mediapipe) на Pi Zero / 3 лучше не включать.

### Автозапуск при загрузке (systemd)
`/etc/systemd/system/facedetector.service`:
```ini
[Unit]
Description=Face Detector CLI
After=multi-user.target

[Service]
WorkingDirectory=/home/pi/FaceDetectorCLI
ExecStart=/home/pi/FaceDetectorCLI/run.sh webcam --picamera --no-display --save-dir /home/pi/snaps
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now facedetector
```

---

## Структура
```
FaceDetectorCLI/
├── main.py              # CLI: команды image / video / webcam
├── core.py              # движок: детекция лиц/людей, возраст/пол/эмоции, руки
├── tracker.py           # слежение поворотной камерой (pan-tilt серво)
├── requirements.txt     # зависимости для ПК
├── requirements-pi.txt  # облегчённые зависимости для Raspberry Pi
├── run.bat / run.sh     # запуск одной командой (создают venv)
└── models_cv/           # модели (скачиваются автоматически)
```
