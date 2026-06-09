"""
Ядро анализа лиц — OpenCV DNN (детекция) + Caffe/ONNX (возраст/пол/эмоции).
Без TensorFlow. Работает на ПК и на Raspberry Pi (ARM).

Модели автоматически скачиваются при первом запуске в models_cv/.
MediaPipe (детекция рук) опционален — на Raspberry Pi его можно не ставить.
"""

import os
import urllib.request

import cv2
import numpy as np

# ── Пути ────────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cv')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Реестр загрузок (несколько зеркал на файл) ──────────────────────────────────
_URLS: dict[str, list[str]] = {
    'deploy.prototxt': [
        'https://raw.githubusercontent.com/opencv/opencv/master'
        '/samples/dnn/face_detector/deploy.prototxt',
    ],
    'res10_300x300_ssd_iter_140000.caffemodel': [
        'https://github.com/opencv/opencv_3rdparty/raw/'
        'dnn_samples_face_detector_20170830/'
        'res10_300x300_ssd_iter_140000.caffemodel',
    ],
    'age_deploy.prototxt': [
        'https://raw.githubusercontent.com/spmallick/learnopencv/'
        'master/AgeGender/age_deploy.prototxt',
        'https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/'
        'master/age_deploy.prototxt',
    ],
    'age_net.caffemodel': [
        'https://github.com/smahesh29/Gender-and-Age-Detection/'
        'raw/master/age_net.caffemodel',
        'https://github.com/spmallick/learnopencv/'
        'raw/master/AgeGender/age_net.caffemodel',
    ],
    'gender_deploy.prototxt': [
        'https://raw.githubusercontent.com/spmallick/learnopencv/'
        'master/AgeGender/gender_deploy.prototxt',
        'https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/'
        'master/gender_deploy.prototxt',
    ],
    'gender_net.caffemodel': [
        'https://github.com/smahesh29/Gender-and-Age-Detection/'
        'raw/master/gender_net.caffemodel',
        'https://github.com/spmallick/learnopencv/'
        'raw/master/AgeGender/gender_net.caffemodel',
    ],
    'emotion-ferplus-8.onnx': [
        'https://github.com/onnx/models/raw/main/validated/'
        'vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx',
        'https://media.githubusercontent.com/media/onnx/models/main/validated/'
        'vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx',
    ],

    # ── Детектор людей (MobileNet-SSD, 20 классов VOC, класс "person") ──────────
    'MobileNetSSD_deploy.prototxt': [
        'https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/'
        'master/MobileNetSSD_deploy.prototxt',
    ],
    'MobileNetSSD_deploy.caffemodel': [
        'https://github.com/djmv/MobilNet_SSD_opencv/'
        'raw/master/MobileNetSSD_deploy.caffemodel',
    ],
}

_MIN_SIZE = {
    'deploy.prototxt': 1_000,
    'res10_300x300_ssd_iter_140000.caffemodel': 5_000_000,
    'age_deploy.prototxt': 1_000,
    'age_net.caffemodel': 20_000_000,
    'gender_deploy.prototxt': 1_000,
    'gender_net.caffemodel': 20_000_000,
    'emotion-ferplus-8.onnx': 10_000_000,
    'MobileNetSSD_deploy.prototxt': 10_000,
    'MobileNetSSD_deploy.caffemodel': 20_000_000,
}


def _download(filename: str) -> str | None:
    """Вернуть локальный путь к модели; пробует зеркала по очереди."""
    path = os.path.join(MODELS_DIR, filename)
    min_sz = _MIN_SIZE.get(filename, 1_000)

    if os.path.exists(path) and os.path.getsize(path) >= min_sz:
        return path

    for url in _URLS.get(filename, []):
        print(f'  [models] Загрузка {filename} ...', flush=True)
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'})
            with urllib.request.urlopen(req, timeout=180) as resp, \
                    open(path, 'wb') as out:
                out.write(resp.read())
            if os.path.getsize(path) >= min_sz:
                print(f'  [models] ✓ {filename}', flush=True)
                return path
            print(f'  [models] ✗ {filename}: файл слишком мал, пробую дальше')
            os.remove(path)
        except Exception as exc:
            print(f'  [models] ✗ {filename}: {exc}')
            if os.path.exists(path):
                os.remove(path)

    print(f'  [models] ⚠ НЕ удалось загрузить {filename} (режим деградации)')
    return None


# ── Ленивая загрузка сетей ───────────────────────────────────────────────────────
_face_net = _age_net = _gender_net = _emotion_ses = None
_models_ok = {'age': None, 'gender': None, 'emotion': None}


def _face_net_():
    global _face_net
    if _face_net is None:
        proto = _download('deploy.prototxt')
        model = _download('res10_300x300_ssd_iter_140000.caffemodel')
        if proto and model:
            _face_net = cv2.dnn.readNetFromCaffe(proto, model)
    return _face_net


def _age_net_():
    global _age_net
    if _models_ok['age'] is False:
        return None
    if _age_net is None:
        proto, model = _download('age_deploy.prototxt'), _download('age_net.caffemodel')
        if proto and model:
            _age_net = cv2.dnn.readNetFromCaffe(proto, model)
            _models_ok['age'] = True
        else:
            _models_ok['age'] = False
    return _age_net


def _gender_net_():
    global _gender_net
    if _models_ok['gender'] is False:
        return None
    if _gender_net is None:
        proto, model = _download('gender_deploy.prototxt'), _download('gender_net.caffemodel')
        if proto and model:
            _gender_net = cv2.dnn.readNetFromCaffe(proto, model)
            _models_ok['gender'] = True
        else:
            _models_ok['gender'] = False
    return _gender_net


def _emotion_ses_():
    global _emotion_ses
    if _models_ok['emotion'] is False:
        return None
    if _emotion_ses is None:
        try:
            import onnxruntime as ort
            path = _download('emotion-ferplus-8.onnx')
            if path:
                _emotion_ses = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                _models_ok['emotion'] = True
            else:
                _models_ok['emotion'] = False
        except Exception as exc:
            print(f'  [models] onnxruntime недоступен: {exc}')
            _models_ok['emotion'] = False
    return _emotion_ses


# ── Детектор людей (MobileNet-SSD) ────────────────────────────────────────────────
_people_net = None
_people_ok = None  # None=не пробовали, True/False

# Классы VOC, на которых обучен MobileNet-SSD. Индекс 15 = "person".
_SSD_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
_PERSON_CLASS = 15


def _people_net_():
    global _people_net, _people_ok
    if _people_ok is False:
        return None
    if _people_net is None:
        proto = _download('MobileNetSSD_deploy.prototxt')
        model = _download('MobileNetSSD_deploy.caffemodel')
        if proto and model:
            _people_net = cv2.dnn.readNetFromCaffe(proto, model)
            _people_ok = True
        else:
            _people_ok = False
    return _people_net


def detect_people(frame: np.ndarray, conf_thr: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Список (x, y, w, h) рамок людей (класс person) с уверенностью > conf_thr."""
    net = _people_net_()
    if net is None:
        return []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                 (300, 300), 127.5)
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        if int(dets[0, 0, i, 1]) != _PERSON_CLASS:
            continue
        if float(dets[0, 0, i, 2]) < conf_thr:
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 + 10 and y2 > y1 + 10:
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def draw_person_box(frame, x, y, w, h, color=(255, 200, 0)):
    """Рамка человека + подпись 'Человек' (Pillow для кириллицы)."""
    try:
        from PIL import Image, ImageDraw
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        r, g, b = color[2], color[1], color[0]
        draw.rectangle([x, y, x + w, y + h], outline=(r, g, b), width=2)
        yp = y - 20 if y > 22 else y + 2
        draw.text((x + 1, yp + 1), 'Человек', font=_pil_font(), fill=(0, 0, 0))
        draw.text((x, yp), 'Человек', font=_pil_font(), fill=(r, g, b))
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, 'Person', (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


# ── Константы ─────────────────────────────────────────────────────────────────────
_MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
_AGE_MEANS = [1, 5, 10, 18, 28, 40, 50, 70]
_FER_LABELS = ['Нейтрально', 'Радость', 'Удивление', 'Грусть',
               'Злость', 'Отвращение', 'Страх', 'Нейтрально']


def detect_faces(frame: np.ndarray, conf_thr: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Список (x, y, w, h) рамок лиц с уверенностью > conf_thr."""
    net = _face_net_()
    if net is None:
        return []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        if float(dets[0, 0, i, 2]) < conf_thr:
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 + 10 and y2 > y1 + 10:
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def _blob227(face_img):
    return cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), _MODEL_MEAN, swapRB=False)


def _predict_age(face_img):
    net = _age_net_()
    if net is None:
        return None
    net.setInput(_blob227(face_img))
    return int(_AGE_MEANS[int(net.forward()[0].argmax())])


def _predict_gender(face_img):
    net = _gender_net_()
    if net is None:
        return None
    net.setInput(_blob227(face_img))
    return 'Мужской' if int(net.forward()[0].argmax()) == 0 else 'Женский'


def _predict_emotion(face_img):
    sess = _emotion_ses_()
    if sess is None:
        return None
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blob = cv2.resize(gray, (64, 64)).astype(np.float32).reshape(1, 1, 64, 64)
    preds = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    idx = int(preds[0].argmax())
    return _FER_LABELS[min(idx, len(_FER_LABELS) - 1)]


def analyze_face(face_img, *, age=True, gender=True, emotion=True) -> dict | None:
    """Анализ одного вырезанного лица → {'age', 'gender', 'emotion'}."""
    if face_img is None or face_img.size == 0:
        return None
    try:
        return {
            'age':     _predict_age(face_img) if age else None,
            'gender':  _predict_gender(face_img) if gender else None,
            'emotion': _predict_emotion(face_img) if emotion else None,
        }
    except Exception as exc:
        print(f'  [analyze_face] ошибка: {exc}')
        return None


# ── Отрисовка (Pillow для кириллицы; ASCII-фолбэк на cv2.putText) ──────────────────
def draw_face_box(frame, x, y, w, h, data, color=(0, 255, 100)):
    """Рамка + подписи. Pillow рендерит кириллицу; при его отсутствии — латиница."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        r, g, b = color[2], color[1], color[0]
        draw.rectangle([x, y, x + w, y + h], outline=(r, g, b), width=2)
        if data:
            font = _pil_font()
            lines = [
                data.get('emotion') or '',
                f"Возраст: {data.get('age') or '?'}",
                f"Пол: {data.get('gender') or '?'}",
            ]
            for i, line in enumerate(l for l in lines if l):
                yp = y - 8 - i * 18
                if yp > 2:
                    draw.text((x + 1, yp + 1), line, font=font, fill=(0, 0, 0))
                    draw.text((x, yp), line, font=font, fill=(r, g, b))
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # Фолбэк без Pillow — латиница через OpenCV
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if data:
            label = f"{data.get('age') or '?'} {data.get('gender') or ''}"
            cv2.putText(frame, label.strip(), (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


_pil_font_cache = None


def _pil_font(size: int = 15):
    global _pil_font_cache
    if _pil_font_cache is not None:
        return _pil_font_cache
    from PIL import ImageFont
    candidates = [
        'C:/Windows/Fonts/arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Raspberry Pi OS
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                _pil_font_cache = ImageFont.truetype(p, size)
                return _pil_font_cache
            except Exception:
                continue
    _pil_font_cache = ImageFont.load_default()
    return _pil_font_cache


# ── Опциональная детекция рук (MediaPipe) ─────────────────────────────────────────
_HAND_MODEL = 'hand_landmarker.task'
_URLS[_HAND_MODEL] = [
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/latest/hand_landmarker.task'
]
_MIN_SIZE[_HAND_MODEL] = 5_000_000

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

_hl_detector = None
_hl_init_done = False


def _get_hand_landmarker():
    global _hl_detector, _hl_init_done
    if _hl_init_done:
        return _hl_detector
    _hl_init_done = True
    try:
        import mediapipe as mp
    except ImportError:
        print('  [hands] MediaPipe не установлен — детекция рук отключена.')
        return None
    model_path = _download(_HAND_MODEL)
    if model_path is None:
        return None
    try:
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        _hl_detector = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        print('  [hands] ✓ HandLandmarker готов', flush=True)
    except Exception as exc:
        print(f'  [hands] ошибка: {exc}')
        _hl_detector = None
    return _hl_detector


def _count_fingers(lm) -> int:
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    count = sum(1 for t, p in zip(tips, pips) if lm[t].y < lm[p].y)
    if abs(lm[4].x - lm[3].x) > 0.04:
        count += 1
    return count


def detect_hands(frame) -> list[dict]:
    det = _get_hand_landmarker()
    if det is None:
        return []
    try:
        import mediapipe as mp
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result or not result.hand_landmarks:
            return []
        hands = []
        for i, lm_list in enumerate(result.hand_landmarks):
            xs = [lm.x * w for lm in lm_list]
            ys = [lm.y * h for lm in lm_list]
            x1, y1 = max(0, int(min(xs)) - 18), max(0, int(min(ys)) - 18)
            x2, y2 = min(w, int(max(xs)) + 18), min(h, int(max(ys)) + 18)
            side = 'Правая'
            if result.handedness and i < len(result.handedness):
                side = 'Правая' if result.handedness[i][0].category_name == 'Right' else 'Левая'
            hands.append({
                'box': (x1, y1, x2 - x1, y2 - y1),
                'handedness': side,
                'fingers': _count_fingers(lm_list),
                'landmarks_px': [(int(lm.x * w), int(lm.y * h)) for lm in lm_list],
            })
        return hands
    except Exception as exc:
        print(f'  [detect_hands] ошибка: {exc}')
        return []


def draw_hands(frame, hands):
    for hand in hands:
        pts = hand['landmarks_px']
        for i, j in _HAND_CONNECTIONS:
            if i < len(pts) and j < len(pts):
                cv2.line(frame, pts[i], pts[j], (150, 0, 255), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 220, 255), -1)
        x, y, w, h = hand['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
    return frame
