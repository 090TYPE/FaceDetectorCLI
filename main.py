#!/usr/bin/env python3
"""
Face Detector CLI — детекция и анализ лиц из командной строки.
Работает на ПК (Windows/Linux/macOS) и на Raspberry Pi.

Режимы:
    image   — анализ одного фото
    video   — анализ видеофайла
    webcam  — анализ потока с камеры (USB или Pi Camera) в реальном времени

Примеры:
    python main.py image photo.jpg
    python main.py image photo.jpg -o out.jpg --hands
    python main.py video clip.mp4 -o annotated.mp4 --no-display
    python main.py webcam --camera 0
    python main.py webcam --picamera --no-display --save-dir snaps   # Raspberry Pi headless

На Raspberry Pi без монитора (headless) обязательно используйте --no-display.
"""

import argparse
import os
import sys
import time

import cv2

import core


# ── Вспомогательное ───────────────────────────────────────────────────────────────
def _has_display() -> bool:
    """Есть ли графическое окружение для cv2.imshow()."""
    if sys.platform.startswith('win') or sys.platform == 'darwin':
        return True
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def _annotate(frame, args, analyze=True):
    """Найти людей, лица (и руки), нарисовать рамки. Вернуть (frame, results)."""
    results = []

    # Люди (целиком) — нужны для --people и для слежения --track
    if getattr(args, 'people', False) or getattr(args, 'track', False):
        for (x, y, w, h) in core.detect_people(frame, conf_thr=args.conf):
            core.draw_person_box(frame, x, y, w, h)
            results.append({'person': (x, y, w, h)})

    # Лица + атрибуты
    for (x, y, w, h) in core.detect_faces(frame, conf_thr=args.conf):
        data = core.analyze_face(
            frame[y:y + h, x:x + w],
            age=not args.no_age,
            gender=not args.no_gender,
            emotion=not args.no_emotion,
        ) if analyze else None
        core.draw_face_box(frame, x, y, w, h, data)
        results.append({'box': (x, y, w, h), **(data or {})})

    if args.hands:
        for hd in core.detect_hands(frame):
            core.draw_hands(frame, [hd])
            results.append({'hand': hd['handedness'], 'fingers': hd['fingers']})

    return frame, results


def _box_center(box):
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


def _box_area(box):
    return box[2] * box[3]


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class TargetLock:
    """Захват цели: увидев человека, ведёт ИМЕННО ЕГО (ближайшего к прошлой
    позиции), удерживая через кратковременные пропадания, и отпускает только
    когда цель исчезла из кадра надолго. Сглаживает позицию (детекция дрожит)."""

    def __init__(self, mode='auto', smooth=0.45, max_lost=25, gate=0.33):
        self.mode = mode
        self.smooth = smooth        # сглаживание позиции (0..1, больше = резче)
        self.max_lost = max_lost    # кадров без цели до сброса захвата
        self.gate = gate            # макс. «прыжок» цели за кадр (доля диагонали)
        self.center = None          # сглаженный центр захваченной цели
        self.box = None             # рамка захваченной цели
        self.lost = 0

    def _candidates(self, results):
        people = [r['person'] for r in results if 'person' in r]
        faces = [r['box'] for r in results if 'box' in r]
        if self.mode == 'face':
            return faces or people
        return people or faces      # 'person'/'auto' → люди, иначе лица

    def update(self, results, frame_size):
        w, h = frame_size
        cands = self._candidates(results)
        if not cands:
            self.lost += 1
            if self.lost > self.max_lost:
                self.center = self.box = None
            return None             # цель не видно → СТОП (не гоним на старую позицию)

        cen = [(_box_center(b), b) for b in cands]
        if self.center is not None:
            gate_px = self.gate * ((w * w + h * h) ** 0.5)
            best = min(cen, key=lambda cb: _dist(cb[0], self.center))
            if _dist(best[0], self.center) <= gate_px:
                chosen = best                      # та же цель рядом — ведём её
                self.lost = 0
            else:
                self.lost += 1                     # рядом никого — цель ушла
                if self.lost <= self.max_lost:
                    return None                     # СТОП, но захват держим (ждём возврата)
                chosen = max(cen, key=lambda cb: _box_area(cb[1]))  # перезахват
                self.lost = 0
        else:
            chosen = max(cen, key=lambda cb: _box_area(cb[1]))      # первичный захват
            self.lost = 0

        nc, self.box = chosen
        if self.center is None:
            self.center = nc
        else:
            s = self.smooth
            self.center = (int(s * nc[0] + (1 - s) * self.center[0]),
                           int(s * nc[1] + (1 - s) * self.center[1]))
        return self.center


def _pick_target(results, mode):
    """Выбрать цель для слежения → (cx, cy) или None.
    mode: 'person' (по людям), 'face' (по лицам), 'auto' (человек, иначе лицо).
    Берём самый крупный объект (обычно ближайший)."""
    people = [r['person'] for r in results if 'person' in r]
    faces = [r['box'] for r in results if 'box' in r]
    if mode in ('person', 'auto') and people:
        return _box_center(max(people, key=_box_area))
    if mode in ('face', 'auto') and faces:
        return _box_center(max(faces, key=_box_area))
    if mode == 'person' and faces:   # людей нет, но есть лицо — следим за ним
        return _box_center(max(faces, key=_box_area))
    return None


def _print_results(results, label=''):
    faces = [r for r in results if 'box' in r]
    hands = [r for r in results if 'hand' in r]
    people = [r for r in results if 'person' in r]
    prefix = f'{label} ' if label else ''
    print(f'{prefix}Людей: {len(people)}, лиц: {len(faces)}'
          + (f', рук: {len(hands)}' if hands else ''))
    for i, r in enumerate(faces, 1):
        parts = []
        if r.get('emotion'):
            parts.append(r['emotion'])
        if r.get('age') is not None:
            parts.append(f"~{r['age']} лет")
        if r.get('gender'):
            parts.append(r['gender'])
        print(f'    Лицо {i}: ' + (', '.join(parts) if parts else '(без атрибутов)'))
    for i, h in enumerate(hands, 1):
        print(f"    Рука {i}: {h['hand']}, пальцев: {h['fingers']}")


# ── Команда: image ────────────────────────────────────────────────────────────────
def cmd_image(args):
    if not os.path.exists(args.path):
        sys.exit(f'Файл не найден: {args.path}')
    frame = cv2.imread(args.path)
    if frame is None:
        sys.exit(f'Не удалось прочитать изображение: {args.path}')

    t0 = time.time()
    frame, results = _annotate(frame, args)
    print(f'Обработано за {time.time() - t0:.2f} c')
    _print_results(results)

    out = args.output or _default_out(args.path, '_annotated')
    cv2.imwrite(out, frame)
    print(f'Сохранено: {out}')

    if not args.no_display and _has_display():
        cv2.imshow('Face Detector — фото (любая клавиша = выход)', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── Команда: video ────────────────────────────────────────────────────────────────
def cmd_video(args):
    if not os.path.exists(args.path):
        sys.exit(f'Файл не найден: {args.path}')
    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        sys.exit(f'Не удалось открыть видео: {args.path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = args.output or _default_out(args.path, '_annotated', ext='.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    show = not args.no_display and _has_display()
    n, t0 = 0, time.time()
    every = max(1, args.every)
    last_results = []
    print(f'Обработка видео ({total or "?"} кадров), анализ каждого {every}-го кадра...')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % every == 0:
                frame, last_results = _annotate(frame, args)
            else:
                # лёгкая отрисовка без повторного анализа: только детекция лиц
                for (x, y, ww, hh) in core.detect_faces(frame, conf_thr=args.conf):
                    cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 100), 2)
            writer.write(frame)

            if show:
                cv2.imshow('Face Detector — видео (q = выход)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if n % 30 == 0:
                pct = f'{100 * n / total:.0f}%' if total else f'{n} кадров'
                print(f'  {pct}  ({n / (time.time() - t0):.1f} FPS)', flush=True)
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
    print(f'Готово. Сохранено: {out_path}')
    _print_results(last_results, label='Последний кадр:')


# ── Команда: webcam ───────────────────────────────────────────────────────────────
def cmd_webcam(args):
    grab = _open_camera(args)
    show = not args.no_display and _has_display()
    if not show and not args.save_dir:
        print('⚠ Нет дисплея и не задан --save-dir. Результаты будут только в консоли.')

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Поворотная камера (слежение за человеком)
    trk = None
    if args.track:
        from tracker import PanTiltTracker
        # значения по умолчанию зависят от типа привода
        if args.servo == 'topogs':           # гимбал Topotek через GroundStation (порт 2337)
            gain = args.track_gain if args.track_gain is not None else 500.0
            deadzone = args.deadzone if args.deadzone is not None else 0.045
            min_rate = args.min_rate if args.min_rate is not None else 55.0
            port = args.gimbal_port if args.gimbal_port != 9003 else 2337
        elif args.servo == 'topotek':        # гимбал Topotek ASCII (порт 9003)
            gain = args.track_gain if args.track_gain is not None else 70.0
            deadzone = args.deadzone if args.deadzone is not None else 0.035
            min_rate = args.min_rate if args.min_rate is not None else 10.0
            port = args.gimbal_port
        else:                                # сервоприводы
            gain = args.track_gain if args.track_gain is not None else 18.0
            deadzone = args.deadzone if args.deadzone is not None else 0.06
            min_rate = args.min_rate if args.min_rate is not None else 7.0
            port = args.gimbal_port
        trk = PanTiltTracker(
            backend=args.servo, pan_pin=args.pan_pin, tilt_pin=args.tilt_pin,
            gimbal_ip=args.gimbal_ip, gimbal_port=port,
            gain=gain, rate_gain=gain, min_rate=min_rate, deadzone=deadzone,
            ctrl=args.gimbal_ctrl,
            invert_pan=args.invert_pan, invert_tilt=args.invert_tilt,
            track_pan=not args.no_pan, track_tilt=not args.no_tilt,
            verbose=(args.track_verbose or not show),
        )
        print(f'[tracker] Слежение включено, цель: {args.track_target}')
    lock = TargetLock(args.track_target) if args.track else None

    every = max(1, args.every)
    n, t0, saved = 0, time.time(), 0
    print('Запуск камеры... (Ctrl+C для остановки' +
          (', q в окне для выхода)' if show else ')'))
    try:
        while True:
            frame = grab()
            if frame is None:
                print('Кадр не получен, остановка.')
                break
            n += 1
            h, w = frame.shape[:2]

            # При слежении детекция нужна каждый кадр (плавный поворот),
            # тяжёлый анализ атрибутов — только каждый N-й кадр.
            do_full = (n % every == 0)
            if trk or do_full:
                frame, results = _annotate(frame, args, analyze=do_full)
            else:
                results = []

            # Захват цели + отрисовка
            target = None
            if trk:
                target = lock.update(results, (w, h))
                if show:
                    if lock.box is not None and target is not None:
                        bx, by, bw, bh = lock.box
                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
                    _draw_tracking_overlay(frame, target, (w, h))

            if do_full and results and (n % (every * 5) == 0 or not show):
                _print_results(results, label=f'[{n}]')
            if do_full and args.save_dir and results:
                p = os.path.join(args.save_dir, f'snap_{int(time.time())}_{n}.jpg')
                cv2.imwrite(p, frame)
                saved += 1

            key = -1
            if show:
                cv2.imshow('Face Detector (q=выход, стрелки/WASD=ручное наведение)', frame)
                key = cv2.waitKeyEx(1)
                if (key & 0xFF) == ord('q'):
                    break

            # Управление гимбалом: ручное (клавиши) важнее авто-слежения
            if trk:
                manual = _manual_from_key(key) if show else None
                if manual is not None and trk.rate_based:
                    trk._be.apply_rate(manual[0], manual[1])
                    lock.center = lock.box = None     # сброс захвата после ручного
                else:
                    trk.update(target, (w, h))

            if args.max_frames and n >= args.max_frames:
                print(f'Достигнут лимит {args.max_frames} кадров.')
                break
    except KeyboardInterrupt:
        print('\nОстановлено пользователем.')
    finally:
        if hasattr(grab, 'close'):
            grab.close()
        if trk:
            trk.center()
            trk.close()
        if show:
            cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f'Кадров: {n}, средний FPS: {n / dt:.1f}' if dt else '')
    if args.save_dir:
        print(f'Сохранено снимков: {saved} в {args.save_dir}/')


def _manual_from_key(key, speed=280):
    """Стрелки/WASD → ручная команда скорости гимбала (pan, tilt) или None."""
    if key in (2424832, ord('a'), ord('A')):   # ← влево
        return (-speed, 0)
    if key in (2555904, ord('d'), ord('D')):   # → вправо
        return (speed, 0)
    if key in (2490368, ord('w'), ord('W')):   # ↑ вверх
        return (0, speed)
    if key in (2621440, ord('s'), ord('S')):   # ↓ вниз
        return (0, -speed)
    return None


def _draw_tracking_overlay(frame, target, frame_size):
    """Перекрестие в центре кадра + линия к цели слежения."""
    w, h = frame_size
    cx, cy = w // 2, h // 2
    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 1)
    if target is not None:
        tx, ty = target
        cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 255), 1)
        cv2.circle(frame, (tx, ty), 8, (0, 0, 255), 2)


def _open_camera(args):
    """Вернуть callable grab() -> frame|None. Поддержка IP/RTSP, USB и Pi Camera."""
    if getattr(args, 'rtsp', None):
        return _open_network(args.rtsp)

    if args.picamera:
        try:
            from picamera2 import Picamera2
        except ImportError:
            sys.exit('picamera2 не установлен. Установите: sudo apt install -y python3-picamera2')
        picam = Picamera2()
        cfg = picam.create_preview_configuration(
            main={'format': 'RGB888', 'size': (args.width, args.height)})
        picam.configure(cfg)
        picam.start()
        time.sleep(1)  # прогрев сенсора

        def grab():
            rgb = picam.capture_array()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        grab.close = picam.stop
        return grab

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f'Не удалось открыть камеру #{args.camera}. '
                 'Попробуйте другой индекс (--camera 1) или --picamera.')

    def grab():
        ok, frame = cap.read()
        return frame if ok else None
    grab.close = cap.release
    return grab


def _open_network(url):
    """Сетевой поток (RTSP) с НИЗКОЙ задержкой: отдельный поток-читатель всегда
    держит самый свежий кадр (накопившиеся выбрасываются), + low-delay FFmpeg."""
    import threading
    # Низколатентные опции FFmpeg (TCP + без буферизации + без переупорядочивания).
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
        'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|reorder_queue_size;0')

    def _connect():
        c = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return c

    print(f'Подключение к сетевой камере (низкая задержка): {url}')
    cap = _connect()
    if not cap.isOpened():
        sys.exit(f'Не удалось открыть поток: {url}\nПроверьте IP и доступность камеры (ping).')

    state = {'cap': cap, 'frame': None, 'run': True, 'reported': False}
    lock = threading.Lock()

    def reader():
        while state['run']:
            try:
                ok, f = state['cap'].read()
            except Exception:
                ok, f = False, None
            if ok and f is not None:
                with lock:
                    state['frame'] = f      # держим ТОЛЬКО последний кадр
                state['reported'] = False
            else:
                if not state['reported']:
                    print('  [net] поток прервался, переподключение...', flush=True)
                    state['reported'] = True
                try:
                    state['cap'].release()
                except Exception:
                    pass
                time.sleep(0.5)
                state['cap'] = _connect()

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    def grab():
        t0 = time.time()
        while time.time() - t0 < 5:
            with lock:
                f = state['frame']
                state['frame'] = None        # забрали — ждём следующий свежий
            if f is not None:
                return f
            time.sleep(0.005)
        return None

    def _close():
        state['run'] = False
        time.sleep(0.1)
        try:
            state['cap'].release()
        except Exception:
            pass
    grab.close = _close
    return grab


def _default_out(path, suffix, ext=None):
    base, e = os.path.splitext(path)
    return base + suffix + (ext or e)


# ── Парсер аргументов ─────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description='Face Detector CLI — детекция лиц/возраста/пола/эмоций (+руки).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest='command', required=True)

    def common(sp):
        sp.add_argument('--conf', type=float, default=0.5, help='Порог уверенности детекции (0-1)')
        sp.add_argument('--people', action='store_true', help='Распознавать людей целиком (MobileNet-SSD)')
        sp.add_argument('--hands', action='store_true', help='Включить детекцию рук (нужен mediapipe)')
        sp.add_argument('--no-age', action='store_true', help='Не определять возраст')
        sp.add_argument('--no-gender', action='store_true', help='Не определять пол')
        sp.add_argument('--no-emotion', action='store_true', help='Не определять эмоции')
        sp.add_argument('--no-display', action='store_true', help='Не открывать окно (headless / Raspberry Pi)')

    sp = sub.add_parser('image', help='Анализ одного фото')
    sp.add_argument('path', help='Путь к изображению')
    sp.add_argument('-o', '--output', help='Куда сохранить результат')
    common(sp)
    sp.set_defaults(func=cmd_image)

    sp = sub.add_parser('video', help='Анализ видеофайла')
    sp.add_argument('path', help='Путь к видео')
    sp.add_argument('-o', '--output', help='Куда сохранить результат (.mp4)')
    sp.add_argument('--every', type=int, default=15, help='Полный анализ каждого N-го кадра')
    common(sp)
    sp.set_defaults(func=cmd_video)

    sp = sub.add_parser('webcam', help='Камера в реальном времени (USB или Pi Camera)')
    sp.add_argument('--camera', type=int, default=0, help='Индекс USB-камеры (по умолчанию 0)')
    sp.add_argument('--picamera', action='store_true', help='Использовать Raspberry Pi Camera (picamera2)')
    sp.add_argument('--rtsp', help='URL сетевой IP-камеры, напр. rtsp://user:pass@192.168.1.10:554/stream')
    sp.add_argument('--width', type=int, default=640, help='Ширина кадра')
    sp.add_argument('--height', type=int, default=480, help='Высота кадра')
    sp.add_argument('--every', type=int, default=10, help='Анализ каждого N-го кадра (больше = легче для CPU)')
    sp.add_argument('--save-dir', help='Папка для сохранения кадров с лицами')
    sp.add_argument('--max-frames', type=int, default=0, help='Остановиться после N кадров (0 = бесконечно)')
    # ── Слежение поворотной камерой (pan-tilt серво) ──────────────────────────
    sp.add_argument('--track', action='store_true',
                    help='Поворачивать камеру за человеком (нужны сервоприводы pan-tilt)')
    sp.add_argument('--track-target', choices=['person', 'face', 'auto'], default='auto',
                    help='За чем следить: person/face/auto (по умолч. auto — человек, иначе лицо)')
    sp.add_argument('--servo', choices=['auto', 'pantilthat', 'servokit', 'gpiozero', 'topotek', 'topogs', 'sim'],
                    default='auto', help='Привод: серво (pantilthat/servokit/gpiozero), '
                                         'гимбал Topotek (topotek=ASCII 9003, topogs=GroundStation 2337), sim, auto')
    sp.add_argument('--pan-pin', type=int, default=17, help='GPIO-пин серво поворота (для gpiozero)')
    sp.add_argument('--tilt-pin', type=int, default=18, help='GPIO-пин серво наклона (для gpiozero)')
    sp.add_argument('--gimbal-ip', default='192.168.144.108', help='IP гимбала Topotek (--servo topotek)')
    sp.add_argument('--gimbal-port', type=int, default=9003, help='UDP-порт управления Topotek (по умолч. 9003)')
    sp.add_argument('--gimbal-ctrl', choices=['rate', 'dir'], default='rate',
                    help='Управление гимбалом: rate (скорость YPR) или dir (команды направления)')
    sp.add_argument('--track-verbose', action='store_true',
                    help='Печатать телеметрию слежения даже при открытом окне (для отладки)')
    sp.add_argument('--track-gain', type=float, default=None,
                    help='Усиление слежения (серво: градусов/кадр ~18; гимбал: масштаб скорости ~70)')
    sp.add_argument('--deadzone', type=float, default=None,
                    help='Мёртвая зона у центра, доля кадра (меньше = точнее центрирует, по умолч. ~0.035)')
    sp.add_argument('--min-rate', type=float, default=None,
                    help='Мин. скорость гимбала, чтобы мотор докручивал до конца')
    sp.add_argument('--invert-pan', action='store_true', help='Инвертировать ось поворота')
    sp.add_argument('--invert-tilt', action='store_true', help='Инвертировать ось наклона')
    sp.add_argument('--no-pan', action='store_true', help='Не вести по горизонтали')
    sp.add_argument('--no-tilt', action='store_true', help='Не вести по вертикали (только горизонталь)')
    common(sp)
    sp.set_defaults(func=cmd_webcam)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
