"""
Управление поворотной камерой (слежение за человеком).

Два типа привода:
  • Сервоприводы (угол): pantilthat / servokit / gpiozero — крутят внешние серво
    на pan-tilt механизме. Управление по УГЛУ (накопление).
  • Сетевой гимбал Topotek (скорость): шлёт команды поворота по UDP (порт 9003)
    самому гимбалу камеры. Управление по СКОРОСТИ (rate) — родной протокол Topotek.
  • sim — без железа, печатает, что бы сделал (для отладки на ПК).

Логика: пропорциональный регулятор держит цель в центре кадра.
"""

import sys
import time


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ── Протокол Topotek (кадры команд) ──────────────────────────────────────────────
def _topotek_ypr_frame(yaw: int, pitch: int, roll: int = 0) -> bytes:
    """Кадр скорости #tpUG6wYPR + yaw/pitch/roll(по байту) + CRC. Скорости -99..99."""
    data = '%02X%02X%02X' % (yaw & 0xFF, pitch & 0xFF, roll & 0xFF)
    body = '#tpUG' + ('%X' % len(data)) + 'wYPR' + data        # len(data)=6 -> '6'
    crc = sum(body.encode('ascii')) & 0xFF
    return (body + '%02X' % crc).encode('ascii')


def _topotek_mode_frame(code: int) -> bytes:
    """Команда режима/направления гимбала #TPUG2wPTZ<code> + CRC.
    code: 0=стоп,1=вверх,2=вниз,3=влево,4=вправо,5=домой,6=lock,7=follow."""
    body = '#TPUG2wPTZ%02X' % (code & 0xFF)
    crc = sum(body.encode('ascii')) & 0xFF
    return (body + '%02X' % crc).encode('ascii')


def _topotek_stop_frame() -> bytes:
    """Кадр остановки вращения #TPUG2wPTZ00 + CRC."""
    return _topotek_mode_frame(0)


# ── Протокол Topotek GroundStation (джойстик, порт 2337, бинарный a9d7) ───────────
def _crc16_xmodem(buf) -> int:
    crc = 0
    for b in buf:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


# 80-байтный шаблон (нейтраль), пойманный из GroundStation; меняем байты 7-11 и CRC.
_TOPOGS_TEMPLATE = bytes.fromhex('a9d74800010000000000000000000000000000000000000000000000000001000000000000010000000000000000000000000000000000000000000000000000000000000000b300')


def _topogs_packet(pan: int, tilt: int, moving: bool) -> bytes:
    """Команда джойстика: pan/tilt как int16 LE (байты 7-8, 9-10), флаг (байт 11),
    CRC16-XMODEM big-endian в конце."""
    p = bytearray(_TOPOGS_TEMPLATE)
    p[7:9] = int(pan).to_bytes(2, 'little', signed=True)
    p[9:11] = int(tilt).to_bytes(2, 'little', signed=True)
    p[11] = 0x04 if moving else 0x00
    crc = _crc16_xmodem(p[0:70])
    p[70:72] = crc.to_bytes(2, 'big')
    return bytes(p)


# ── Бэкенды привода ───────────────────────────────────────────────────────────────
class _Backend:
    """Базовый интерфейс. rate_based=False → управление углом apply(pan,tilt)."""
    name = 'base'
    rate_based = False

    def apply(self, pan: float, tilt: float):
        raise NotImplementedError

    def apply_rate(self, yaw: float, pitch: float):
        raise NotImplementedError

    def stop(self):
        pass

    def close(self):
        pass


class _PanTiltHat(_Backend):
    name = 'pantilthat'

    def __init__(self, **_):
        import pantilthat
        self._h = pantilthat
        self._h.pan(0); self._h.tilt(0)

    def apply(self, pan, tilt):
        self._h.pan(int(_clamp(pan, -90, 90)))
        self._h.tilt(int(_clamp(tilt, -90, 90)))

    def close(self):
        try: self._h.pan(0); self._h.tilt(0)
        except Exception: pass


class _ServoKit(_Backend):
    name = 'servokit'

    def __init__(self, pan_ch=0, tilt_ch=1, **_):
        from adafruit_servokit import ServoKit
        self._kit = ServoKit(channels=16)
        self._pan_ch, self._tilt_ch = pan_ch, tilt_ch
        self.apply(0, 0)

    def apply(self, pan, tilt):
        self._kit.servo[self._pan_ch].angle = _clamp(pan + 90, 0, 180)
        self._kit.servo[self._tilt_ch].angle = _clamp(tilt + 90, 0, 180)


class _GpioZero(_Backend):
    name = 'gpiozero'

    def __init__(self, pan_pin=17, tilt_pin=18, **_):
        from gpiozero import AngularServo
        self._pan = AngularServo(pan_pin, min_angle=-90, max_angle=90)
        self._tilt = AngularServo(tilt_pin, min_angle=-90, max_angle=90)
        self.apply(0, 0)

    def apply(self, pan, tilt):
        self._pan.angle = _clamp(pan, -90, 90)
        self._tilt.angle = _clamp(tilt, -90, 90)

    def close(self):
        try: self._pan.detach(); self._tilt.detach()
        except Exception: pass


class _TopotekNet(_Backend):
    """Сетевой гимбал Topotek: команды скорости по UDP (порт 9003)."""
    name = 'topotek'
    rate_based = True

    def __init__(self, gimbal_ip='192.168.144.108', gimbal_port=9003, **_):
        import socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (gimbal_ip, gimbal_port)
        # перевести гимбал в режим follow (body-frame) — иначе YPR может игнорироваться
        for _i in range(3):
            self._sock.sendto(_topotek_mode_frame(7), self._addr)
        self.stop()

    def set_follow(self):
        self._sock.sendto(_topotek_mode_frame(7), self._addr)

    def apply_rate(self, yaw, pitch, roll=0):
        self._sock.sendto(_topotek_ypr_frame(int(yaw), int(pitch), int(roll)), self._addr)

    def apply_dir(self, code):
        """Прямое направление: 1=вверх,2=вниз,3=влево,4=вправо."""
        self._sock.sendto(_topotek_mode_frame(int(code)), self._addr)

    def stop(self):
        try: self._sock.sendto(_topotek_stop_frame(), self._addr)
        except Exception: pass

    def close(self):
        self.stop()
        try: self._sock.close()
        except Exception: pass


class _TopotekGS(_Backend):
    """Гимбал Topotek через протокол GroundStation: джойстик по UDP, порт 2337.
    Команды скорости pan/tilt в бинарном формате a9d7 с CRC16-XMODEM."""
    name = 'topogs'
    rate_based = True

    def __init__(self, gimbal_ip='192.168.144.108', gimbal_port=2337, **_):
        import socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (gimbal_ip, gimbal_port)
        self.stop()

    def apply_rate(self, yaw, pitch, roll=0):
        # СВОП: байты7-8 = вертикаль(tilt), байты9-10 = горизонталь(pan)
        self._sock.sendto(_topogs_packet(int(pitch), int(yaw), True), self._addr)

    def stop(self):
        try:
            # явная нулевая скорость с флагом 04 = СТОП (флаг 00 гимбал не тормозит)
            self._sock.sendto(_topogs_packet(0, 0, True), self._addr)
        except Exception:
            pass

    def close(self):
        self.stop()
        try:
            self._sock.close()
        except Exception:
            pass


class _Sim(_Backend):
    name = 'sim'

    def apply(self, pan, tilt):
        pass

    def apply_rate(self, yaw, pitch):
        pass


def _make_backend(name, **kw):
    order = [name] if name != 'auto' else ['pantilthat', 'servokit', 'gpiozero', 'sim']
    for b in order:
        try:
            if b == 'pantilthat': return _PanTiltHat(**kw)
            if b == 'servokit':   return _ServoKit(**kw)
            if b == 'gpiozero':   return _GpioZero(**kw)
            if b == 'topotek':    return _TopotekNet(**kw)
            if b == 'topogs':     return _TopotekGS(**kw)
            if b == 'sim':        return _Sim()
        except Exception as exc:
            if name != 'auto':
                sys.exit(f'Привод "{name}" недоступен: {exc}')
    return _Sim()


# ── Трекер ────────────────────────────────────────────────────────────────────────
class PanTiltTracker:
    """
    Держит цель в центре кадра.
      update(target_center, frame_size) — каждый кадр; target_center=(cx,cy) или None.
    Сервоприводы управляются по углу, гимбал Topotek — по скорости.
    """

    def __init__(self, backend='auto', *, pan_pin=17, tilt_pin=18,
                 gimbal_ip='192.168.144.108', gimbal_port=9003,
                 gain=18.0, rate_gain=60.0, min_rate=7.0, deadzone=0.06, smooth=0.5,
                 ctrl='rate', invert_pan=False, invert_tilt=False,
                 track_pan=True, track_tilt=True, verbose=False):
        self._be = _make_backend(backend, pan_pin=pan_pin, tilt_pin=tilt_pin,
                                 gimbal_ip=gimbal_ip, gimbal_port=gimbal_port)
        self.backend_name = self._be.name
        self.rate_based = getattr(self._be, 'rate_based', False)
        self.pan = 0.0
        self.tilt = 0.0
        self.gain = gain              # угол: доворот за кадр (градусы)
        self.rate_gain = rate_gain    # скорость: масштаб (полная ошибка → rate_gain)
        self.min_rate = min_rate      # мин. скорость, чтобы мотор тронулся (преодолеть порог)
        self.max_rate = 660 if self.backend_name == 'topogs' else 99  # потолок скорости
        self.deadzone = deadzone
        self.smooth = smooth
        self.ctrl = ctrl              # 'rate' (YPR-скорость) или 'dir' (команды направления)
        self.invert_pan = invert_pan
        self.invert_tilt = invert_tilt
        self.track_pan = track_pan      # вести по горизонтали
        self.track_tilt = track_tilt    # вести по вертикали
        self.verbose = verbose and self._be.name in ('sim', 'topotek', 'topogs')
        self._last_send = 0.0
        self._stopped = True
        self._cur_dir = 0
        if not self.rate_based:
            self._be.apply(self.pan, self.tilt)
        mode = 'скорость (rate)' if self.rate_based else 'угол'
        extra = f'  {gimbal_ip}:{gimbal_port}' if self.backend_name == 'topotek' else ''
        print(f'[tracker] Привод: {self.backend_name}, режим: {mode}{extra}'
              + ('  (симуляция)' if self.backend_name == 'sim' else ''))

    def update(self, target_center, frame_size):
        """Подвести камеру к цели. Возвращает True, если двигались."""
        if target_center is None:
            if self.rate_based:
                now = time.time()
                if now - self._last_send >= 0.05:   # непрерывно шлём нейтраль (стоп)
                    self._last_send = now
                    self._be.stop()
            return False

        cx, cy = target_center
        w, h = frame_size
        ex = (cx - w / 2) / (w / 2)   # -1 (лево) .. +1 (право)
        ey = (cy - h / 2) / (h / 2)   # -1 (верх) .. +1 (низ)

        if self.rate_based:
            if self.ctrl == 'dir':
                return self._update_dir(ex, ey)
            return self._update_rate(ex, ey)
        return self._update_angle(ex, ey)

    # ── режим направления (прямые команды Topotek влево/вправо/вверх/вниз) ────────
    def _update_dir(self, ex, ey):
        inx = -1 if self.invert_pan else 1
        iny = -1 if self.invert_tilt else 1
        ax, ay = ex * inx, ey * iny
        if abs(ax) < self.deadzone and abs(ay) < self.deadzone:
            code = 0                                  # в центре → стоп
        elif abs(ax) >= abs(ay):
            code = 3 if ax < 0 else 4                 # влево / вправо
        else:
            code = 1 if ay < 0 else 2                 # вверх / вниз

        now = time.time()
        # шлём при смене направления или периодически (вдруг гимбал по таймауту встал)
        if code != self._cur_dir or now - self._last_send > 0.3:
            if code == 0:
                self._be.stop()
                self._stopped = True
            else:
                self._be.apply_dir(code)
                self._stopped = False
            self._cur_dir = code
            self._last_send = now
            if self.verbose:
                names = {0: 'стоп', 1: 'вверх', 2: 'вниз', 3: 'влево', 4: 'вправо'}
                print(f'[tracker] {names[code]}  (ошибка x={ex:+.2f} y={ey:+.2f})', flush=True)
        return code != 0

    # ── режим скорости (гимбал Topotek) ──────────────────────────────────────────
    def _update_rate(self, ex, ey):
        def rate(err, invert):
            if abs(err) < self.deadzone:
                return 0.0
            r = self.rate_gain * err * (1 if not invert else -1)
            # минимальная скорость, чтобы преодолеть порог трогания мотора
            if 0 < abs(r) < self.min_rate:
                r = self.min_rate if r > 0 else -self.min_rate
            return _clamp(r, -self.max_rate, self.max_rate)
        yaw = rate(ex, self.invert_pan) if self.track_pan else 0.0
        pitch = rate(ey, self.invert_tilt) if self.track_tilt else 0.0

        now = time.time()
        if now - self._last_send < 0.05:     # не чаще ~20 Гц
            return yaw != 0 or pitch != 0
        self._last_send = now

        if yaw == 0 and pitch == 0:
            self._be.stop()                  # непрерывно шлём нейтраль
            return False
        self._be.apply_rate(yaw, pitch)
        if self.verbose:
            print(f'[tracker] yaw_rate={yaw:+.0f} pitch_rate={pitch:+.0f}'
                  f'  (ошибка x={ex:+.2f} y={ey:+.2f})', flush=True)
        return True

    # ── режим угла (сервоприводы) ────────────────────────────────────────────────
    def _update_angle(self, ex, ey):
        moved = False
        if abs(ex) > self.deadzone:
            d = self.gain * ex * (1 if not self.invert_pan else -1)
            self.pan = _clamp(self.pan + self.smooth * d, -90, 90)
            moved = True
        if abs(ey) > self.deadzone:
            d = self.gain * ey * (1 if not self.invert_tilt else -1)
            self.tilt = _clamp(self.tilt - self.smooth * d, -90, 90)
            moved = True
        if moved:
            self._be.apply(self.pan, self.tilt)
            if self.verbose:
                print(f'[tracker] pan={self.pan:+.0f}° tilt={self.tilt:+.0f}°'
                      f'  (ошибка x={ex:+.2f} y={ey:+.2f})', flush=True)
        return moved

    def center(self):
        if self.rate_based:
            self._be.stop()
        else:
            self.pan = self.tilt = 0.0
            self._be.apply(0, 0)

    def close(self):
        self._be.close()
