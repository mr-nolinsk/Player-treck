bl_info = {
    "name": "Body Tracker (Webcam Motion Capture)",
    "author": "Claude",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Body Tracker",
    "description": "Отслеживает человека через веб-камеру (MediaPipe Pose) и в реальном времени "
                   "поворачивает кости выбранного рига (голова, руки, ноги, спина) повторяя за ним.",
    "category": "Animation",
}

import bpy
import sys
import subprocess
import threading
import queue
import time
from mathutils import Vector, Quaternion, Matrix

# ---------------------------------------------------------------------------
# Проверка/установка зависимостей (opencv-python, mediapipe) в python Blender'а
# ---------------------------------------------------------------------------

def get_blender_python():
    return sys.executable


def dependencies_available():
    try:
        import cv2  # noqa
        import mediapipe  # noqa
        return True
    except Exception:
        return False


def open_camera(index):
    """Пытается открыть камеру наиболее надёжным способом для текущей ОС.
    На Windows разные камеры "любят" разные бэкенды: часть веб-камер открывается
    только через DirectShow, часть (особенно встроенные на Windows 11 с
    Windows Studio Effects) — только через Media Foundation (MSMF).

    Отдельная частая проблема — картинка вида "телевизионные помехи": так
    выглядит поток, если OpenCV декодирует его в неправильном формате пикселей.
    Обычно помогает явно запросить кодек MJPG вместо формата по умолчанию —
    пробуем это первым делом, и только если не получилось, откатываемся."""
    import cv2

    def try_open(backend, force_mjpg):
        if backend is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        if force_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # "прогреваем" — первые пара кадров после смены формата часто пустые/битые
        ok, frame = False, None
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        if ok and frame is not None:
            return cap
        cap.release()
        return None

    if sys.platform.startswith("win"):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            for force_mjpg in (False, True):
                cap = try_open(backend, force_mjpg)
                if cap is not None:
                    return cap
        return cv2.VideoCapture(index)
    else:
        cap = try_open(None, True)
        if cap is not None:
            return cap
        return cv2.VideoCapture(index)


def get_windows_camera_names():
    """Возвращает список настоящих имён камер (например, 'OBS Virtual Camera',
    'Integrated Webcam') в том же порядке, в котором их видит DirectShow —
    том же, что используют индексы в cv2.VideoCapture(index, cv2.CAP_DSHOW).
    Требует необязательный пакет pygrabber. Возвращает None, если недоступно."""
    if not sys.platform.startswith("win"):
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return None


_VIRTUAL_CAMERA_NAME_HINTS = (
    "virtual", "obs", "droidcam", "camo", "manycam", "snap camera",
    "iriun", "epoccam", "ivcam", "ndi", "streamlabs", "xsplit",
    "elgato", "reincubate", "vysor", "spacedesk",
)


def looks_like_virtual_camera(name):
    if not name:
        return False
    lowered = name.lower()
    return any(hint in lowered for hint in _VIRTUAL_CAMERA_NAME_HINTS)


# ---------------------------------------------------------------------------
# Модель PoseLandmarker (новый MediaPipe Tasks API).
# Начиная с mediapipe 0.10.30 старый mp.solutions.pose удалён из библиотеки,
# поэтому используем новый API, которому нужен отдельный файл модели (.task).
# ---------------------------------------------------------------------------

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
MODEL_FILENAME = "pose_landmarker_lite.task"


def get_model_dir():
    path = bpy.utils.user_resource('SCRIPTS', path="body_tracker_data", create=True)
    return path


def get_model_path():
    return __import__("os").path.join(get_model_dir(), MODEL_FILENAME)


def model_available():
    import os
    return os.path.isfile(get_model_path()) and os.path.getsize(get_model_path()) > 0


class BODYTRACKER_OT_download_model(bpy.types.Operator):
    bl_idname = "bodytracker.download_model"
    bl_label = "Скачать модель распознавания позы"
    bl_description = "Скачивает файл модели pose_landmarker (~5-30 МБ), нужен интернет"

    def execute(self, context):
        import urllib.request
        import os

        dest = get_model_path()
        try:
            self.report({'INFO'}, "Скачиваю модель...")
            urllib.request.urlretrieve(MODEL_URL, dest)
        except Exception as e:
            self.report({'ERROR'}, f"Не удалось скачать модель: {e}. "
                                    f"Скачайте вручную по ссылке {MODEL_URL} "
                                    f"и положите файл сюда: {dest}")
            return {'CANCELLED'}

        if model_available():
            self.report({'INFO'}, "Модель успешно скачана!")
        else:
            self.report({'ERROR'}, "Файл модели не найден после скачивания")
            return {'CANCELLED'}
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Список найденных камер (кэш, заполняется по кнопке "Обновить список камер")
# ---------------------------------------------------------------------------

_camera_cache = []  # список кортежей (index:int, label:str)


def get_camera_enum_items(self, context):
    if not _camera_cache:
        return [("0", "Камера 0 (по умолчанию, нажмите 'Обновить список')", "")]
    return [(str(idx), label, "") for idx, label in _camera_cache]


class BODYTRACKER_OT_scan_cameras(bpy.types.Operator):
    bl_idname = "bodytracker.scan_cameras"
    bl_label = "Обновить список камер"
    bl_description = "Проверяет индексы 0-14 и находит все подключённые камеры"

    def execute(self, context):
        if not dependencies_available():
            self.report({'ERROR'}, "Сначала установите зависимости")
            return {'CANCELLED'}

        names = get_windows_camera_names()  # None, если недоступно (не Windows / нет pygrabber)

        found = []
        for index in range(15):
            cap = open_camera(index)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    if names and index < len(names):
                        device_name = names[index]
                    else:
                        device_name = f"Камера {index}"
                    suffix = " ⚠ вероятно виртуальная" if looks_like_virtual_camera(device_name) else ""
                    label = f"{device_name} ({w}x{h}){suffix}"
                    found.append((index, label))
            cap.release()

        _camera_cache.clear()
        _camera_cache.extend(found)

        if not found:
            self.report({'WARNING'}, "Камеры не найдены")
        elif names is None and sys.platform.startswith("win"):
            self.report({'INFO'}, f"Найдено камер: {len(found)}. Совет: установите пакет "
                                   f"'pygrabber' (см. панель), чтобы видеть настоящие имена "
                                   f"устройств и не путать реальную камеру с виртуальной")
        else:
            self.report({'INFO'}, f"Найдено камер: {len(found)}")

        for area in context.screen.areas:
            area.tag_redraw()

        return {'FINISHED'}


class BODYTRACKER_OT_install_pygrabber(bpy.types.Operator):
    bl_idname = "bodytracker.install_pygrabber"
    bl_label = "Показывать настоящие имена камер (pygrabber)"
    bl_description = ("Ставит необязательный пакет pygrabber (только Windows), чтобы список "
                       "камер показывал настоящие названия устройств вместо номеров — "
                       "так виртуальные камеры (OBS, Zoom и т.д.) видно сразу")

    def execute(self, context):
        py = get_blender_python()
        try:
            subprocess.check_call([py, "-m", "pip", "install", "pygrabber"])
        except Exception as e:
            self.report({'ERROR'}, f"Не удалось установить pygrabber: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, "pygrabber установлен — нажмите 'Обновить список камер'")
        return {'FINISHED'}


class BODYTRACKER_OT_install_deps(bpy.types.Operator):
    bl_idname = "bodytracker.install_deps"
    bl_label = "Установить зависимости (opencv-python, mediapipe)"
    bl_description = ("Скачивает и устанавливает opencv-python и mediapipe в python, "
                       "встроенный в Blender. Нужен интернет и права на запись "
                       "в папку Blender (иногда нужен запуск Blender от администратора)")

    def execute(self, context):
        py = get_blender_python()
        try:
            subprocess.check_call([py, "-m", "ensurepip"])
            subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([py, "-m", "pip", "install", "opencv-python"])
            subprocess.check_call([py, "-m", "pip", "install", "mediapipe"])
        except Exception as e:
            self.report({'ERROR'}, f"Не удалось установить зависимости: {e}")
            return {'CANCELLED'}

        if dependencies_available():
            self.report({'INFO'}, "Зависимости успешно установлены!")
        else:
            self.report({'WARNING'}, "Установка прошла, но модули всё ещё не импортируются. "
                                      "Перезапустите Blender.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Индексы точек MediaPipe Pose, которые нам нужны
# ---------------------------------------------------------------------------
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LM = {
    "nose": 0,
    "ear_l": 7, "ear_r": 8,
    "shoulder_l": 11, "shoulder_r": 12,
    "elbow_l": 13, "elbow_r": 14,
    "wrist_l": 15, "wrist_r": 16,
    "index_l": 19, "index_r": 20,
    "hip_l": 23, "hip_r": 24,
    "knee_l": 25, "knee_r": 26,
    "ankle_l": 27, "ankle_r": 28,
    "foot_l": 31, "foot_r": 32,
}

# Простые (не составные) кости — голова и позвоночник.
# Формат: bone_prop_name: (landmark_start, landmark_end)
SIMPLE_BONE_PAIRS = {
    "spine_bone":      ("hips_center", "shoulders_center"),
    # Голова: от центра плеч до центра ушей (а не до носа) — нос физически
    # выступает вперёд лица, из-за чего голова получала постоянный наклон вниз/вперёд.
    "head_bone":       ("shoulders_center", "ears_center"),
}

# Составные цепочки (рука, нога). Если в настройках не указана какая-то
# промежуточная кость (например, нет отдельного предплечья/голени),
# направление автоматически "дотягивается" до следующей заданной кости —
# так рука/нога не перестаёт полноценно отслеживаться, а просто одна кость
# охватывает больший участок (например, плечо сразу до кисти).
# "bones" — кости по порядку от корня цепочки к кончику,
# "joints" — landmark-точка начала соответствующей кости,
# "tip" — landmark-точка кончика цепочки (кисть/стопа), если это последняя заданная кость.
CHAINS = {
    "arm_l": {
        "bones":  ["upper_arm_l_bone", "forearm_l_bone", "hand_l_bone"],
        "joints": ["shoulder_l", "elbow_l", "wrist_l"],
        "tip": "index_l",
    },
    "arm_r": {
        "bones":  ["upper_arm_r_bone", "forearm_r_bone", "hand_r_bone"],
        "joints": ["shoulder_r", "elbow_r", "wrist_r"],
        "tip": "index_r",
    },
    "leg_l": {
        "bones":  ["thigh_l_bone", "shin_l_bone", "foot_l_bone"],
        "joints": ["hip_l", "knee_l", "ankle_l"],
        "tip": "foot_l",
    },
    "leg_r": {
        "bones":  ["thigh_r_bone", "shin_r_bone", "foot_r_bone"],
        "joints": ["hip_r", "knee_r", "ankle_r"],
        "tip": "foot_r",
    },
}

# При включённом "зеркале" левая/правая сторона тела пользователя меняются местами
# (как в зеркале) — так персонаж двигает "своей" левой рукой, когда пользователь
# поднимает свою правую руку перед камерой.
MIRROR_SWAP = str.maketrans({})  # placeholder, real swap done in code below


def swap_lr(key):
    if key.endswith("_l"):
        return key[:-2] + "_r"
    if key.endswith("_r"):
        return key[:-2] + "_l"
    return key


# ---------------------------------------------------------------------------
# Настройки (Property Group)
# ---------------------------------------------------------------------------

class BODYTRACKER_bone_settings(bpy.types.PropertyGroup):
    head_bone: bpy.props.StringProperty(name="Голова")
    spine_bone: bpy.props.StringProperty(name="Позвоночник")
    upper_arm_l_bone: bpy.props.StringProperty(name="Плечо (лев.)")
    forearm_l_bone: bpy.props.StringProperty(name="Предплечье (лев.)")
    hand_l_bone: bpy.props.StringProperty(name="Кисть (лев.)")
    upper_arm_r_bone: bpy.props.StringProperty(name="Плечо (прав.)")
    forearm_r_bone: bpy.props.StringProperty(name="Предплечье (прав.)")
    hand_r_bone: bpy.props.StringProperty(name="Кисть (прав.)")
    thigh_l_bone: bpy.props.StringProperty(name="Бедро (лев.)")
    shin_l_bone: bpy.props.StringProperty(name="Голень (лев.)")
    foot_l_bone: bpy.props.StringProperty(name="Стопа (лев.)")
    thigh_r_bone: bpy.props.StringProperty(name="Бедро (прав.)")
    shin_r_bone: bpy.props.StringProperty(name="Голень (прав.)")
    foot_r_bone: bpy.props.StringProperty(name="Стопа (прав.)")


class BODYTRACKER_settings(bpy.types.PropertyGroup):
    armature: bpy.props.PointerProperty(name="Риг", type=bpy.types.Object)

    input_mode: bpy.props.EnumProperty(
        name="Источник",
        items=[
            ('WEBCAM', "Веб-камера", "Отслеживание в реальном времени с камеры"),
            ('VIDEO_FILE', "Видео-файл", "Извлечь анимацию из заранее записанного видео"),
        ],
        default='WEBCAM',
    )

    camera_index: bpy.props.EnumProperty(name="Камера", items=get_camera_enum_items)
    mirror: bpy.props.BoolProperty(name="Зеркалить", default=True,
                                    description="Левая часть тела пользователя управляет правой частью персонажа (как в зеркале)")
    smoothing: bpy.props.FloatProperty(name="Сглаживание", default=0.7, min=0.0, max=0.95)
    record_keyframes: bpy.props.BoolProperty(name="Записывать ключевые кадры", default=False)
    keyframe_step: bpy.props.IntProperty(name="Шаг кадра", default=1, min=1)
    show_camera_preview: bpy.props.BoolProperty(name="Показывать окно камеры", default=True)
    is_running: bpy.props.BoolProperty(default=False)

    # --- Настройки режима "Видео-файл" ---
    video_path: bpy.props.StringProperty(
        name="Файл видео", subtype='FILE_PATH',
        description="Видео с человеком, из которого нужно извлечь анимацию")
    video_start_frame: bpy.props.IntProperty(
        name="Начальный кадр сцены", default=1, min=1,
        description="С какого кадра сцены Blender начнётся запечённая анимация")
    video_frame_skip: bpy.props.IntProperty(
        name="Обрабатывать каждый N-й кадр", default=1, min=1,
        description="1 = каждый кадр видео (точнее, но дольше). Больше — быстрее, но грубее")
    video_match_speed: bpy.props.BoolProperty(
        name="Сохранять реальную скорость",
        default=True,
        description="Учитывать разницу FPS видео и сцены, чтобы анимация не ускорялась/замедлялась. "
                    "Если выключено — 1 обработанный кадр видео = 1 кадр сцены")
    is_baking: bpy.props.BoolProperty(default=False)


# ---------------------------------------------------------------------------
# Поток захвата камеры + распознавания позы (не трогает bpy!)
# ---------------------------------------------------------------------------

def smooth_landmarks(prev, new, alpha=0.4):
    """Экспоненциальное сглаживание координат точек тела между кадрами.
    Сглаживание 'на входе' (до вычисления направлений костей) убирает дрожание
    камеры гораздо эффективнее, чем сглаживание только финальных поворотов —
    шум не успевает накопиться по цепочке костей."""
    if prev is None or len(prev) != len(new):
        return new
    result = []
    for (ox, oy, oz, ov), (nx, ny, nz, nv) in zip(prev, new):
        result.append((
            ox + (nx - ox) * alpha,
            oy + (ny - oy) * alpha,
            oz + (nz - oz) * alpha,
            nv,
        ))
    return result


class CaptureThread(threading.Thread):
    def __init__(self, cam_index, result_queue, show_preview):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.result_queue = result_queue
        self.show_preview = show_preview
        self._stop_event = threading.Event()
        self._smoothed_landmarks = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python import BaseOptions

        cap = open_camera(self.cam_index)
        if not cap.isOpened():
            self.result_queue.put({"error": f"Не удалось открыть камеру #{self.cam_index}"})
            return

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=get_model_path()),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        start_time = time.monotonic()
        last_ts = -1

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            if timestamp_ms <= last_ts:
                timestamp_ms = last_ts + 1
            last_ts = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            landmarks = None
            if result.pose_world_landmarks:
                pose0 = result.pose_world_landmarks[0]
                raw_landmarks = [(lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0) or 1.0)
                                  for lm in pose0]
                self._smoothed_landmarks = smooth_landmarks(self._smoothed_landmarks, raw_landmarks)
                landmarks = self._smoothed_landmarks

            if self.show_preview:
                if result.pose_landmarks:
                    h, w = frame.shape[:2]
                    for lm in result.pose_landmarks[0]:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
                cv2.imshow("Body Tracker - камера (Q чтобы закрыть только окно)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.show_preview = False
                    cv2.destroyWindow("Body Tracker - камера (Q чтобы закрыть только окно)")

            if landmarks is not None:
                # оставляем в очереди только самый свежий результат
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        break
                self.result_queue.put({"landmarks": landmarks})

        landmarker.close()
        cap.release()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Вспомогательные функции для вычисления направлений и вращений костей
# ---------------------------------------------------------------------------

def mp_to_blender_vec(x, y, z):
    """Переводит координаты MediaPipe (world landmarks) в систему координат Blender.
    MediaPipe: x вправо, y вниз, z к камере (примерно).
    Blender: x вправо, y вперёд (от камеры), z вверх."""
    return Vector((x, z, -y))


def get_point(landmarks, key, mirror):
    real_key = swap_lr(key) if mirror else key
    idx = LM.get(real_key)
    if idx is None:
        return None
    x, y, z, vis = landmarks[idx]
    if vis < 0.3:
        return None
    v = mp_to_blender_vec(x, y, z)
    if mirror:
        v.x = -v.x
    return v


def get_direction(landmarks, start_key, end_key, mirror):
    def resolve(key):
        if key == "hips_center":
            l = get_point(landmarks, "hip_l", mirror)
            r = get_point(landmarks, "hip_r", mirror)
            return None if (l is None or r is None) else (l + r) / 2
        if key == "shoulders_center":
            l = get_point(landmarks, "shoulder_l", mirror)
            r = get_point(landmarks, "shoulder_r", mirror)
            return None if (l is None or r is None) else (l + r) / 2
        if key == "ears_center":
            l = get_point(landmarks, "ear_l", mirror)
            r = get_point(landmarks, "ear_r", mirror)
            if l is None or r is None:
                # если уши не видны (например, в профиль) — подстраховка носом
                return get_point(landmarks, "nose", mirror)
            return (l + r) / 2
        return get_point(landmarks, key, mirror)

    p1 = resolve(start_key)
    p2 = resolve(end_key)
    if p1 is None or p2 is None:
        return None
    d = p2 - p1
    if d.length < 1e-6:
        return None
    return d.normalized()


def stable_rotation_between(v1, v2):
    """Кватернион кратчайшего поворота из v1 в v2 (оба должны быть единичными).
    Встроенный Vector.rotation_difference() численно нестабилен, когда векторы
    почти противоположны (угол около 180°) — именно такой случай возникает,
    когда рука в покое направлена вниз (как у стоящего Minecraft-персонажа),
    а на видео поднимается почти строго вверх. В этой зоне у обычной формулы
    "почти нулевой" числитель/знаменатель, и результат может быть случайным —
    из-за этого рука визуально не долетает до конца или ведёт себя рывками."""
    d = max(-1.0, min(1.0, v1.dot(v2)))

    if d > 0.9999995:
        return Quaternion((1, 0, 0, 0))

    if d < -0.9999995:
        # Почти точно противоположные направления — ось вращения не определена
        # однозначно, берём любую ось, перпендикулярную v1.
        axis = v1.cross(Vector((1.0, 0.0, 0.0)))
        if axis.length < 1e-6:
            axis = v1.cross(Vector((0.0, 1.0, 0.0)))
        axis.normalize()
        return Quaternion(axis, 3.14159265358979)

    # Численно устойчивая формула (без деления на sin(угла), которое обнуляется
    # у обычной acos+cross формулы при приближении к 180°).
    s = ((1.0 + d) * 2.0) ** 0.5
    invs = 1.0 / s
    c = v1.cross(v2) * invs
    quat = Quaternion((s * 0.5, c.x, c.y, c.z))
    quat.normalize()
    return quat


def apply_bone_direction(pose_bone, target_dir_world, armature_obj, smoothing):
    """Поворачивает pose_bone так, чтобы направление его локальной оси Y (head->tail)
    совпадало с target_dir_world (в мировых координатах сцены)."""
    if pose_bone.rotation_mode != 'QUATERNION':
        pose_bone.rotation_mode = 'QUATERNION'

    prev_quat = pose_bone.rotation_quaternion.copy()

    # Сброс до "нейтрального" состояния, чтобы получить базовую (armature-space) матрицу
    pose_bone.rotation_quaternion = Quaternion((1, 0, 0, 0))
    bpy.context.view_layer.update()

    mat_world_neutral = armature_obj.matrix_world @ pose_bone.matrix
    neutral_rot = mat_world_neutral.to_quaternion()
    neutral_dir = (neutral_rot @ Vector((0, 1, 0))).normalized()

    q_world = stable_rotation_between(neutral_dir, target_dir_world)

    local_q = neutral_rot.inverted() @ q_world @ neutral_rot

    target_quat = local_q
    if smoothing > 0.0:
        target_quat = prev_quat.slerp(local_q, 1.0 - smoothing)

    pose_bone.rotation_quaternion = target_quat


def _bone_depth(pose_bone):
    """Сколько родителей у кости — 0 для корневой кости."""
    depth = 0
    b = pose_bone.parent
    while b is not None:
        depth += 1
        b = b.parent
    return depth


def apply_landmarks_to_armature(arm, bones_cfg, landmarks, mirror, smoothing, insert_keyframe=False):
    """Применяет один кадр landmarks к костям рига. Если insert_keyframe=True,
    сразу вставляет ключевые кадры rotation_quaternion на текущем кадре сцены.

    Кости обрабатываются от корня к потомкам (по фактической иерархии рига),
    а не в произвольном порядке — иначе, например, кость руки может
    пересчитаться раньше родительской кости позвоночника и получить
    рассинхронизированный (дёрганый) результат на этом кадре.

    Составные цепочки (рука/нога) автоматически сжимаются, если каких-то
    промежуточных костей нет в риге (например, нет отдельного предплечья) —
    тогда одна кость просто "дотягивается" до следующей заданной."""

    entries = []

    # Простые кости: голова, позвоночник
    for prop_name, (start_key, end_key) in SIMPLE_BONE_PAIRS.items():
        bone_name = getattr(bones_cfg, prop_name)
        if not bone_name:
            continue
        pose_bone = arm.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        entries.append((pose_bone, start_key, end_key))

    # Цепочки: рука, нога — с автоматическим сжатием при пропущенных костях
    for chain in CHAINS.values():
        assigned = []
        for bone_prop, joint_key in zip(chain["bones"], chain["joints"]):
            bone_name = getattr(bones_cfg, bone_prop)
            if not bone_name:
                continue
            pose_bone = arm.pose.bones.get(bone_name)
            if pose_bone is None:
                continue
            assigned.append((pose_bone, joint_key))

        for i, (pose_bone, start_key) in enumerate(assigned):
            end_key = assigned[i + 1][1] if (i + 1) < len(assigned) else chain["tip"]
            entries.append((pose_bone, start_key, end_key))

    entries.sort(key=lambda e: _bone_depth(e[0]))

    for pose_bone, start_key, end_key in entries:
        direction = get_direction(landmarks, start_key, end_key, mirror)
        if direction is None:
            continue
        direction_world = direction.normalized()
        apply_bone_direction(pose_bone, direction_world, arm, smoothing)

        if insert_keyframe:
            pose_bone.keyframe_insert(data_path="rotation_quaternion")


# ---------------------------------------------------------------------------
# Модальный оператор запуска трекинга
# ---------------------------------------------------------------------------

class BODYTRACKER_OT_start(bpy.types.Operator):
    bl_idname = "bodytracker.start"
    bl_label = "Начать отслеживание"

    _timer = None
    _thread = None
    _queue = None

    def modal(self, context, event):
        settings = context.scene.body_tracker_settings

        if not settings.is_running:
            return self.cancel(context)

        if event.type == 'TIMER':
            data = None
            while not self._queue.empty():
                try:
                    data = self._queue.get_nowait()
                except queue.Empty:
                    break

            if data:
                if "error" in data:
                    self.report({'ERROR'}, data["error"])
                    settings.is_running = False
                    return self.cancel(context)

                self.apply_pose(context, settings, data["landmarks"])

        return {'PASS_THROUGH'}

    def apply_pose(self, context, settings, landmarks):
        arm = settings.armature
        if arm is None or arm.type != 'ARMATURE':
            return
        bones_cfg = context.scene.body_tracker_bones

        should_key = (settings.record_keyframes and
                      context.scene.frame_current % settings.keyframe_step == 0)

        apply_landmarks_to_armature(arm, bones_cfg, landmarks, settings.mirror,
                                     settings.smoothing, insert_keyframe=should_key)

        if should_key:
            context.scene.frame_set(context.scene.frame_current + 1)

    def execute(self, context):
        if not dependencies_available():
            self.report({'ERROR'}, "Сначала установите зависимости (кнопка выше в панели)")
            return {'CANCELLED'}
        if not model_available():
            self.report({'ERROR'}, "Сначала скачайте модель распознавания позы (кнопка в панели)")
            return {'CANCELLED'}

        settings = context.scene.body_tracker_settings
        if settings.armature is None:
            self.report({'ERROR'}, "Выберите риг (Armature) в настройках")
            return {'CANCELLED'}

        self._queue = queue.Queue(maxsize=2)
        self._thread = CaptureThread(int(settings.camera_index), self._queue, settings.show_camera_preview)
        self._thread.start()

        settings.is_running = True
        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / 30.0, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        if self._thread:
            self._thread.stop()
            self._thread.join(timeout=2.0)
            self._thread = None
        context.scene.body_tracker_settings.is_running = False
        return {'CANCELLED'}


class BODYTRACKER_OT_stop(bpy.types.Operator):
    bl_idname = "bodytracker.stop"
    bl_label = "Остановить отслеживание"

    def execute(self, context):
        context.scene.body_tracker_settings.is_running = False
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Оператор: запечь анимацию из видео-файла
# ---------------------------------------------------------------------------

class BODYTRACKER_OT_bake_video(bpy.types.Operator):
    bl_idname = "bodytracker.bake_video"
    bl_label = "Запечь анимацию из видео"
    bl_description = "Покадрово обрабатывает видео и записывает ключевые кадры на риг (ESC для отмены)"

    _timer = None
    cap = None
    pose_ctx = None
    frame_counter = 0
    total_frames = 0
    video_fps = 30.0
    scene_fps = 30.0
    start_frame = 1
    frame_skip = 1
    _smoothed_landmarks = None

    def modal(self, context, event):
        settings = context.scene.body_tracker_settings

        if not settings.is_baking:
            return self.finish(context, cancelled=True)

        if event.type == 'ESC':
            self.report({'INFO'}, "Запекание отменено пользователем")
            return self.finish(context, cancelled=True)

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        ok, frame = self.cap.read()
        if not ok:
            self.report({'INFO'}, f"Готово! Обработано кадров видео: {self.frame_counter}")
            return self.finish(context, cancelled=False)

        self.frame_counter += 1

        if self.frame_counter % self.frame_skip == 0:
            import cv2
            import mediapipe as mp

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((self.frame_counter - 1) / self.video_fps * 1000)
            if timestamp_ms <= self.last_ts:
                timestamp_ms = self.last_ts + 1
            self.last_ts = timestamp_ms

            result = self.pose_ctx.detect_for_video(mp_image, timestamp_ms)

            if result.pose_world_landmarks:
                pose0 = result.pose_world_landmarks[0]
                raw_landmarks = [(lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0) or 1.0)
                                  for lm in pose0]
                self._smoothed_landmarks = smooth_landmarks(self._smoothed_landmarks, raw_landmarks)
                landmarks = self._smoothed_landmarks

                if settings.video_match_speed:
                    time_seconds = (self.frame_counter - 1) / self.video_fps
                    scene_frame = self.start_frame + round(time_seconds * self.scene_fps)
                else:
                    scene_frame = self.start_frame + (self.frame_counter - 1)

                context.scene.frame_set(scene_frame)

                arm = settings.armature
                bones_cfg = context.scene.body_tracker_bones
                apply_landmarks_to_armature(arm, bones_cfg, landmarks, settings.mirror,
                                             settings.smoothing, insert_keyframe=True)

        if self.total_frames:
            pct = int(100 * self.frame_counter / self.total_frames)
            context.workspace.status_text_set(
                f"Body Tracker: обработка видео {pct}% ({self.frame_counter}/{self.total_frames}) — ESC для отмены")

        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not dependencies_available():
            self.report({'ERROR'}, "Сначала установите зависимости")
            return {'CANCELLED'}
        if not model_available():
            self.report({'ERROR'}, "Сначала скачайте модель распознавания позы (кнопка в панели)")
            return {'CANCELLED'}

        settings = context.scene.body_tracker_settings
        if settings.armature is None:
            self.report({'ERROR'}, "Выберите риг (Armature) в настройках")
            return {'CANCELLED'}
        if not settings.video_path:
            self.report({'ERROR'}, "Укажите путь к видео-файлу")
            return {'CANCELLED'}

        video_path = bpy.path.abspath(settings.video_path)

        import cv2

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.report({'ERROR'}, f"Не удалось открыть видео: {video_path}")
            return {'CANCELLED'}

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.scene_fps = context.scene.render.fps / context.scene.render.fps_base
        self.start_frame = settings.video_start_frame
        self.frame_skip = max(1, settings.video_frame_skip)
        self.frame_counter = 0
        self.last_ts = -1
        self._smoothed_landmarks = None

        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python import BaseOptions

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=get_model_path()),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.pose_ctx = mp_vision.PoseLandmarker.create_from_options(options)

        settings.is_baking = True
        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / 1000.0, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self, context, cancelled):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.pose_ctx:
            self.pose_ctx.close()
            self.pose_ctx = None
        context.workspace.status_text_set(None)
        context.scene.body_tracker_settings.is_baking = False
        return {'CANCELLED'} if cancelled else {'FINISHED'}


class BODYTRACKER_OT_stop_bake(bpy.types.Operator):
    bl_idname = "bodytracker.stop_bake"
    bl_label = "Остановить запекание"

    def execute(self, context):
        context.scene.body_tracker_settings.is_baking = False
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# UI панель
# ---------------------------------------------------------------------------

class BODYTRACKER_PT_panel(bpy.types.Panel):
    bl_label = "Body Tracker"
    bl_idname = "BODYTRACKER_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Body Tracker"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.body_tracker_settings
        bones_cfg = context.scene.body_tracker_bones

        if not dependencies_available():
            box = layout.box()
            box.label(text="Зависимости не установлены", icon='ERROR')
            box.operator("bodytracker.install_deps")
            return

        if not model_available():
            box = layout.box()
            box.label(text="Модель распознавания позы не скачана", icon='ERROR')
            box.operator("bodytracker.download_model")
            return

        layout.prop(settings, "armature")

        layout.separator()
        layout.prop(settings, "input_mode", expand=True)

        col = layout.column()
        if settings.input_mode == 'WEBCAM':
            row = col.row(align=True)
            row.prop(settings, "camera_index")
            row.operator("bodytracker.scan_cameras", text="", icon='FILE_REFRESH')
            if sys.platform.startswith("win") and get_windows_camera_names() is None:
                col.operator("bodytracker.install_pygrabber", icon='INFO')
            col.prop(settings, "show_camera_preview")
            col.prop(settings, "record_keyframes")
            if settings.record_keyframes:
                col.prop(settings, "keyframe_step")
        else:
            col.prop(settings, "video_path")
            col.prop(settings, "video_start_frame")
            col.prop(settings, "video_frame_skip")
            col.prop(settings, "video_match_speed")

        col.prop(settings, "mirror")
        col.prop(settings, "smoothing")

        layout.separator()
        if settings.armature:
            box = layout.box()
            box.label(text="Соответствие костей (выберите кость рига для каждой части тела):")
            arm_data = settings.armature.data
            fields = [
                "head_bone", "spine_bone",
                "upper_arm_l_bone", "forearm_l_bone", "hand_l_bone",
                "upper_arm_r_bone", "forearm_r_bone", "hand_r_bone",
                "thigh_l_bone", "shin_l_bone", "foot_l_bone",
                "thigh_r_bone", "shin_r_bone", "foot_r_bone",
            ]
            for f in fields:
                box.prop_search(bones_cfg, f, arm_data, "bones")
        else:
            layout.label(text="Сначала выберите риг выше", icon='INFO')

        layout.separator()
        if settings.input_mode == 'WEBCAM':
            if settings.is_running:
                layout.operator("bodytracker.stop", icon='PAUSE')
            else:
                layout.operator("bodytracker.start", icon='PLAY')
        else:
            if settings.is_baking:
                layout.operator("bodytracker.stop_bake", icon='PAUSE')
                layout.label(text="Обработка... (ESC тоже отменяет)")
            else:
                layout.operator("bodytracker.bake_video", icon='RENDER_ANIMATION')


# ---------------------------------------------------------------------------
# Регистрация
# ---------------------------------------------------------------------------

classes = (
    BODYTRACKER_OT_install_deps,
    BODYTRACKER_OT_download_model,
    BODYTRACKER_OT_scan_cameras,
    BODYTRACKER_OT_install_pygrabber,
    BODYTRACKER_bone_settings,
    BODYTRACKER_settings,
    BODYTRACKER_OT_start,
    BODYTRACKER_OT_stop,
    BODYTRACKER_OT_bake_video,
    BODYTRACKER_OT_stop_bake,
    BODYTRACKER_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.body_tracker_settings = bpy.props.PointerProperty(type=BODYTRACKER_settings)
    bpy.types.Scene.body_tracker_bones = bpy.props.PointerProperty(type=BODYTRACKER_bone_settings)


def unregister():
    del bpy.types.Scene.body_tracker_bones
    del bpy.types.Scene.body_tracker_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
