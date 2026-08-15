# Реестр CSP API функций
# Формат: namespace.function -> {version, context, description, required}

CSP_API_REGISTRY = {
    # ac.* namespace
    "ac.getPatchVersion": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Получить версию CSP",
        "required": True,
        "returns": "int"
    },
    "ac.getSim": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Получить объект симуляции",
        "required": False,
        "returns": "SimObject"
    },
    "ac.getTrackSpline": {
        "version": "0.1.78+",
        "context": "replay,live",
        "description": "Получить сплайн трассы",
        "required": False,
        "returns": "TrackSpline"
    },
    "ac.getTrackName": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Получить имя трассы",
        "required": False,
        "returns": "string"
    },
    "ac.getLayoutName": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Получить имя лейаута",
        "required": False,
        "returns": "string"
    },
    "ac.getDocumentsPath": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Получить путь к Documents",
        "required": False,
        "returns": "string"
    },
    "ac.grabCamera": {
        "version": "0.1.78+",
        "context": "replay,live",
        "description": "Захватить контроль над камерой",
        "required": False,
        "returns": "CameraObject"
    },
    "ac.log": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Вывести в лог",
        "required": False,
        "returns": "void"
    },

    # render.* namespace
    "render.createRay": {
        "version": "0.2.0+",
        "context": "replay,live",
        "description": "Создать луч для raycasting",
        "required": False,
        "returns": "RayObject"
    },

    # Ray methods
    "ray:physics": {
        "version": "0.2.0+",
        "context": "replay,live",
        "description": "Проверить пересечение с физикой",
        "required": False,
        "returns": "hitPos, hitNormal"
    },
    "ray:track": {
        "version": "0.2.0+",
        "context": "replay,live",
        "description": "Проверить пересечение с трассой",
        "required": False,
        "returns": "distance"
    },
    "ray:position": {
        "version": "0.2.0+",
        "context": "replay,live",
        "description": "Получить позицию попадания",
        "required": False,
        "returns": "vec3"
    },
    "ray:normal": {
        "version": "0.2.0+",
        "context": "replay,live",
        "description": "Получить нормаль поверхности",
        "required": False,
        "returns": "vec3"
    },

    # ui.* namespace
    "ui.text": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Вывести текст в UI",
        "required": False,
        "returns": "void"
    },
    "ui.textWrapped": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Вывести текст с переносом",
        "required": False,
        "returns": "void"
    },
    "ui.button": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Создать кнопку",
        "required": False,
        "returns": "bool"
    },
    "ui.separator": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Разделитель в UI",
        "required": False,
        "returns": "void"
    },

    # sim.* namespace (через ac.getSim())
    "sim.isReplayActive": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Проверить, активен ли реплей",
        "required": False,
        "returns": "bool"
    },
    "sim.timeSeconds": {
        "version": "0.1.0+",
        "context": "replay,live",
        "description": "Текущее время сессии",
        "required": False,
        "returns": "float"
    },

    # vec3 type
    "vec3": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Векторный тип",
        "required": False,
        "returns": "vec3"
    },

    # setTimeout
    "setTimeout": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Вызвать функцию через задержку",
        "required": False,
        "returns": "void"
    }
}

# Функции, которые требуют особого контекста
CONTEXT_RESTRICTED = {
    "ac.getTrackSpline": ["replay", "live"],
    "ac.grabCamera": ["replay", "live"],
    "render.createRay": ["replay", "live"],
    "ray:physics": ["replay", "live"],
    "ray:track": ["replay", "live"],
}

# Функции, которые могут отсутствовать в старых версиях
VERSION_RESTRICTED = {
    "0.2.0+": [
        "render.createRay",
        "ray:physics",
        "ray:track",
        "ray:position",
        "ray:normal",
    ],
    "0.1.78+": [
        "ac.getTrackSpline",
        "ac.grabCamera",
    ]
}
