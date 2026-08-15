"""
Mock runtime для CSP API.
Позволяет запускать Lua-код без Assetto Corsa.
"""

class MockVec3:
    """Mock vec3 type."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def add(self, other):
        return MockVec3(self.x + other.x, self.y + other.y, self.z + other.z)

class MockRay:
    """Mock ray object для raycasting."""
    def __init__(self, origin, direction, length):
        self.origin = origin
        self.direction = direction
        self.length = length

    def physics(self):
        """Mock physics intersection."""
        # Возвращаем mock hit position и normal
        hit_pos = MockVec3(self.origin.x, 0.0, self.origin.z)
        hit_normal = MockVec3(0.0, 1.0, 0.0)
        return hit_pos, hit_normal

    def track(self):
        """Mock track intersection."""
        return 100.0  # Расстояние до трассы

    def position(self):
        """Mock hit position."""
        return MockVec3(self.origin.x, 0.0, self.origin.z)

    def normal(self):
        """Mock surface normal."""
        return MockVec3(0.0, 1.0, 0.0)

class MockCamera:
    """Mock camera object."""
    def __init__(self, name):
        self.name = name
        self.transform = MockTransform()
        self.fov = 60.0

    def dispose(self):
        pass

class MockTransform:
    """Mock transform."""
    def __init__(self):
        self.position = MockVec3()

class MockTrackSpline:
    """Mock track spline."""
    def __init__(self):
        self.splinePointCount = 100

    def getSplinePoint(self, index):
        """Mock spline point."""
        return type('Point', (), {'position': MockVec3(index * 5, 0, 0)})()

class MockSim:
    """Mock simulation object."""
    def __init__(self):
        self.isReplayActive = True
        self.timeSeconds = 10.5
        self.timeTotalSeconds = 120.0

class MockAC:
    """Mock ac namespace."""
    def __init__(self):
        self._logs = []

    def getPatchVersion(self):
        return 179  # CSP version 0.1.79

    def getSim(self):
        return MockSim()

    def getTrackSpline(self):
        return MockTrackSpline()

    def getTrackName(self):
        return "pk_akina"

    def getLayoutName(self):
        return ""

    def getDocumentsPath(self):
        return "/tmp/mock_documents"

    def grabCamera(self, reason):
        return MockCamera(reason)

    def log(self, message):
        self._logs.append(message)
        print(f"[AC LOG] {message}")

class MockRender:
    """Mock render namespace."""
    def createRay(self, origin, direction, length):
        return MockRay(origin, direction, length)

class MockUI:
    """Mock UI namespace."""
    def text(self, msg):
        print(f"[UI] {msg}")

    def textWrapped(self, msg):
        print(f"[UI] {msg}")

    def button(self, label):
        return False  # Кнопка не нажата

    def separator(self):
        pass

class MockRuntime:
    """Полный mock runtime для CSP."""
    def __init__(self):
        self.ac = MockAC()
        self.render = MockRender()
        self.ui = MockUI()
        self.vec3 = MockVec3
        self.setTimeout = lambda fn, delay: None  # Не выполняем в mock

    def get_globals(self):
        """Получить globals для Lua environment."""
        return {
            'ac': self.ac,
            'render': self.render,
            'ui': self.ui,
            'vec3': self.vec3,
            'setTimeout': self.setTimeout,
        }
