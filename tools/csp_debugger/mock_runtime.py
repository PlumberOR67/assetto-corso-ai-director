"""Mock runtime for the CSP API.

Lets Lua code run without Assetto Corsa. Method names are camelCase on purpose:
Lua code calls them by their CSP names, for example ``ac.getSim()``.
"""

from collections.abc import Callable


class MockVec3:
    """Mock vec3 type."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def add(self, other: "MockVec3") -> "MockVec3":
        """Add two vectors.

        Args:
            other: The vector to add.

        Returns:
            A new vector, the sum of both.
        """
        return MockVec3(self.x + other.x, self.y + other.y, self.z + other.z)


class MockRay:
    """Mock ray object for raycasting."""

    def __init__(self, origin: MockVec3, direction: MockVec3, length: float) -> None:
        self.origin = origin
        self.direction = direction
        self.length = length

    def physics(self) -> tuple[MockVec3, MockVec3]:
        """Mock physics intersection.

        Returns:
            A hit position on the ground below the origin, and an up normal.
        """
        # Return a mock hit position and normal
        hit_pos = MockVec3(self.origin.x, 0.0, self.origin.z)
        hit_normal = MockVec3(0.0, 1.0, 0.0)
        return hit_pos, hit_normal

    def track(self) -> float:
        """Mock track intersection.

        Returns:
            A fixed distance to the track.
        """
        return 100.0  # Distance to the track

    def position(self) -> MockVec3:
        """Mock hit position.

        Returns:
            The point on the ground below the origin.
        """
        return MockVec3(self.origin.x, 0.0, self.origin.z)

    def normal(self) -> MockVec3:
        """Mock surface normal.

        Returns:
            A vector pointing straight up.
        """
        return MockVec3(0.0, 1.0, 0.0)


class MockCamera:
    """Mock camera object."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.transform = MockTransform()
        self.fov = 60.0

    def dispose(self) -> None:
        """Release the camera. The mock has nothing to release."""


class MockTransform:
    """Mock transform."""

    def __init__(self) -> None:
        self.position = MockVec3()


class MockTrackSpline:
    """Mock track spline."""

    def __init__(self) -> None:
        self.splinePointCount = 100

    def getSplinePoint(self, index: int) -> object:  # noqa: N802 - CSP API name
        """Mock spline point.

        Args:
            index: The spline point index.

        Returns:
            An object with a ``position`` 5 m further along x per index.
        """
        return type("Point", (), {"position": MockVec3(index * 5, 0, 0)})()


class MockSim:
    """Mock simulation object."""

    def __init__(self) -> None:
        self.isReplayActive = True
        self.timeSeconds = 10.5
        self.timeTotalSeconds = 120.0


class MockAC:
    """Mock ac namespace."""

    def __init__(self) -> None:
        self._logs: list[str] = []

    def getPatchVersion(self) -> int:  # noqa: N802 - CSP API name
        """Return a fixed CSP version.

        Returns:
            179, which stands for CSP 0.1.79.
        """
        return 179  # CSP version 0.1.79

    def getSim(self) -> MockSim:  # noqa: N802 - CSP API name
        """Return a mock simulation object.

        Returns:
            A new mock simulation object.
        """
        return MockSim()

    def getTrackSpline(self) -> MockTrackSpline:  # noqa: N802 - CSP API name
        """Return a mock track spline.

        Returns:
            A new mock track spline.
        """
        return MockTrackSpline()

    def getTrackName(self) -> str:  # noqa: N802 - CSP API name
        """Return a fixed track name.

        Returns:
            ``pk_akina``.
        """
        return "pk_akina"

    def getLayoutName(self) -> str:  # noqa: N802 - CSP API name
        """Return an empty layout name.

        Returns:
            An empty string, meaning the default layout.
        """
        return ""

    def getDocumentsPath(self) -> str:  # noqa: N802 - CSP API name
        """Return a fake Documents path.

        Returns:
            A fixed fake path. The mock never writes to it.
        """
        return "/tmp/mock_documents"  # noqa: S108 - a fake path; nothing is written there

    def grabCamera(self, reason: str) -> MockCamera:  # noqa: N802 - CSP API name
        """Return a mock camera.

        Args:
            reason: Why the camera is taken. Used as the camera name.

        Returns:
            A new mock camera.
        """
        return MockCamera(reason)

    def log(self, message: str) -> None:
        """Store a log message and print it.

        Args:
            message: The message to log.
        """
        self._logs.append(message)
        print(f"[AC LOG] {message}")


class MockRender:
    """Mock render namespace."""

    def createRay(  # noqa: N802 - CSP API name
        self, origin: MockVec3, direction: MockVec3, length: float
    ) -> MockRay:
        """Create a mock ray.

        Args:
            origin: Where the ray starts.
            direction: Which way the ray points.
            length: How far the ray reaches.

        Returns:
            A new mock ray.
        """
        return MockRay(origin, direction, length)


class MockUI:
    """Mock UI namespace."""

    def text(self, msg: str) -> None:
        """Print text instead of drawing it.

        Args:
            msg: The text.
        """
        print(f"[UI] {msg}")

    def textWrapped(self, msg: str) -> None:  # noqa: N802 - CSP API name
        """Print wrapped text instead of drawing it.

        Args:
            msg: The text.
        """
        print(f"[UI] {msg}")

    def button(self, _label: str) -> bool:
        """Report that the button was not pressed.

        Args:
            _label: The button label. The mock ignores it.

        Returns:
            Always False.
        """
        return False  # The button is not pressed

    def separator(self) -> None:
        """Draw a separator. The mock does nothing."""


class MockRuntime:
    """Complete mock runtime for CSP."""

    def __init__(self) -> None:
        self.ac = MockAC()
        self.render = MockRender()
        self.ui = MockUI()
        self.vec3 = MockVec3
        self.setTimeout: Callable[[Callable[[], None], float], None] = (
            lambda _fn, _delay: None  # The mock does not run it
        )

    def get_globals(self) -> dict[str, object]:
        """Get the globals for the Lua environment.

        Returns:
            The CSP namespaces and helpers, keyed by their Lua names.
        """
        return {
            "ac": self.ac,
            "render": self.render,
            "ui": self.ui,
            "vec3": self.vec3,
            "setTimeout": self.setTimeout,
        }
