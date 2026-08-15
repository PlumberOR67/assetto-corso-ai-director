"""The camera driver port — which currently has no implementation, on purpose.

**Read this before assuming the game can be driven from Python.** What follows is
what was actually established, so the next person does not repeat the search.

**Kunos's own Python app API** offers camera *mode* switching and car focus —
``ac.setCameraMode``, ``ac.focusCar``, ``ac.setCameraCar``, ``ac.getCameraCarCount``
— plus *relative* free-camera nudges such as ``ac.freeCameraMoveForward`` and
``ac.freeCameraRotatePitch``. There is **no absolute-placement call**. You can ask
for "chase camera on car 3"; you cannot ask for "camera at this world position
looking this direction".

**Custom Shaders Patch** does expose absolute placement, and it is the only thing
that does. ``ac.grabCamera(reason)`` returns an ``ac.GrabbedCamera`` carrying a
writable ``mat4x4 transform`` plus ``fov`` (vertical degrees), ``dofDistance``,
``dofFactor``, ``exposure``, ``ownShare`` and ``cameraRestoreThreshold``, with
``active()``, ``dispose()``, ``normalize()`` and ``alignCar()``. The grab is
**exclusive**: it returns ``nil`` plus the name of the script already holding the
camera, so a competing tool must be reported to the user by name rather than
failing silently.

**The open question that blocks everything.** Nothing found states whether
``grabCamera`` functions during *replay playback*, as opposed to live driving.
The one verified real-world usage runs in a live game mode. Circumstantially, the
existing automatic replay-directing tool uses a different mechanism entirely,
which is weak evidence *against*. Until it is settled, this port stays a stub and
the project's camera output is a ``cameras.ini`` track camera set, which
demonstrably works in unmodified Assetto Corsa today.

**The experiment that would settle it now exists**, at
``spikes/csp_replay_camera/``. It is a Custom Shaders Patch Lua app that grabs
the camera during a replay, moves it a metre for two seconds, gives it back, and
prints four lines. It needs somebody with the game to run it once. What it
answers decides whether this port gets an implementation, a narrower one that
works only in a live session, or deletion — and which of those it is has been a
guess for the whole life of this project.

Writing an implementation now would mean shipping code that cannot be tested and
may not work — precisely the kind of confident guess this project refuses to make.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from assetto_ai_director.domain.camera import CameraKeyframe

__all__ = ["CameraDriver"]


@runtime_checkable
class CameraDriver(Protocol):
    """Applies camera state to a running simulator.

    No implementation ships with this project. See the module docstring for what
    was searched and what was found.
    """

    @property
    def name(self) -> str:
        """A short identifier for this driver.

        Returns:
            The driver name.
        """
        ...

    def acquire(self, reason: str) -> None:
        """Take exclusive control of the camera.

        Args:
            reason: Human-readable justification, shown to the user by the
                simulator and by any competing tool that is refused the grab.

        Raises:
            RuntimeError: If another script already holds the camera. The message
                should name the holder — a silent failure here looks exactly like
                a broken timeline.
        """
        ...

    def apply(self, keyframe: CameraKeyframe) -> None:
        """Push one frame of camera state to the simulator.

        Args:
            keyframe: The state to apply.
        """
        ...

    def release(self) -> None:
        """Give the camera back to the simulator."""
        ...
