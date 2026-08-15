"""Frame-tagged vectors, poses and the camera basis.

Camera orientation is stored as **position + look direction + up**, not as Euler
angles. Two reasons, both practical:

* Euler angles gimbal-lock, and a camera tracking a car through a hairpin passes
  straight through the degenerate orientation.
* Custom Shaders Patch's ``ac.GrabbedCamera`` exposes a ``mat4x4 transform`` and
  the one verified real-world usage computes look as ``target - position``, i.e.
  a **direction**, not a point. Storing what the consumer actually wants avoids a
  lossy conversion at the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

from assetto_ai_director.domain.frames import ReferenceFrame
from assetto_ai_director.domain.units import Metres, Radians

__all__ = ["CameraBasis", "Vec3", "wrap_to_pi_array_free"]

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Vec3:
    """A 3-component vector tagged with the reference frame it lives in.

    Assetto Corsa uses a y-up convention: ``x`` and ``z`` span the ground plane
    and ``y`` is elevation. This class follows it, so ``y`` is height everywhere
    in the codebase and the ground-plane heading is ``atan2(x, z)``.

    Attributes:
        x: First component. Lateral in car-local frame.
        y: Second component. Always elevation.
        z: Third component. Longitudinal in car-local frame.
        frame: The frame these components are expressed in.
    """

    x: float
    y: float
    z: float
    frame: ReferenceFrame = ReferenceFrame.WORLD

    def __add__(self, other: Vec3) -> Vec3:
        """Add two vectors of the same frame.

        Args:
            other: The vector to add.

        Returns:
            The component-wise sum, in the shared frame.

        Raises:
            ValueError: If the two vectors are in different frames.
        """
        self._require_same_frame(other, "add")
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z, self.frame)

    def __sub__(self, other: Vec3) -> Vec3:
        """Subtract a vector of the same frame.

        Args:
            other: The vector to subtract.

        Returns:
            The component-wise difference, in the shared frame.

        Raises:
            ValueError: If the two vectors are in different frames.
        """
        self._require_same_frame(other, "subtract")
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z, self.frame)

    def __mul__(self, scalar: float) -> Vec3:
        """Scale the vector.

        Args:
            scalar: The factor to multiply each component by.

        Returns:
            The scaled vector, in the same frame.
        """
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar, self.frame)

    __rmul__ = __mul__

    def dot(self, other: Vec3) -> float:
        """Compute the dot product with another vector of the same frame.

        Args:
            other: The other vector.

        Returns:
            The scalar dot product.

        Raises:
            ValueError: If the two vectors are in different frames.
        """
        self._require_same_frame(other, "dot")
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        """Compute the cross product with another vector of the same frame.

        Args:
            other: The other vector.

        Returns:
            The vector cross product, in the shared frame.

        Raises:
            ValueError: If the two vectors are in different frames.
        """
        self._require_same_frame(other, "cross")
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
            self.frame,
        )

    @property
    def length(self) -> float:
        """The Euclidean magnitude of the vector.

        Returns:
            ``sqrt(x^2 + y^2 + z^2)``.
        """
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def length_ground(self) -> float:
        """The magnitude projected onto the ground plane, ignoring elevation.

        Speed on a downhill course is dominated by the ground-plane component;
        including the vertical term inflates it on steep sections.

        Returns:
            ``sqrt(x^2 + z^2)``.
        """
        return math.hypot(self.x, self.z)

    def normalised(self) -> Vec3:
        """Return a unit-length copy.

        Returns:
            The vector scaled to unit length, or a zero vector if the input
            magnitude is below the numerical epsilon.
        """
        magnitude = self.length
        if magnitude < _EPSILON:
            return Vec3(0.0, 0.0, 0.0, self.frame)
        inverse = 1.0 / magnitude
        return Vec3(self.x * inverse, self.y * inverse, self.z * inverse, self.frame)

    @property
    def heading(self) -> Radians:
        """The ground-plane bearing of this vector.

        Uses ``atan2(x, z)`` — measured from the ``+z`` axis toward ``+x`` — to
        match Assetto Corsa's y-up, z-forward convention. Every angular
        comparison in the codebase uses this same convention, so path tangent and
        car heading are directly subtractable.

        Returns:
            The bearing in radians, in ``(-pi, pi]``.
        """
        return Radians(math.atan2(self.x, self.z))

    def distance_to(self, other: Vec3) -> Metres:
        """Compute the distance to another point in the same frame.

        Args:
            other: The other point.

        Returns:
            The Euclidean distance in metres.

        Raises:
            ValueError: If the two vectors are in different frames.
        """
        return Metres((self - other).length)

    def with_frame(self, frame: ReferenceFrame) -> Self:
        """Re-tag the vector as belonging to a different frame.

        This performs **no rotation**. It exists for the single legitimate case
        where a frame was previously :attr:`ReferenceFrame.UNKNOWN` and a
        calibration step has since established what it actually was.

        Args:
            frame: The frame to tag the components with.

        Returns:
            A copy carrying the new frame tag.
        """
        return type(self)(self.x, self.y, self.z, frame)

    def _require_same_frame(self, other: Vec3, operation: str) -> None:
        """Raise if two vectors are not in the same reference frame.

        Args:
            other: The other operand.
            operation: Name of the operation, for the error message.

        Raises:
            ValueError: If the frames differ.
        """
        if self.frame is not other.frame:
            message = (
                f"Cannot {operation} vectors in different frames: "
                f"{self.frame} and {other.frame}. Convert explicitly first."
            )
            raise ValueError(message)


ZERO_WORLD = Vec3(0.0, 0.0, 0.0, ReferenceFrame.WORLD)
UP_WORLD = Vec3(0.0, 1.0, 0.0, ReferenceFrame.WORLD)


@dataclass(frozen=True, slots=True)
class CameraBasis:
    """A camera's placement: where it is, where it points, and which way is up.

    Deliberately stores ``look`` as a **direction** rather than a target point,
    because that is what the one verified consumer — CSP's ``GrabbedCamera`` —
    expects. Converting a target to a direction is lossless; the reverse is not,
    since the distance is unrecoverable.

    Attributes:
        position: Camera location in world space.
        look: Unit direction the camera faces. Not a point.
        up: Unit up vector. Orthogonalised against ``look`` by
            :meth:`orthonormalised`.
    """

    position: Vec3
    look: Vec3
    up: Vec3 = UP_WORLD

    @classmethod
    def looking_at(cls, position: Vec3, target: Vec3, up: Vec3 = UP_WORLD) -> CameraBasis:
        """Build a basis for a camera at ``position`` aimed at ``target``.

        Args:
            position: Where the camera sits, in world space.
            target: The point to aim at, in world space.
            up: Desired up direction; orthogonalised against the look direction.

        Returns:
            An orthonormal camera basis.
        """
        return cls(position, (target - position).normalised(), up).orthonormalised()

    def orthonormalised(self) -> CameraBasis:
        """Return a copy with an orthonormal basis.

        Mirrors CSP's ``GrabbedCamera:normalize()``. Repeatedly nudging ``up``
        while tracking a car — a banked corner, a crested hill — lets the basis
        drift out of orthogonality, and a skewed basis shears the rendered image.

        Falls back to the world up axis when ``look`` is vertical and the chosen
        ``up`` is degenerate, which happens on a directly overhead helicopter shot.

        Returns:
            A copy whose ``look`` and ``up`` are unit length and perpendicular.
        """
        forward = self.look.normalised()
        right = self.up.cross(forward)
        if right.length < _EPSILON:
            # Looking straight up or down: pick any axis perpendicular to forward.
            fallback = Vec3(0.0, 0.0, 1.0, forward.frame)
            right = fallback.cross(forward)
            if right.length < _EPSILON:
                right = Vec3(1.0, 0.0, 0.0, forward.frame).cross(forward)
        right = right.normalised()
        true_up = forward.cross(right).normalised()
        return CameraBasis(self.position, forward, true_up)

    def distance_to(self, point: Vec3) -> Metres:
        """Compute the distance from the camera to a world point.

        Args:
            point: The point of interest, in world space.

        Returns:
            The Euclidean distance in metres. Drives focus distance and the
            field-of-view zoom tracking that trackside cameras use.
        """
        return self.position.distance_to(point)


def wrap_to_pi_array_free(angle: float) -> float:
    """Wrap a bare ``float`` angle into ``(-pi, pi]``.

    A convenience twin of :func:`assetto_ai_director.domain.units.wrap_pi` for
    call sites working with untyped floats, so neither has to import the other's
    unit type.

    Args:
        angle: An angle in radians, of any magnitude.

    Returns:
        The equivalent angle in ``(-pi, pi]``.
    """
    wrapped = math.remainder(angle, math.tau)
    return math.pi if wrapped == -math.pi else wrapped
