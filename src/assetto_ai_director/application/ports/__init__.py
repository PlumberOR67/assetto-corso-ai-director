"""Ports: the contracts the application layer talks through.

Every module here defines Protocols and nothing else. No implementations, no
imports from :mod:`assetto_ai_director.infrastructure` — enforced by a dedicated
``import-linter`` contract.

Two of these ports are unusual and worth reading before using them:
:class:`~assetto_ai_director.application.ports.camera_driver.CameraDriver` and
:class:`~assetto_ai_director.application.ports.replay_transport.ReplayTransport`
document capabilities that, as far as could be established, **do not exist**.
They are written down anyway so the absence is explicit and searchable, rather
than being rediscovered by the next person who assumes the game must surely
expose a way to seek a replay.
"""

from __future__ import annotations
