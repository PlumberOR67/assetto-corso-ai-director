"""Shared GUI test setup.

The important thing here is the settings redirection. ``QSettings`` on Windows
writes to the registry by default, so a test that constructs a window would
mutate the developer's own preferences — window size, output folder, recent files
— and leave them changed after the run. Every test in this directory gets a store
backed by a temporary file instead.

The redirection is process-wide and has to happen before any ``QSettings`` is
constructed, which is why it is a session fixture with ``autouse``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="GUI extra not installed")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from assetto_ai_director.presentation.gui.settings import (
    APPLICATION_NAME,
    ORGANISATION_NAME,
    Settings,
)
from assetto_ai_director.presentation.gui.theme import apply_theme

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def _isolated_settings(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Keep the suite out of the developer's real settings.

    No teardown: the redirection is process-wide and the process ends with the
    run, so putting it back would only create a window in which a late-collected
    test could write to the real store.
    """
    directory = tmp_path_factory.mktemp("settings")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(directory))


@pytest.fixture(scope="module")
def app() -> QApplication:
    """One themed application for the whole module; Qt allows only one.

    The theme is applied whether or not the instance already existed. Reusing an
    untouched instance created by some earlier import leaves the application in
    Qt's default light style, and every theme assertion then tests nothing.
    """
    application = QApplication.instance() or QApplication([])
    apply_theme(application)  # type: ignore[arg-type]
    return application  # type: ignore[return-value]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A settings store private to one test.

    Args:
        tmp_path: Temporary directory.

    Returns:
        A store whose output root already points somewhere writable, so a test
        that triggers a write does not have to arrange one.
    """
    backing = QSettings(
        str(tmp_path / "settings.ini"),
        QSettings.Format.IniFormat,
    )
    store = Settings(backing)
    store.output_root = tmp_path / "out"
    return store


@pytest.fixture
def named_settings() -> Settings:
    """A store using the application's real organisation and application names.

    Only for tests that need to prove the names are what the installer and the
    packaged build will look for.

    Returns:
        The store.
    """
    return Settings(QSettings(ORGANISATION_NAME, APPLICATION_NAME))
