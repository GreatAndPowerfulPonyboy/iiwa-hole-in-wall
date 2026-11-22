"""Meshcat playback with recording.

HEADLESS=1 disables Meshcat (get_meshcat() returns None, play() does nothing).
Otherwise one Meshcat server starts on first use; the browser wait is skipped
unless HITW_WAIT_BROWSER=1.  Recordings can be saved as self-contained HTML
(Meshcat.StaticHtml).  Arm and wall are both driven by time.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
from pydrake.all import Meshcat, StartMeshcat

from .scene import Scene, set_state

HEADLESS = os.environ.get("HEADLESS", "0") not in ("", "0")
MESHCAT_BROWSER_WAIT = 15.0
_meshcat: Meshcat | None = None


def get_meshcat() -> Meshcat | None:
    global _meshcat
    if HEADLESS:
        return None
    if _meshcat is None:
        _meshcat = StartMeshcat()
        print(f"Meshcat: {_meshcat.web_url()}")
        if os.environ.get("HITW_WAIT_BROWSER", "0") == "1":
            deadline = time.time() + MESHCAT_BROWSER_WAIT
            while time.time() < deadline and _meshcat.GetNumActiveConnections() == 0:
                time.sleep(0.1)
    return _meshcat


def reset_meshcat() -> None:
    """Clear the scene tree and any recording before building a new scene."""
    if _meshcat is not None:
        _meshcat.Delete()
        _meshcat.DeleteRecording()


def play(
    scene: Scene,
    q_of_t: Callable[[float], np.ndarray],
    wall_y_of_t: Callable[[float], float],
    T: float,
    fps: float = 30.0,
    html_path: str | Path | None = None,
    realtime: bool = False,
) -> Path | None:
    """Animate arm and wall from t = 0 to T, publish the recording and
    optionally write it to HTML.  q_of_t returns the full 7-vector."""
    if scene.meshcat is None:
        return None
    meshcat = scene.meshcat
    meshcat.StartRecording(frames_per_second=fps, set_visualizations_while_recording=False)
    n = int(np.ceil(T * fps)) + 1
    t_wall = time.time()
    for k in range(n):
        t = min(k / fps, T)
        set_state(scene, q_of_t(t), wall_y_of_t(t))
        scene.context.SetTime(t)
        scene.diagram.ForcedPublish(scene.context)
        if realtime:
            time.sleep(max(0.0, t_wall + t - time.time()))
    meshcat.StopRecording()
    meshcat.PublishRecording()
    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(meshcat.StaticHtml())
        return html_path
    return None


def keep_open() -> None:
    if _meshcat is not None:
        input("Press Enter to quit.")
