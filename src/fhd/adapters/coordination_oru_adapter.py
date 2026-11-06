"""Bridge from ORU's trajectory-envelope coordinator to No02's frozen executor.

The ORU Java process retains its original critical-section / precedence /
dynamic-priority machinery. Python supplies actual robot state every 0.4 s and
receives the critical point (path index) chosen by ORU; that point is converted
to a legal No02 stop target and checked by the frozen executor.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from fhd.execution.coordinator import Action, Coordinator, Observation

DEFAULT_JAVA = r"D:\No02_external_repos\jdk11\jdk-11.0.32.1+1\bin\java.exe"
DEFAULT_CP = (
    r"D:\No02_external_repos\coordination_oru\build\classes\java\main;"
    r"D:\No02_external_repos\coordination_oru\build\install\coordination_oru\lib\*"
)


class OruJavaBridge:
    def __init__(self, java_exe: str = DEFAULT_JAVA, classpath: str = DEFAULT_CP):
        cmd = [java_exe, "-Djava.awt.headless=true", "-Dfile.encoding=UTF-8",
               "-cp", classpath,
               "se.oru.coordination.coordination_oru.bridge.No02OruBridge"]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        self.last_error = None

    def _send(self, obj) -> dict:
        if self.proc.poll() is not None:
            raise RuntimeError(f"ORU bridge exited: {self.proc.returncode}")
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("ORU bridge closed stdout")
        out = json.loads(line)
        if not out.get("ok", False):
            self.last_error = out.get("error")
            raise RuntimeError(f"ORU bridge error: {self.last_error}")
        return out

    def init(self, paths: np.ndarray):
        payload = {str(i + 1): np.asarray(paths[i], dtype=float).tolist()
                   for i in range(len(paths))}
        return self._send({"cmd": "init", "paths": payload})

    def step(self, t_s: float, states: list[dict]) -> dict:
        robots = {str(i + 1): s for i, s in enumerate(states)}
        return self._send({"cmd": "step", "t_ms": int(round(t_s * 1000.0)),
                           "robots": robots}).get("cp", {})

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write('{"cmd":"close"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=0.5)
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class OruCoordinator(Coordinator):
    name = "coordination_oru"

    def __init__(self, paths, wait_slots, d_safe, dt_slot, horizon_s, pause_p,
                 config=None):
        super().__init__(paths, wait_slots, d_safe, dt_slot, horizon_s, pause_p,
                         config)
        c = self.config
        self.bridge = OruJavaBridge(c.get("java_exe", DEFAULT_JAVA),
                                    c.get("classpath", DEFAULT_CP))
        self.bridge.init(self.paths)
        self.hold_s = float(c.get("oru_hold_s", 0.8))
        self._last_cp = {}
        self._last_t = {}
        self.errors = 0
        self.last_error = None

    def _path_index(self, s: float, i: int) -> int:
        return int(np.clip(np.searchsorted(self.s_wp[i], s + 1e-6, side="right") - 1,
                           0, self.paths.shape[1] - 1))

    def decide(self, obs: Observation) -> list[Action]:
        self.observe(obs)
        states = []
        for i, rb in enumerate(obs.robots):
            idx = self._path_index(rb.s, i)
            theta = (float(np.arctan2(rb.vy, rb.vx))
                     if abs(rb.vx) + abs(rb.vy) > 1e-6 else 0.0)
            states.append({"x": rb.x, "y": rb.y, "idx": idx,
                           "vx": rb.vx, "vy": rb.vy, "theta": theta})
        try:
            cps = self.bridge.step(obs.t, states)
        except Exception as exc:
            self.errors += 1
            self.last_error = str(exc)
            self.last_diagnostics = {"error": self.last_error}
            return []
        actions = []
        for i, rb in enumerate(obs.robots):
            cp = int(cps.get(str(i + 1), -1))
            if cp < 0 or rb.mode != "NORMAL":
                continue
            idx = self._path_index(rb.s, i)
            if cp <= idx:
                # ORU says do not pass the current critical point; issue a
                # legal immediate stop just ahead rather than an instant freeze.
                s_target = min(rb.s + max(0.05, rb.speed * self.dt_slot + 0.05),
                               rb.length)
            else:
                limit = float(self.s_wp[i, min(cp, self.paths.shape[1] - 1)]) - 0.10
                brake = rb.speed * rb.speed / 4.0 + rb.speed * self.dt_slot + 0.05
                cands = [float(self.s_wp[i, k]) for k in range(cp)
                         if float(self.s_wp[i, k]) - rb.s >= brake
                         and float(self.s_wp[i, k]) < limit]
                if cands:
                    s_target = max(cands)  # latest point from which CP is safely stoppable
                else:
                    s_target = max(rb.s + 0.05, limit - 0.05)
            if s_target >= rb.length - 1e-6:
                s_target = rb.length - 1e-4
            if obs.t - self._last_t.get(i, -1e9) < 0.5:
                continue
            self._last_cp[i] = cp
            self._last_t[i] = obs.t
            actions.append(Action(
                robot=i, kind="stop", s_target=s_target,
                duration_s=self.hold_s,
                candidate_wp=min(cp, self.paths.shape[1] - 1),
                reason=f"ORU_cp_{cp}_idx_{idx}"))
        self.last_diagnostics = {"cps": cps, "actions": len(actions),
                                 "errors": self.errors}
        self.last_decisions = actions
        return actions

    def close(self):
        self.bridge.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
