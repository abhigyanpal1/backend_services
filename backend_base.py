# backend_base.py
import time
import threading
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def create_app(service_name: str = "SRE Backend") -> FastAPI:
    app = FastAPI(title=f"{service_name}", version="0.1.0")

    # -----------------------------
    # Global flags / demo state (per-process/module)
    # -----------------------------
    SERVICE_DOWN = False
    SERVICE_DOWN_REASON = "Maintenance mode"

    SIM_DISK_FILES: List[str] = []  # tracks files created by /simulate/disk_fill

    # Agent state + incident store (simple in-memory)
    AGENT_RUNNING = False
    INCIDENTS: List[Dict[str, Any]] = []  # newest appended; UI can reverse

    # -----------------------------
    # Helpers
    # -----------------------------
    def _now_iso():
        return datetime.utcnow().isoformat() + "Z"

    def record_incident(
        incident_type: str,
        severity: str,
        details: str,
        evidence: Optional[Dict[str, Any]] = None,
    ):
        INCIDENTS.append({
            "time": _now_iso(),
            "service": service_name,
            "type": incident_type,
            "severity": severity,
            "details": details,
            "evidence": evidence or {},
        })

    def burn_cpu_worker(seconds: int):
        """Top-level function so multiprocessing can pickle it on Windows."""
        end = time.time() + seconds
        x = 0
        while time.time() < end:
            # Heavier math to spike CPU more
            x = (x * 3 + 7) % 1000003
            x = (x * 13 + 17) % 10000019
            x = (x ^ 0xABCDEF) % 10000079

    # -----------------------------
    # UI expected payload schema (minimal)
    # -----------------------------
    class StartAgentPayload(BaseModel):
        env: Optional[str] = "Linux"

    # -----------------------------
    # Endpoints
    # -----------------------------
    @app.get("/health")
    def health():
        """
        Health checkpoint for probes/dashboards.
        Reflects simulated service down/up state.
        """
        if SERVICE_DOWN:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "DOWN",
                    "reason": SERVICE_DOWN_REASON,
                    "service": service_name,
                    "timestamp": _now_iso(),
                },
            )
        return {
            "status": "OK",
            "service": service_name,
            "timestamp": _now_iso(),
        }

    @app.get("/metrics/cpu")
    def cpu_metrics():
        """
        Returns CPU utilization from the server running this backend.
        """
        cpu = psutil.cpu_percent(interval=0.3)
        cores = psutil.cpu_count(logical=True)
        load = psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
        return {
            "service": service_name,
            "cpu_percent": cpu,
            "cores": cores,
            "load_avg": load,
            "timestamp": _now_iso(),
        }

    @app.get("/metrics/disk")
    def disk_metrics(path: str = "/"):
        """
        Returns disk usage for the given path (default '/').
        """
        du = psutil.disk_usage(path)
        return {
            "service": service_name,
            "path": path,
            "total_bytes": du.total,
            "used_bytes": du.used,
            "free_bytes": du.free,
            "used_percent": du.percent,
            "timestamp": _now_iso(),
        }

    # -----------------------------
    # Agent Controls (minimal demo state)
    # -----------------------------
    @app.post("/agent/start")
    def agent_start(payload: StartAgentPayload):
        nonlocal AGENT_RUNNING
        AGENT_RUNNING = True
        record_incident(
            incident_type="Agent Started",
            severity="INFO",
            details=f"Agent started via UI. env={payload.env}",
            evidence={"env": payload.env},
        )
        return {"ok": True, "message": "Agent started", "env": payload.env, "service": service_name}

    @app.post("/agent/stop")
    def agent_stop():
        nonlocal AGENT_RUNNING
        AGENT_RUNNING = False
        record_incident(
            incident_type="Agent Stopped",
            severity="INFO",
            details="Agent stopped via UI.",
        )
        return {"ok": True, "message": "Agent stopped", "service": service_name}

    @app.post("/agent/simulate")
    def agent_simulate():
        """
        Minimal demo-safe simulate:
        - Always records an incident so UI/email never show 'No Incident'
        - Optionally triggers a light CPU spike by default
        """
        if not AGENT_RUNNING:
            record_incident(
                incident_type="Simulation Requested (Agent Not Running)",
                severity="MEDIUM",
                details="Simulate clicked while agent not running. Recording incident.",
            )
            return {"ok": True, "message": "Simulation recorded (agent not running).", "service": service_name}

        # Trigger a short CPU spike in a background thread
        seconds = 10
        workers = 2

        def start_processes(s: int, n: int):
            n = max(1, int(n))
            procs = []
            for _ in range(n):
                p = mp.Process(target=burn_cpu_worker, args=(s,), daemon=True)
                p.start()
                procs.append(p)
            for p in procs:
                p.join()

        threading.Thread(target=start_processes, args=(seconds, workers), daemon=True).start()

        record_incident(
            incident_type="CPU Spike",
            severity="HIGH",
            details=f"Simulated CPU spike for ~{seconds}s with {workers} workers.",
            evidence={"seconds": seconds, "workers": workers},
        )
        return {"ok": True, "message": "Incident simulated and recorded.", "service": service_name}

    @app.get("/incidents")
    def fetch_incidents():
        return INCIDENTS[-200:]

    # -----------------------------
    # Simulation Endpoints
    # -----------------------------
    @app.post("/simulate/service_down")
    def simulate_service_down(reason: str = "Simulated outage"):
        nonlocal SERVICE_DOWN, SERVICE_DOWN_REASON
        SERVICE_DOWN = True
        SERVICE_DOWN_REASON = reason

        record_incident(
            incident_type="Backend URL Unhealthy",
            severity="HIGH",
            details=f"Service forced DOWN. reason={reason}",
            evidence={"reason": reason},
        )

        return {"ok": True, "message": "Service now returns 503 on /health", "reason": reason, "service": service_name}

    @app.post("/simulate/service_up")
    def simulate_service_up():
        nonlocal SERVICE_DOWN
        SERVICE_DOWN = False

        record_incident(
            incident_type="Service Restored",
            severity="INFO",
            details="Service restored (200 on /health).",
        )

        return {"ok": True, "message": "Service restored (200 on /health)", "service": service_name}

    @app.post("/simulate/cpu")
    def simulate_cpu(seconds: int = 15, workers: int = 4):
        """
        Strong CPU spike:
        - Uses multiprocessing with a TOP-LEVEL worker function (picklable).
        - 'workers' controls how many CPU-burning processes to run.
        """
        def start_processes(s: int, n: int):
            n = max(1, int(n))
            procs = []
            for _ in range(n):
                p = mp.Process(target=burn_cpu_worker, args=(s,), daemon=True)
                p.start()
                procs.append(p)
            for p in procs:
                p.join()

        threading.Thread(target=start_processes, args=(seconds, workers), daemon=True).start()

        record_incident(
            incident_type="CPU Spike",
            severity="HIGH",
            details=f"Simulated CPU burn for ~{seconds}s with {workers} workers via /simulate/cpu",
            evidence={"seconds": seconds, "workers": workers},
        )

        return {"ok": True, "message": f"CPU burn started for ~{seconds}s with {workers} workers", "service": service_name}

    @app.post("/simulate/disk_fill")
    def simulate_disk_fill(mb: int = 50):
        """
        SAFE demo: creates a temp file in /tmp of size `mb` MB.
        Simulates disk usage changes without touching system files.
        """
        nonlocal SIM_DISK_FILES
        mb = max(1, int(mb))
        tmp_dir = Path("/tmp/sre_demo")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        file_path = tmp_dir / f"disk_demo_{int(time.time())}.bin"
        chunk = b"\0" * (1024 * 1024)  # 1MB

        with open(file_path, "wb") as f:
            for _ in range(mb):
                f.write(chunk)

        SIM_DISK_FILES.append(str(file_path))

        record_incident(
            incident_type="Disk Usage High",
            severity="MEDIUM",
            details=f"Simulated disk fill: created {mb}MB at {file_path}",
            evidence={"file": str(file_path), "size_mb": mb},
        )

        return {"ok": True, "created": str(file_path), "size_mb": mb, "tracked_files": len(SIM_DISK_FILES), "service": service_name}

    @app.post("/simulate/disk_cleanup")
    def simulate_disk_cleanup():
        """
        Removes demo files created via /simulate/disk_fill
        """
        nonlocal SIM_DISK_FILES
        removed = []
        for fp in list(SIM_DISK_FILES):
            try:
                Path(fp).unlink(missing_ok=True)
                removed.append(fp)
            except Exception:
                pass
            finally:
                if fp in SIM_DISK_FILES:
                    SIM_DISK_FILES.remove(fp)

        tmp_dir = Path("/tmp/sre_demo")
        try:
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
        except Exception:
            pass

        record_incident(
            incident_type="Disk Cleanup",
            severity="INFO",
            details=f"Removed {len(removed)} demo disk files.",
            evidence={"removed": removed},
        )

        return {"ok": True, "removed": removed, "remaining_tracked_files": len(SIM_DISK_FILES), "service": service_name}

    return app