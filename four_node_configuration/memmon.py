# utilities/memmon.py
from __future__ import annotations
import os, json, time, threading
from pathlib import Path
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None


class PeakRSSMonitor:
    def __init__(self, out_json: Path, sample_s: float = 0.2, meta: dict | None = None):
        self.out_json = Path(out_json)
        self.sample_s = float(sample_s)
        self.meta = meta or {}

        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

        self._peak_mb: float | None = None
        self._min_mb: float | None = None
        self._sum_mb: float = 0.0
        self._n: int = 0
        self._t0: float | None = None

    def start(self) -> None:
        self.out_json.parent.mkdir(parents=True, exist_ok=True)

        # reset counters (important if object is reused)
        self._stop_evt.clear()
        self._thread = None
        self._peak_mb = None
        self._min_mb = None
        self._sum_mb = 0.0
        self._n = 0
        self._t0 = time.time()

        if psutil is None:
            return

        proc = psutil.Process(os.getpid())

        def loop():
            while not self._stop_evt.is_set():
                try:
                    rss_mb = proc.memory_info().rss / (1024 * 1024)

                    # peak
                    if self._peak_mb is None or rss_mb > self._peak_mb:
                        self._peak_mb = rss_mb

                    # min
                    if self._min_mb is None or rss_mb < self._min_mb:
                        self._min_mb = rss_mb

                    # avg accumulator
                    self._sum_mb += rss_mb
                    self._n += 1
                except Exception:
                    pass

                time.sleep(self.sample_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_and_write(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

        duration_s = None
        if self._t0 is not None:
            duration_s = time.time() - self._t0

        avg_mb = None
        if self._n > 0:
            avg_mb = self._sum_mb / self._n

        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "sample_s": self.sample_s,
            "duration_s": (round(duration_s, 2) if isinstance(duration_s, (int, float)) else None),
            "n_samples": self._n,
            "min_rss_mb": (round(self._min_mb, 2) if isinstance(self._min_mb, (int, float)) else None),
            "avg_rss_mb": (round(avg_mb, 2) if isinstance(avg_mb, (int, float)) else None),
            "peak_rss_mb": (round(self._peak_mb, 2) if isinstance(self._peak_mb, (int, float)) else None),
            "psutil_available": (psutil is not None),
            **self.meta,
        }

        self.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class TotalRSSMonitor:
    """
    Monitora RSS totale: parent + children (ricorsivo).
    Da usare SOLO nel processo principale.
    Salva: peak totale e anche media (avg) totale.
    """
    def __init__(self, out_json: Path, sample_s: float = 0.5, meta: dict | None = None, recursive: bool = True):
        self.out_json = Path(out_json)
        self.sample_s = float(sample_s)
        self.meta = meta or {}
        self.recursive = bool(recursive)

        self._stop_evt = threading.Event()
        self._thread = None

        self._peak_mb = None
        self._sum_mb = 0.0
        self._n = 0

    def start(self):
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        if psutil is None:
            self._peak_mb = None
            return

        parent = psutil.Process(os.getpid())
        self._peak_mb = 0.0
        self._sum_mb = 0.0
        self._n = 0

        def loop():
            while not self._stop_evt.is_set():
                total_mb = 0.0
                try:
                    procs = [parent] + parent.children(recursive=self.recursive)
                    for p in procs:
                        try:
                            total_mb += p.memory_info().rss / (1024 * 1024)
                        except Exception:
                            pass
                except Exception:
                    pass

                # stats
                self._sum_mb += total_mb
                self._n += 1
                if total_mb > (self._peak_mb or 0.0):
                    self._peak_mb = total_mb

                time.sleep(self.sample_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_and_write(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

        avg_mb = (self._sum_mb / self._n) if self._n > 0 else None

        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "pid_parent": os.getpid(),
            "sample_s": self.sample_s,
            "peak_total_rss_mb": (round(self._peak_mb, 2) if isinstance(self._peak_mb, (int, float)) else None),
            "avg_total_rss_mb": (round(avg_mb, 2) if isinstance(avg_mb, (int, float)) else None),
            "samples": self._n,
            "psutil_available": (psutil is not None),
            "recursive_children": self.recursive,
            **self.meta,
        }
        self.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ProcMonitor:
    """
    Monitora RSS + CPU del processo corrente.
    Salva:
      - peak_rss_mb
      - avg_rss_mb
      - cpu_avg_percent
      - cpu_peak_percent
      - cpu_time_s

    Extra (consigliati):
      - n_cpus
      - cpu_avg_percent_norm / cpu_peak_percent_norm  (CPU% normalizzata su 1 CPU=100/n_cpus)
      - num_threads_peak
      - (opzionale) metriche include_children
    """

    def __init__(self, out_json: Path, sample_s: float = 0.2, meta: dict | None = None, include_children: bool = False):
        self.out_json = Path(out_json)
        self.sample_s = float(sample_s)
        self.meta = meta or {}
        self.include_children = bool(include_children)

        self._stop_evt = threading.Event()
        self._thread = None

        self._rss_peak = 0.0
        self._rss_sum = 0.0
        self._rss_n = 0

        self._cpu_peak = 0.0
        self._cpu_sum = 0.0
        self._cpu_n = 0

        self._threads_peak = 0

        # opzionale: processo+figli
        self._rssc_peak = 0.0
        self._rssc_sum = 0.0
        self._rssc_n = 0

        self._cpuc_peak = 0.0
        self._cpuc_sum = 0.0
        self._cpuc_n = 0

        self._n_cpus = None

    def start(self):
        self.out_json.parent.mkdir(parents=True, exist_ok=True)

        # RESET
        self._stop_evt.clear()
        self._thread = None

        self._rss_peak = self._rss_sum = 0.0
        self._rss_n = 0

        self._cpu_peak = self._cpu_sum = 0.0
        self._cpu_n = 0

        self._threads_peak = 0

        self._rssc_peak = self._rssc_sum = 0.0
        self._rssc_n = 0

        self._cpuc_peak = self._cpuc_sum = 0.0
        self._cpuc_n = 0

        if psutil is None:
            return

        proc = psutil.Process(os.getpid())
        self._n_cpus = psutil.cpu_count(logical=True) or 1

        # warmup CPU counters (NON blocca)
        try:
            proc.cpu_percent(interval=None)
            if self.include_children:
                for c in proc.children(recursive=True):
                    try:
                        c.cpu_percent(interval=None)
                    except Exception:
                        pass
        except Exception:
            pass

        def safe_children():
            try:
                return proc.children(recursive=True)
            except Exception:
                return []

        def loop():
            while not self._stop_evt.is_set():
                try:
                    # RSS del processo
                    rss_mb = proc.memory_info().rss / (1024 * 1024)
                    self._rss_sum += rss_mb
                    self._rss_n += 1
                    if rss_mb > self._rss_peak:
                        self._rss_peak = rss_mb

                    # Thread count
                    try:
                        nt = proc.num_threads()
                        if nt > self._threads_peak:
                            self._threads_peak = nt
                    except Exception:
                        pass

                    # CPU del processo: QUESTA chiamata BLOCCA per sample_s.
                    # NON aggiungere time.sleep dopo.
                    cpu_p = proc.cpu_percent(interval=self.sample_s)
                    self._cpu_sum += cpu_p
                    self._cpu_n += 1
                    if cpu_p > self._cpu_peak:
                        self._cpu_peak = cpu_p

                    # Opzionale: processo + figli (non blocca; usa delta dall'ultima chiamata)
                    if self.include_children:
                        rssc = rss_mb
                        cpuc = cpu_p
                        for c in safe_children():
                            try:
                                rssc += c.memory_info().rss / (1024 * 1024)
                            except Exception:
                                pass
                            try:
                                cpuc += c.cpu_percent(interval=None)
                            except Exception:
                                pass

                        self._rssc_sum += rssc
                        self._rssc_n += 1
                        if rssc > self._rssc_peak:
                            self._rssc_peak = rssc

                        self._cpuc_sum += cpuc
                        self._cpuc_n += 1
                        if cpuc > self._cpuc_peak:
                            self._cpuc_peak = cpuc

                except Exception:
                    # race conditions / processo terminato / permessi ecc.
                    pass

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_and_write(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2, int(self.sample_s * 3)))

        rss_avg = self._rss_sum / self._rss_n if self._rss_n else None
        cpu_avg = self._cpu_sum / self._cpu_n if self._cpu_n else None

        rssc_avg = self._rssc_sum / self._rssc_n if self._rssc_n else None
        cpuc_avg = self._cpuc_sum / self._cpuc_n if self._cpuc_n else None

        cpu_time_s = None
        try:
            if psutil is not None:
                ct = psutil.Process(os.getpid()).cpu_times()
                cpu_time_s = float(ct.user + ct.system)
        except Exception:
            pass

        # normalizzazione: utile se confronti macchine diverse
        n_cpus = self._n_cpus or 1
        cpu_avg_norm = (cpu_avg / n_cpus) if cpu_avg is not None else None
        cpu_peak_norm = (self._cpu_peak / n_cpus) if self._cpu_n else None

        cpuc_avg_norm = (cpuc_avg / n_cpus) if cpuc_avg is not None else None
        cpuc_peak_norm = (self._cpuc_peak / n_cpus) if self._cpuc_n else None

        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "sample_s": self.sample_s,
            "n_cpus": int(n_cpus),

            "peak_rss_mb": round(self._rss_peak, 2),
            "avg_rss_mb": round(rss_avg, 2) if rss_avg is not None else None,

            # CPU% in stile top: 100% = 1 core pieno
            "cpu_avg_percent": round(cpu_avg, 2) if cpu_avg is not None else None,
            "cpu_peak_percent": round(self._cpu_peak, 2) if self._cpu_n else None,

            # CPU normalizzata (0-100 ~ saturazione macchina)
            "cpu_avg_percent_norm": round(cpu_avg_norm * 100, 2) if cpu_avg_norm is not None else None,
            "cpu_peak_percent_norm": round(cpu_peak_norm * 100, 2) if cpu_peak_norm is not None else None,

            "num_threads_peak": int(self._threads_peak),

            # opzionale: processo + figli
            "include_children": self.include_children,
            "peak_rss_mb_with_children": round(self._rssc_peak, 2) if self._rssc_n else None,
            "avg_rss_mb_with_children": round(rssc_avg, 2) if rssc_avg is not None else None,
            "cpu_avg_percent_with_children": round(cpuc_avg, 2) if cpuc_avg is not None else None,
            "cpu_peak_percent_with_children": round(self._cpuc_peak, 2) if self._cpuc_n else None,
            "cpu_avg_percent_norm_with_children": round(cpuc_avg_norm * 100, 2) if cpuc_avg_norm is not None else None,
            "cpu_peak_percent_norm_with_children": round(cpuc_peak_norm * 100, 2) if cpuc_peak_norm is not None else None,

            "cpu_time_s": round(cpu_time_s, 2) if cpu_time_s is not None else None,

            **self.meta,
        }

        self.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class SystemMonitor:
    """
    Monitora risorse di SISTEMA (macchina intera).
    Salva:
      - cpu_avg_percent, cpu_peak_percent   (0-100%)
      - ram_used_avg_mb, ram_used_peak_mb
      - ram_percent_avg, ram_percent_peak
      - swap_used_avg_mb, swap_used_peak_mb (se disponibile)
      - loadavg (Linux, se disponibile)
    """
    def __init__(self, out_json: Path, sample_s: float = 0.5, meta: dict | None = None):
        self.out_json = Path(out_json)
        self.sample_s = float(sample_s)
        self.meta = meta or {}

        self._stop_evt = threading.Event()
        self._thread = None

        self._cpu_sum = 0.0
        self._cpu_n = 0
        self._cpu_peak = 0.0

        self._ram_used_sum = 0.0
        self._ram_used_n = 0
        self._ram_used_peak = 0.0

        self._ram_pct_sum = 0.0
        self._ram_pct_n = 0
        self._ram_pct_peak = 0.0

        self._swap_used_sum = 0.0
        self._swap_used_n = 0
        self._swap_used_peak = 0.0

        self._load1_peak = None
        self._load5_peak = None
        self._load15_peak = None

        self._n_cpus = None
        self._total_ram_mb = None

    def start(self):
        self.out_json.parent.mkdir(parents=True, exist_ok=True)

        self._stop_evt.clear()
        self._thread = None

        self._cpu_sum = 0.0
        self._cpu_n = 0
        self._cpu_peak = 0.0

        self._ram_used_sum = 0.0
        self._ram_used_n = 0
        self._ram_used_peak = 0.0

        self._ram_pct_sum = 0.0
        self._ram_pct_n = 0
        self._ram_pct_peak = 0.0

        self._swap_used_sum = 0.0
        self._swap_used_n = 0
        self._swap_used_peak = 0.0

        self._load1_peak = self._load5_peak = self._load15_peak = None

        if psutil is None:
            return

        self._n_cpus = psutil.cpu_count(logical=True) or 1
        try:
            vm = psutil.virtual_memory()
            self._total_ram_mb = vm.total / (1024 * 1024)
        except Exception:
            self._total_ram_mb = None

        # Warmup: inizializza la misura CPU di sistema
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        def loop():
            while not self._stop_evt.is_set():
                try:
                    # CPU sistema (0-100)
                    cpu_p = psutil.cpu_percent(interval=self.sample_s)
                    self._cpu_sum += cpu_p
                    self._cpu_n += 1
                    if cpu_p > self._cpu_peak:
                        self._cpu_peak = cpu_p

                    # RAM
                    vm = psutil.virtual_memory()
                    used_mb = (vm.total - vm.available) / (1024 * 1024)
                    self._ram_used_sum += used_mb
                    self._ram_used_n += 1
                    if used_mb > self._ram_used_peak:
                        self._ram_used_peak = used_mb

                    self._ram_pct_sum += float(vm.percent)
                    self._ram_pct_n += 1
                    if float(vm.percent) > self._ram_pct_peak:
                        self._ram_pct_peak = float(vm.percent)

                    # SWAP (se c'è)
                    try:
                        sm = psutil.swap_memory()
                        swap_used_mb = sm.used / (1024 * 1024)
                        self._swap_used_sum += swap_used_mb
                        self._swap_used_n += 1
                        if swap_used_mb > self._swap_used_peak:
                            self._swap_used_peak = swap_used_mb
                    except Exception:
                        pass

                    # Loadavg (Linux/macOS)
                    try:
                        la1, la5, la15 = os.getloadavg()
                        self._load1_peak = la1 if self._load1_peak is None else max(self._load1_peak, la1)
                        self._load5_peak = la5 if self._load5_peak is None else max(self._load5_peak, la5)
                        self._load15_peak = la15 if self._load15_peak is None else max(self._load15_peak, la15)
                    except Exception:
                        pass

                except Exception:
                    pass

        import os
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_and_write(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2, int(self.sample_s * 3)))

        cpu_avg = self._cpu_sum / self._cpu_n if self._cpu_n else None
        ram_used_avg = self._ram_used_sum / self._ram_used_n if self._ram_used_n else None
        ram_pct_avg = self._ram_pct_sum / self._ram_pct_n if self._ram_pct_n else None
        swap_used_avg = self._swap_used_sum / self._swap_used_n if self._swap_used_n else None

        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "sample_s": self.sample_s,
            "n_cpus": int(self._n_cpus) if self._n_cpus else None,
            "total_ram_mb": round(self._total_ram_mb, 2) if self._total_ram_mb is not None else None,

            "cpu_avg_percent": round(cpu_avg, 2) if cpu_avg is not None else None,
            "cpu_peak_percent": round(self._cpu_peak, 2) if self._cpu_n else None,

            "ram_used_avg_mb": round(ram_used_avg, 2) if ram_used_avg is not None else None,
            "ram_used_peak_mb": round(self._ram_used_peak, 2) if self._ram_used_n else None,
            "ram_percent_avg": round(ram_pct_avg, 2) if ram_pct_avg is not None else None,
            "ram_percent_peak": round(self._ram_pct_peak, 2) if self._ram_pct_n else None,

            "swap_used_avg_mb": round(swap_used_avg, 2) if swap_used_avg is not None else None,
            "swap_used_peak_mb": round(self._swap_used_peak, 2) if self._swap_used_n else None,

            "load1_peak": round(self._load1_peak, 3) if self._load1_peak is not None else None,
            "load5_peak": round(self._load5_peak, 3) if self._load5_peak is not None else None,
            "load15_peak": round(self._load15_peak, 3) if self._load15_peak is not None else None,

            **self.meta,
        }

        self.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")