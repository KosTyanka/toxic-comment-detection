"""Wall-clock timing and sampled resident memory (including native allocations)."""
import os
import threading
from time import perf_counter
import psutil


class ResourceMonitor:
    """Sample absolute process RSS every 10ms; brief spikes may be missed."""
    def __enter__(self):
        self.process = psutil.Process(os.getpid())
        self.start_rss = self.peak_rss = self.process.memory_info().rss
        self.stop = threading.Event()
        self.started = perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self) -> None:
        while not self.stop.wait(0.01):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.seconds = perf_counter() - self.started
        self.stop.set()
        self.thread.join()
        self.peak_mb = self.peak_rss / 1024**2
        self.increment_mb = max(0, self.peak_rss - self.start_rss) / 1024**2
