"""
log_util.py —— 日志工具：按小时分文件写入日志，同时输出到终端

目录结构:
    logs/YYYYMMDD/MMDDHH.log

    - 文件名表示该小时段，如 logs/20261104/110414.log 表示 11 月 4 日 14 点的日志
    - 应用启动时立即创建当前小时的文件
    - 之后每逢整点自动切换到新文件（如 logs/20261104/110415.log）
    - 时间统一按北京时间（UTC+8），避免容器默认 UTC 导致小时对不上
"""
import os
import sys
import logging
import threading
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(BASE_DIR, "logs")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 北京时间（中国无夏令时，固定 UTC+8）
TZ = timezone(timedelta(hours=8))
LOG_FILE_SUFFIX = ".log"


def now_local():
    return datetime.now(TZ)

# 单条日志中长文本（请求体/响应体）的最大保留长度
MAX_TEXT_LEN = 3000


class HourlyFileHandler(logging.Handler):
    """按小时切分的文件处理器：每满一小时新建一个 txt 文件"""

    def __init__(self, log_root=LOG_ROOT, encoding="utf-8"):
        super().__init__()
        self.log_root = log_root
        self.encoding = encoding
        self._stream = None
        self._hour = None
        self._lock = threading.Lock()
        self.current_path = None

    @staticmethod
    def _hour_key():
        return now_local().strftime("%Y%m%d%H")

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = None

    def _open_new_file(self):
        self._close_stream()
        now = now_local()
        day_dir = os.path.join(self.log_root, now.strftime("%Y%m%d"))
        os.makedirs(day_dir, exist_ok=True)
        path = os.path.join(day_dir, now.strftime("%m%d%H") + LOG_FILE_SUFFIX)
        self._stream = open(path, "a", encoding=self.encoding)
        self._hour = self._hour_key()
        self.current_path = path

    def emit(self, record):
        try:
            with self._lock:
                if self._stream is None or self._hour != self._hour_key():
                    self._open_new_file()
                self._stream.write(self.format(record) + "\n")
                self._stream.flush()
        except Exception:
            self.handleError(record)

    def open_now(self):
        """应用启动时立即创建日志文件（不等第一条日志）"""
        with self._lock:
            if self._stream is None:
                self._open_new_file()
        return self.current_path

    def close(self):
        with self._lock:
            self._close_stream()
        super().close()


def setup_logging(level=logging.INFO):
    """初始化日志（重复调用不会重复添加 handler），返回 app logger"""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_zshchj_handler", False):
            root.removeHandler(h)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    # 日志时间同样按北京时间输出
    formatter.converter = lambda ts: datetime.fromtimestamp(ts, TZ).timetuple()

    file_handler = HourlyFileHandler()
    file_handler.setFormatter(formatter)
    file_handler._zshchj_handler = True

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler._zshchj_handler = True

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(level)

    current_path = file_handler.open_now()
    logger = logging.getLogger("app")
    logger.info("日志已启动 | 目录=%s | 当前文件=%s", LOG_ROOT, current_path)
    return logger


def get_logger(name):
    return logging.getLogger(name)


def brief(text, limit=MAX_TEXT_LEN):
    """截断超长文本，便于日志阅读"""
    if text is None:
        return "-"
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(truncated, total {len(text)} chars)"
