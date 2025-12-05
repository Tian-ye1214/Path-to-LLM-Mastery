import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

_logger = None
_current_log_file = None


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",     # 青色
        logging.INFO: "\033[0m",       # 白色(默认)
        logging.WARNING: "\033[33m",   # 黄色
        logging.ERROR: "\033[31m",     # 红色
        logging.CRITICAL: "\033[91m",  # 亮红色
    }
    RESET = "\033[0m"
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
    
    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def get_logger() -> logging.Logger:
    """获取全局logger实例"""
    global _logger
    if _logger is None:
        _logger = logging.getLogger("AgentDemo")
        _logger.setLevel(logging.DEBUG)

        if not _logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_format = ColorFormatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            _logger.addHandler(console_handler)
    
    return _logger


def setup_task_logger(task_name: str = "task") -> logging.Logger:
    """
    为新任务设置日志，创建新的日志文件
    
    Parameters:
        task_name: 任务名称，用于日志文件命名
    
    Returns:
        配置好的logger实例
    """
    global _logger, _current_log_file

    safe_task_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in task_name)
    safe_task_name = safe_task_name[:50]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{safe_task_name}_{timestamp}.log"
    log_filepath = LOG_DIR / log_filename

    _logger = logging.getLogger("AgentDemo")
    _logger.setLevel(logging.DEBUG)

    _logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = ColorFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    _logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    _logger.addHandler(file_handler)
    
    _current_log_file = log_filepath
    _logger.info(f"日志文件已创建: {log_filepath}")
    
    return _logger


def get_current_log_file() -> str:
    """获取当前日志文件路径"""
    global _current_log_file
    return str(_current_log_file) if _current_log_file else None


def close_logger():
    """关闭当前logger的所有handler"""
    global _logger
    if _logger:
        for handler in _logger.handlers[:]:
            handler.close()
            _logger.removeHandler(handler)


def _flush_handlers():
    """刷新所有handler的缓冲区"""
    logger = get_logger()
    for handler in logger.handlers:
        handler.flush()

def debug(msg, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)
    _flush_handlers()

def info(msg, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)
    _flush_handlers()

def warning(msg, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)
    _flush_handlers()

def error(msg, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)
    _flush_handlers()

def critical(msg, *args, **kwargs):
    get_logger().critical(msg, *args, **kwargs)
    _flush_handlers()
