import os
import json
import logging
import time

from utils.device import get_device


def _get_device_str():
    d = get_device()
    return f'{d.type}_{d.index if d.index is not None else os.getpid()}'


class JsonlFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "message": record.getMessage(),
            "device": _get_device_str(),
            "timestamp": time.time(),
        }
        if hasattr(record, "extra"):
            log_record.update(record.extra)
        return json.dumps(log_record)


def get_logger(exp_type, model_conf, log_path, level=logging.INFO):
    # Build structured directory path
    run_ts = int(time.time())
    run_dir = os.path.join(log_path, exp_type, model_conf, str(run_ts))
    os.makedirs(run_dir, exist_ok=True)  # Create dirs recursively if needed

    # Build log file path
    log_f = os.path.join(run_dir, f"{_get_device_str()}.log")

    logger = logging.getLogger(log_f)
    logger.setLevel(level)
    handler = logging.FileHandler(log_f)
    handler.setFormatter(JsonlFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
