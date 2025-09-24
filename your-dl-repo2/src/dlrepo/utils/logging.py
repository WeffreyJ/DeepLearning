import logging
def get_logger(name=None, level=logging.INFO):
    logger = logging.getLogger(name if name else __name__)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s", datefmt="%H:%M:%S")
        h.setFormatter(fmt); logger.addHandler(h)
    logger.setLevel(level); return logger
