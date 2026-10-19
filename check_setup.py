import sys

from src import config
from src.logger import get_logger

logger = get_logger("check_setup")

config.ensure_dirs()
logger.info("Python version: %s", sys.version.split()[0])
logger.info("Upload dir: %s", config.UPLOAD_DIR)
logger.info("Index dir: %s", config.INDEX_DIR)
logger.info("Embedding model: %s", config.EMBEDDING_MODEL)
logger.info("Setup OK")