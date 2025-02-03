from loguru import logger

# Configuration du logger
logger.add("logs/app.log", rotation="1 MB", level="INFO", format="{time} - {level} - {message}")

def get_logger():
    return logger
