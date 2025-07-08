import logging

# Configure logging globally
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt= "%Y-%m-%d  %H:%M:%S",
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()   
    ]
)

def log_exception_error_level():
    logger = logging.getLogger("log")
    try:
        1 / 0
    except Exception as e:
        logger.error('Exception occurred',exc_info=True)
    return logger

logger = log_exception_error_level()

