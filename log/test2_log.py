import logging 


logger1 = logging.getLogger("module1")
logger1.setLevel(logging.DEBUG)


logger2 = logging.getLogger("module2")
logger2.setLevel(logging.WARNING)


level = logging.basicConfig(
    level = logging.DEBUG,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger1.debug("The debug for m1")
logger1.warning("The war for m1")


logger2.error("The error for module2")