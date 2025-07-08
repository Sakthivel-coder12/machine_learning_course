import logging

logging.basicConfig(
    level= logging.DEBUG,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt= "%Y-%m-%d  %H-%M-%S",
    handlers=[
        logging.FileHandler("app1.log")
    ]
)

logger1 = logging.getLogger("Arithemethic")
logger1.setLevel(logging.DEBUG)

def add(a,b):
    result = a + b
    logger1.debug(f"The addition is {a} + {b} = {result}")
    return result


def sub(a,b):
    result = a - b
    logger1.debug(f"The substraction is {a} - {b} = {result}")
    return result


def mul(a,b):
    result = a * b
    logger1.debug(f"The multiplication is {a} * {b} = {result}")
    return result

def div(a,b):
    try:
        result = a / b
        logger1.debug(f"The division is {a} / {b} = {result}")
        return result
    except ZeroDivisionError as z:
        logger1.error("The Division by zero error...!")
        return None
add(10,15)
sub(10,5)
mul(10,10)
div(10,0)