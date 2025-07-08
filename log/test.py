from logger import logging as l   ## Here it is customized file logger in that i have written the defination of the logging format 

def add(num1,num2):
    l.debug("The addition is done")
    return num1 + num2

l.info("The addition function is called")
add(13,18)