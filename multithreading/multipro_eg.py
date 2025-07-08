import multiprocessing
import time
import sys
import math

sys.set_int_max_str_digits(100000)


def compute_factorial(number):
        print(f"The factorial of number is {number}")
        result = math.factorial(number)
        print(f"Factorial of {number} is {result}") 
        return result

    
if  __name__ == "__main__":
    numbers = [1000,9000,8000]
    s = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(compute_factorial,numbers)
    end = time.time()

    print(f"Result :{results}") 
    print(f"Time taken {end - s:.6f} seconds...")