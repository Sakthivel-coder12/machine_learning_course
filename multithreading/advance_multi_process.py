from concurrent.futures import ProcessPoolExecutor
import time
def print_numbers(numbers):
    time.sleep(1)
    return f"number : {numbers}"

num = [1,2,3,4,5,6,7,8,9]

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as executor:
        result = executor.map(print_numbers,num)
        
    for i in result:
        print(i)