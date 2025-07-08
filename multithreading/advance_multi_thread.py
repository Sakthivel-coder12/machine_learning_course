from concurrent.futures import ThreadPoolExecutor

import time

def print_numbers(numbers):
    time.sleep(1)   ## In this case even though we have given the l min sleep it will allocate all the numbers at once and exeucte very fast and i will be applicable for the advance_multi_Procesing also 
    return f"number : {numbers}"

num = [1,2,3,4,5,6,7,8,9]

with ThreadPoolExecutor(max_workers=3) as executor:
    result = executor.map(print_numbers,num)
for i in result:
    print(i)

