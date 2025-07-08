import multiprocessing
import time

def print_number():
    for i in range(5):
        time.sleep(2)
        print(f"Number :{i}")
def print_letter():
    for i in "abcde":
        time.sleep(2)
        print(f"Letter : {i}")
if __name__ == "__main__":
    t1 = multiprocessing.Process(target=print_number)
    t2 = multiprocessing.Process(target=print_letter)
    t = time.time()

    t1.start()
    t2.start()

    t1.join()
    t2.join()


    f = time.time() - t 
    print(f"The time is {f:.6f}")