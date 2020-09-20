#!/usr/bin/env python
import multiprocessing as mp
import queue # imported for using queue.Empty exception
import numpy as np
import time
import pprint
import gpu_linalg

def get_eigsys(lock, mat, tasks_to_accomplish, tasks_that_are_done):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will
                raise queue.Empty exception if the queue is empty.
                queue(False) function would do the same task also.
            '''
            task_ampl = tasks_to_accomplish.get_nowait()
            lock.acquire()
            w, v = gpu_linalg.eig(mat, left=False, right=True)
            lock.release()

        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion
                message to task_that_are_done queue
            '''
            tasks_that_are_done.put((task_ampl,w,v))
            time.sleep(.5)
    return True

def main():
    #Default process for linux is forked, which CUDA does not accept. Change to spawn
    mp.set_start_method('spawn')

    size = 4
    ampls = np.linspace(10.0, 30.0, 5)
    number_of_task = len(ampls)
    number_of_processes = number_of_task
    tasks_to_accomplish = mp.Queue()
    tasks_that_are_done = mp.Queue()
    processes = []

    for h in ampls:
        tasks_to_accomplish.put(h)

    # creating processes
    lock = mp.Lock()
    for h in ampls:
        #Generate Floquet Matrix here. Random for now
        floquet_matrix = np.random.random((size,size)) + (1j) * np.random.random((size,size))
        p = mp.Process(target=get_eigsys, args=(lock, floquet_matrix, tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    while not tasks_that_are_done.empty():
        amp, evals, evecs = tasks_that_are_done.get()
        print("\n**********************************\n")
        print("Ampl:", amp, "\n")
        print("Evals:", evals, "\n\n")
        print("Evecs:",)
        pprint.pprint(evecs)
    return True

if __name__ == '__main__':
    main()