#!/usr/bin/env python
import multiprocessing as mp
import queue  # imported for using queue.Empty exception
import numpy as np
import time
import pprint
import gpu_linalg


def get_eigsys(lock, mat, task_params, task_output):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will
                raise queue.Empty exception if the queue is empty.
                queue(False) function would do the same task also.
            '''
            task_ampl = task_params.get_nowait()
            lock.acquire()
            w, v = gpu_linalg.eig(mat, left=False, right=True, verbose=False)
            lock.release()

        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion
                message to task_that_are_done queue
            '''
            task_output.put((task_ampl, w, v))
            time.sleep(.5)
    return True


def run_floquet(size=5, omega=30.0, amps=20.0):
    # Default process for linux is forked, which CUDA does not accept. Change to spawn
    mp.set_start_method('spawn')

    amplitude_queue = mp.Queue()
    eigsystems_queue = mp.Queue()

    # Create task queue
    for h in [amps]:
        amplitude_queue.put(h)

    # Creating floquet translation matrix and diagonalization processes
    diag_processes = []
    lock = mp.Lock()
    for h in [amps]:
        # Generate Periodic Translation matrix here. Random for now
        umat = np.random.random((size, size)) + 1j * np.random.random((size, size))
        proc_diag = mp.Process(target=get_eigsys, args=(lock, umat, amplitude_queue, eigsystems_queue))
        diag_processes.append(proc_diag)
        proc_diag.start()

    # completing diagonalization process
    for p in diag_processes:
        p.join()

    # print the output
    while not eigsystems_queue.empty():
        amp, evals, evecs = eigsystems_queue.get()
        print("\n**********************************\n")
        print("Ampl:", amp, "\n")
        print("Evals:", evals, "\n\n")
        print("Evecs:", )
        pprint.pprint(evecs)
    return True


if __name__ == '__main__':
    run_floquet(size=3, omega=30.0, amps=np.linspace(10.0, 30.0, 5))