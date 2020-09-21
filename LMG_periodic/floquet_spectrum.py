#!/usr/bin/env python
import multiprocessing as mp
import queue  # imported for using queue.Empty exception
import numpy as np
import time
import pprint
import gpu_linalg
from odeintw import odeintw

nprocs = 2

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

def floquet_col(ic,h, params):
    w = params["omega"]
    h0, h1 = params["hamilt"]
    T = 2.0 * np.pi/w
    times = np.linspace(0.0, T, 1000, endpoint=True)
    sol = odeintw(lambda psi, t, h0, h1, h, w: -1j * np.dot(h0 + h1 * h * np.sin(w * t),psi),
                                                                            ic, times, args=(h0, h1,h,w))
    return sol[-1]

def run_floquet(params):
    h0, _ = params["hamilt"]
    rows, _ = np.shape(h0)
    # Default process for linux is forked, which CUDA does not accept. Change to spawn
    mp.set_start_method('spawn')
    amplitude_queue = mp.Queue()
    eigsystems_queue = mp.Queue()

    # Create task queue
    for h in params["amps"]:
        amplitude_queue.put(h)

    # Creating floquet translation matrix and diagonalization processes
    diag_processes = []
    lock = mp.Lock()
    id = np.eye(rows)
    start_matrix = id + 1j * np.zeros_like(id)
    for h in params["amps"]:
        # Generate Periodic Translation matrix here.
        with mp.Pool(processes=nprocs) as p:
            umat_t = p.starmap(floquet_col,[(ic, h, params) for ic in start_matrix])
            umat = np.array(umat_t).T
        proc_diag = mp.Process(target=get_eigsys, args=(lock, umat, amplitude_queue, eigsystems_queue))
        diag_processes.append(proc_diag)
        proc_diag.start()

    # completing diagonalization process
    for p in diag_processes:
        p.join()

    #Convert queue to list
    eigsystems_list = []
    while not eigsystems_queue.empty():
        eigsystems_list.append(eigsystems_queue.get())

    eigsystems_list.sort(key=lambda tup: tup[0])  # sorts in place by amplitude

    # sort the eigsystem according to orthogonality of adjacent eigenvectors
    count = 1

    while count < len(eigsystems_list):
        amp, evals, evecs = eigsystems_list[count]
        amp_prev, evals_prev, evecs_prev = eigsystems_list[count-1]
        pmat = np.round(np.dot(evecs.T.conjugate(), evecs_prev))
        #Complete antipermutation sorting.
        count += 1

    return True

if __name__ == '__main__':
    omega = 40.0
    amps = np.linspace(60.0, 60.3, 4)
    h0 = 0.5 * np.array([[0, 1], [1, 0]])
    h1 = 0.5 * np.array([[1, 0], [0, -1]])
    params = {"omega": omega, "amps": amps, "hamilt": (h0, h1)}
    run_floquet(params)