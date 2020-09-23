#!/usr/bin/env python
import multiprocessing as mp
import queue  # imported for using queue.Empty exception
import numpy as np
import time
# import matplotlib.pyplot as plt
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


def floquet_col(ic, h, p):
    freq = p["omega"]
    und, drv = p["hamilt"]
    T = 2.0 * np.pi / freq
    times = np.linspace(0.0, T, 1000, endpoint=True)
    sol = odeintw(lambda psi, t, undriven, driven, amp, f: -1j * np.dot(undriven + driven * amp * np.sin(f * t), psi),
                  ic, times, args=(und, drv, h, freq))
    return sol[-1]


def run_floquet(p):
    """
    Obtains Floquet Eigenvalues and Eigenvectors of a Hamiltonian matrix of the type
    \\begin{equation}
    H(t) = H_0 + h_0 \\cos{\\omega t} H_1,
    \\end{equation}
    by evolving over one time period $T=2\\pi/\\omega$ each column of a  unit matrix as
    an initial condition of the Schrodinger equation

    H |\\psi\\rangle = i \\partial_t |\\psi\rangle,

    in order to obtain the monodromy matrix, which has eigenvalues $e^{i\\epsilon T}$ for quasienergies $\\epsilon$

    Parameters:     p : Dictionary of the type {"omega": omega, "amps": amps, "hamilt": (ham0, ham1)}, where
                        omega : Drive Frequency
                        amps  : Iterable of drive amplitudes
                        ham0  : Matrix $H_0$ as numpy array of dimensions = 2
                        ham1  : Matrix $H_1$ as numpy array of dimensions = 2

    Returns:        esys : Iterable of the same dimensions as p["amps"], where
                           each iteration returns a tuple (a, E, U) where
                           a : amplitude taken from input iterable
                           E : array of Floquet quasienergies $\\epsilon$ of the same size as $H_0$[:,0]
                           U : numpy array of column ordered eigenvectors of the size $H_0$
    """

    h0, _ = p["hamilt"]
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
    idt = np.eye(rows)
    start_matrix = idt + 1j * np.zeros_like(idt)
    for h in params["amps"]:
        # Generate Periodic Translation matrix here.
        with mp.Pool(processes=nprocs) as p:
            umat_t = p.starmap(floquet_col, [(ic, h, params) for ic in start_matrix])
            umat = np.array(umat_t).T
        proc_diag = mp.Process(target=get_eigsys, args=(lock, umat, amplitude_queue, eigsystems_queue))
        diag_processes.append(proc_diag)
        proc_diag.start()

    # completing diagonalization process
    for proc in diag_processes:
        proc.join()

    # Convert queue to list
    eigsystems_list = []
    while not eigsystems_queue.empty():
        eigsystems_list.append(eigsystems_queue.get())

    eigsystems_list.sort(key=lambda tup: tup[0])  # sorts in place by amplitude

    # sort the eigsystem according to orthogonality of adjacent eigenvectors
    count = 1
    while count < len(eigsystems_list):
        amp, evals, evecs = eigsystems_list[count]
        amp_prev, evals_prev, evecs_prev = eigsystems_list[count - 1]

        # This should be an identity matrix
        # Unless avoided crossings have shuffled the evecs
        # In which case it will be a shuffled identity matrix
        dot_prods = np.round(np.dot(evecs_prev.conj().T, evecs))
        # Find out which unique off diagonal indices are nonzero
        # These correspond to the evecs that have shuffled
        swap_idx = np.vstack(np.nonzero(np.triu(dot_prods))).T
        # Now, unshuffle the evecs
        for a, b in swap_idx:
            evecs[:, [b, a]] = evecs[:, [a, b]]
            evals[[b, a]] = evals[[a, b]]
        count += 1
        eigsystems_list[count] = amp, evals, evecs

    return eigsystems_list


# def harmonic_osc_floquet_evals():
#    maxlev = 20
#    lamb = np.linspace(0.0, 10.0, 1000)
#    w = 2.11
#    qens = np.array([((2 * n + 1) + lamb ** 2 / (2 * (w ** 2 - 4))) % w for n in range(maxlev)])
#    for n in range(maxlev):
#        plt.scatter(lamb, qens[n, :], s=5, c='b')
#    plt.show()


if __name__ == '__main__':
    omega = 40.0
    amps = np.linspace(60.0, 60.3, 4)
    ham0 = 0.5 * np.array([[0, 1], [1, 0]])
    ham1 = 0.5 * np.array([[1, 0], [0, -1]])
    params = {"omega": omega, "amps": amps, "hamilt": (ham0, ham1)}
    run_floquet(params)
