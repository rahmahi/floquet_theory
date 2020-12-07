# 3 Dec CODE

# note : here  I have tried to introduce the average of phasefunc and its variations
#              I also have changed the code in such a way that frequecy get varies
#              instead of amplitude of symmetry breaking field.

import numpy as np
from odeintw import odeintw
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numpy.linalg import multi_dot, norm, eig
import math
import time
start = time.time()

# 2.40482555769577	5.52007811028631	8.65372791291101	11.7915344390142	14.9309177084877

N = 40
kacN = N
h0 = 0.1
omega = 600
w = omega
tsteps = 500
hsteps = 100
h1 = 1/(2. * kacN) * omega * 0.0
h2 = 1/(2. * kacN) * omega * 15.5
print(' h ranges in',h1,'&',h2)
hs = np.linspace(h1, h2, hsteps)
hpp = (np.ones((N+1,len(hs))) * hs).T
psi = np.eye(N+1) + 1j * np.zeros((N+1,N+1))

H0 = np.zeros((N+1,N+1))
H1 = np.zeros((N+1,N+1))

filename1 = 'mfd_fl_h_0-15p5_varies'+str(N)+'real.jpeg'
filename2 = 'mfd_fl_h_0-15p5_varies'+str(N)+'imag.jpeg'
filename3 = 'mfd_fl_h_0-15p5_varies'+str(N)+'avg_real.jpeg'
filename4 = 'mfd_fl_h_0-15p5_varies'+str(N)+'avg_imag.jpeg'

def delta_func(x,y):
        if ( x == y ):
            return 1.0
        else:
            return 0.0
        
def floq_jac(psi0, t, h, h0, H1, H0, w):    
    drive = h0 + h * np.cos(w * t)
    jac = 1j * N * (H0 + drive * H1)    
    return jac
        
def floq_func(psi0, t, h, h0, H1, H0, w):
    floq_h = np.dot(floq_jac(psi0, t, h, h0, H1, H0, w), psi0)
    return floq_h

def floq_evolv(psi,H0,H1,h,w):
    T = 2 * np.pi/w                                    
    t = np.linspace(0,2 * np.pi/w,tsteps)                      
    floqEvolution_mat = np.zeros((N+1,N+1)) + 1j * np.zeros((N+1,N+1))

    
    for m in np.arange(N):
        psi0 = psi[m]
        psi_t = odeintw(floq_func,psi0,t,args=(h, h0, H1, H0, w))
        floqEvolution_mat[m] = psi_t[-1]     
    #print('unitary or not', np.dot(np.conjugate(floqEvolution_mat).T,floqEvolution_mat).real)    
    evals, evecs = eig(floqEvolution_mat)
    phasefunc = 1j  * np.log(evals + 1j * 0) * w/(2 * np.pi)
    return phasefunc

if __name__ == '__main__':
    nprocs = 15
    p = Pool(processes = nprocs)          
    print("running for N = ", N, "with",nprocs,"processors")
    start = time.time()       
   
    spin = 0.5
    #s = np.arange(-spin,spin+1) 
    s = -0.5 + (1./N) * np.arange(N+1)
    H0 = np.diagflat(0.5 * s **2/kacN)
    #H0 = np.diagflat(0.5 * s * (s+1))
    #H0 = np.array([0.5 *  s[i] * s[j]for i in range(N) for j in range(N)]).reshape((N,N))   
    for i in range(N+1):
        for j in range(N+1):
            H1[i][j] = 0.5 * (np.sqrt(spin * (spin+1) - s[i] * (s[i]+1)) * delta_func(s[j],s[i]+1)\
                              + np.sqrt(spin * (spin+1) - s[i] * (s[i]-1)) * delta_func(s[j],s[i]-1))
    
    #print('H0 is----- \n',H0,'\nH1 is----- \n',H1,'\n')
    
    data = p.starmap(floq_evolv,[(psi,H0,H1,h,w) for h in hs])    
    
    #print('real plot ---')
    
    title = "Mean field Floquet dynamics : N =" + str(N) + 'with real part of data'
    plt.figure(figsize = (12,6))
    for xx in np.arange(len(hs)):
        plt.scatter(2 * hpp[xx] * kacN /w, data[xx].real,\
                    color = 'blue', marker='.', s = 0.8)
    plt.title(title)    
    plt.xlabel("2h kacN/omega")
    plt.ylabel("phase function")
    plt.grid()
    plt.savefig(filename1,dpi=500)
    
    #print('imaginary plot ---')
    
    title = "Mean field Floquet dynamics : N =" + str(N) + 'with imaginary part of data'
    plt.figure(figsize = (12,6))
    for xx in np.arange(len(hs)):
        plt.scatter(2 * hpp[xx] * kacN /w, data[xx].imag,\
                    color = 'blue', marker='.', s = 0.8)
    plt.title(title)    
    plt.xlabel("2h kacN/omega")
    plt.ylabel("phase function")
    plt.grid()
    plt.savefig(filename2,dpi=500)
    
    #print('average real data plot ---')
    
    title = "Mean field Floquet dynamics : N =" + str(N) + 'with average real part of data'
    plt.figure(figsize = (12,6))
    print(np.shape(data),np.shape(hs))
    avg = np.zeros(len(hs))
    for i in np.arange(len(hs)):
        avg[i] = np.average(data[i].real)
    plt.plot(2 * hs * kacN/w, avg)
    plt.title(title)    
    plt.xlabel("2h kacN/omega")
    plt.ylabel("phase function")
    plt.grid()
    plt.savefig(filename3,dpi=500)
    
    #print('average imaginary data plot ---')
    
    title = "Mean field Floquet dynamics : N =" + str(N) + 'with average imaginary part of data'
    plt.figure(figsize = (12,6))
    print(np.shape(data),np.shape(hs))
    avg = np.zeros(len(hs))
    for i in np.arange(len(hs)):
        avg[i] = np.average(data[i].imag)
    plt.plot(2 * hs * kacN/w, avg)
    plt.title(title)    
    plt.xlabel("2h kacN/omega")
    plt.ylabel("phase function")
    plt.grid()
    plt.savefig(filename4,dpi=500)
    
    #print('Total time taken',(time.time()-start)/60,'minute')
