# gjr_recursion.pyx
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def gjr_recursion(np.ndarray[np.float64_t] resids, np.ndarray[np.float64_t] params, double sigma):
    cdef double cont = params[0]
    cdef double rho = params[1]
    cdef double phi_p = params[2]
    cdef double phi_n = params[3]
    cdef double t1 = 10e-5
    cdef int n = resids.shape[0]
    cdef np.ndarray[np.float64_t] sprocess = np.zeros(n, dtype=np.float64)
    cdef double backcast = max(cont / (1 - rho - (phi_p + phi_n) / 2), t1)
    sprocess[0] = max(backcast, t1)
    cdef int t

    for t in range(1, n):
        sprocess[t] = cont + rho * sprocess[t-1] + ((phi_p if resids[t-1] > 0 else phi_n) * resids[t-1] ** 2 / (2 * sigma ** 2))
        sprocess[t] = max(sprocess[t], t1)

    return sprocess