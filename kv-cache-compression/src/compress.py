import numpy as np 

def compute_svd(K_head, V_head):
    Uk, Sk, Vtk = np.linalg.svd(K_head, full_matrices = False)
    Uv, Sv, Vtv = np.linalg.svd(V_head, full_matrices = False)
    #plt.plot(Sk)
    #plt.plot(Sv)
    #plt.show() # see how fast S decays
    return Uk, Sk, Vtk, Uv, Sv, Vtv
