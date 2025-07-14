import numpy as np
import sys
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=3)
import matplotlib
import matplotlib.pyplot as plt
import math
import pprint

n_x=5
n_y=5

dx=1.0/n_x
dy=1.0/n_y

Re=100

# -----------------------------------------------------------------------


def Lij(i, j):
    return (i - 1) * n_y + (j - 1)

# Boundary
isWall = np.zeros((n_y + 2, n_x + 2), dtype=np.bool)
isWall[ 0, :] = True
isWall[-1, :] = True
isWall[ :, 0] = True
isWall[ :,-1] = True
pprint.pprint(isWall)

# CSR
nxy = n_x * n_y
col = list()
ptr = [0]

for i in range(1, n_y + 1):
    for j in range(1, n_x + 1):
        count = 0

        if not isWall[i-1, j]:
            col.append(Lij(i-1, j))
            count += 1

        if not isWall[i, j-1]:
            col.append(Lij(i, j-1))
            count += 1

        if not isWall[i, j]:
            col.append(Lij(i, j))
            count += 1

        if not isWall[i, j+1]:
            col.append(Lij(i, j+1))
            count += 1

        if not isWall[i+1, j]:
            col.append(Lij(i+1, j))
            count += 1

        ptr.append(ptr[-1] + count)

col = np.array(col, dtype=np.int64)
ptr = np.array(ptr, dtype=np.int64)

# val = np.ones_like(col, dtype=np.int64)
A = np.zeros_like(col, dtype=np.float64)
Su = np.zeros(nxy, dtype=np.float64)
Sv = np.zeros(nxy, dtype=np.float64)
A_diagonal = np.zeros(nxy, dtype=np.float64)

# pprint.pprint(col)
# pprint.pprint(ptr)

from scipy.sparse import csr_matrix
csr = csr_matrix((val, col, ptr), shape=(nxy, nxy))
# A = csr_matrix.todense(csr)

# pprint.pprint(A)
# pprint.pprint(csr.indices)
# pprint.pprint(csr.indptr)

# k = 13
# print(csr.indptr[k], csr.indptr[k+1])
# print(csr.data[csr.indptr[k]:csr.indptr[k+1]])
# print(csr.indices[csr.indptr[k]:csr.indptr[k+1]])


def momentum_link_coefficients(u_star, v_star, u_face, v_face, p, A, Su, Sv, alpha):

    D_e = dy / (dx * Re)
    D_w = dy / (dx * Re)
    D_n = dx / (dy * Re)
    D_s = dx / (dy * Re)

    # interior cells on CSR Format
    for i in range(2, n_y):
        for j in range(2, n_x):
            k = Lij(i, j)

            F_e = dy * u_face[i, j]
            F_w = dy * u_face[i, j-1]
            F_n = dx * v_face[i-1, j]
            F_s = dx * v_face[i, j]

            Ae = D_e + max(0.0,-F_e)
            Aw = D_w + max(0.0, F_w)
            An = D_n + max(0.0,-F_n)
            As = D_s + max(0.0, F_s)
            Ap = Aw + Ae + An + As + (F_e-F_w) + (F_n-F_s)

            Su[k] = 0.5 * (p[i, j-1] - p[i, j+1]) * dx
            Sv[k] = 0.5 * (p[i+1, j] - p[i-1, j]) * dy

            # Under-Relaxation, Solving Equation System
            Ap = Ap / alpha
            Su[k] = Su[k] + (1 - alpha) * Ap * u_star[i, j]
            Sv[k] = Sv[k] + (1 - alpha) * Ap * v_star[i, j]

            A[ptr[k]:ptr[k+1]] = [An, Aw, Ap, Ae, As]
            A_diagonal[k] = Ap

    # left wall on CSR Format
    j = 1
    for i in range(2, n_y):
        k = Lij(i, j)

        F_e = dy * u_face[i, j]
        F_w = dy * u_face[i, j-1]
        F_n = dx * v_face[i-1, j]
        F_s = dx * v_face[i, j]

        Ae =   D_e + max(0.0,-F_e)
        Aw = 2*D_w + max(0.0, F_w)
        An =   D_n + max(0.0,-F_n)
        As =   D_s + max(0.0, F_s)
        Ap = Aw + Ae + An + As + (F_e - F_w) + (F_n - F_s)

        Su[k] = 0.5 * (p[i, j] - p[i, j+1]) * dx + D_w * u_star[i, j-1]
        Sv[k] = 0.5 * (p[i+1, j] - p[i-1, j]) * dy

        # Under-Relaxation, Solving Equation System
        Ap = Ap / alpha
        Su[k] = Su[k] + (1 - alpha) * Ap * u_star[i, j]
        Sv[k] = Sv[k] + (1 - alpha) * Ap * v_star[i, j]

        A[ptr[k]:ptr[k+1]] = [An, Ap, Ae, As]
        A_diagonal[k] = Ap

    #bottom wall on CSR Format
    i = n_y
    for j in range(2, n_x):
        k = Lij(i, j)

        Fe = dy * u_face[i, j]
        Fw = dy * u_face[i, j-1]
        Fn = dx * v_face[i-1, j]
        Fs = dx * v_face[i, j]

        Ae =   D_e + max(0.0,-F_e)
        Aw =   D_w + max(0.0, F_w)
        An =   D_n + max(0.0,-F_n)
        As = 2*D_s + max(0.0, F_s)
        Ap = A_w + A_e + A_n + A_s + (F_e - F_w) + (F_n - F_s)

        Su[k] = 0.5 * (p[i, j-1] - p[i, j+1]) * dx
        Sv[k] = 0.5 * (p[i, j] - p[i-1 ,j]) * dy + D_s * v_star[i+1, j]

        # Under-Relaxation, Solving Equation System
        Ap = Ap / alpha
        Su[k] = Su[k] + (1 - alpha) * Ap * u_star[i, j]
        Sv[k] = Sv[k] + (1 - alpha) * Ap * v_star[i, j]

        A[ptr[k]:ptr[k+1]] = [An, Aw, Ap, Ae]
        A_diagonal[k] = Ap

    #right wall on CSR Format
    j = n_x
    for i in range(2, n_y):
        k = Lij(i, j)

        F_e = dy * u_face[i, j]
        F_w = dy * u_face[i, j-1]
        F_n = dx * v_face[i-1, j]
        F_s = dx * v_face[i, j]

        Ae =   D_e + max(0.0,-F_e)
        Aw = 2*D_w + max(0.0, F_w)
        An =   D_n + max(0.0,-F_n)
        As =   D_s + max(0.0, F_s)
        Ap = Aw + Ae + An + As + (F_e - F_w) + (F_n - F_s)

        Su[k] = 0.5 * (p[i, j-1] - p[i, j]) * dx + D_w * u_star[i, j+1]
        Sv[k] = 0.5 * (p[i+1, j] - p[i-1, j]) * dy

        # Under-Relaxation, Solving Equation System
        Ap = Ap / alpha
        Su[k] = Su[k] + (1 - alpha) * Ap * u_star[i, j]
        Sv[k] = Sv[k] + (1 - alpha) * Ap * v_star[i, j]

        A[ptr[k]:ptr[k+1]] = [An, Aw, Ap, As]
        A_diagonal[k] = Ap

    #top wall on CSR Format
    i = 1
    for j in range(2, n_y):
        k = Lij(i, j)

        F_e = dy * u_face[i, j]
        F_w = dy * u_face[i, j-1]
        F_n = dx * v_face[i-1, j]
        F_s = dx * v_face[i, j]

        Ae =   D_e + max(0.0,-F_e)
        Aw =   D_w + max(0.0, F_w)
        An = 2*D_n + max(0.0,-F_n)
        As =   D_s + max(0.0, F_s)
        Ap = Aw + Ae + An + As + (F_e - F_w) + (F_n - F_s)

        Su[k] = 0.5 * (p[i, j-1] - p[i, j+1]) * dx
        Sv[k] = 0.5 * (p[i+1, j] - p[i, j]) * dy + D_n * v_star[i-1, j]

        # Under-Relaxation, Solving Equation System
        Ap = Ap / alpha
        Su[k] = Su[k] + (1 - alpha) * Ap * u_star[i, j]
        Sv[k] = Sv[k] + (1 - alpha) * Ap * v_star[i, j]

        A[ptr[k]:ptr[k+1]] = [Aw, Ap, Ae, As]
        A_diagonal[k] = Ap

    #top left corner on CSR Format
    i = 1
    j = 1
    k = Lij(i, j)

    F_e = dy * u_face[i, j]
    F_w = dy * u_face[i, j-1]
    F_n = dx * v_face[i-1, j]
    F_s = dx * v_face[i, j]

    Ae =   D_e + max(0.0,-F_e)
    Aw = 2*D_w + max(0.0, F_w)
    An = 2*D_n + max(0.0,-F_n)
    As =   D_s + max(0.0, F_s)
    Ap = Aw + Ae + An + As + (F_e - F_w) + (F_n - F_s)

    Su[k] = 0.5 * (p[i, j] - p[i, j+1]) * dx + D_w * u_star[i, j-1]
    Sv[k] = 0.5 * (p[i+1, j] - p[i, j]) * dy + D_n * v_star[i-1, j]

    # Under-Relaxation, Solving Equation System
    Ap = Ap / alpha
    Su[k] = Su[k] + (1 - alpha) * Ap * u_star[i, j]
    Sv[k] = Sv[k] + (1 - alpha) * Ap * v_star[i, j]

    A[ptr[k]:ptr[k+1]] = [Ap, Ae, As]
    A_diagonal[k] = Ap

    #top right corner on CSR Format
    i = 1
    j = n_x
    k = Lij(i, j)

    F_e = dy * u_face[i, j]
    F_w = dy * u_face[i, j-1]
    F_n = dx * v_face[i-1, j]
    F_s = dx * v_face[i, j]

    Ae = 2*D_e + max(0.0,-F_e)
    Aw =   D_w + max(0.0, F_w)
    An = 2*D_n + max(0.0,-F_n)
    As =   D_s + max(0.0, F_s)
    Ap = Aw + Ae + An + As + (F_e - F_w) + (F_n - F_s)

    Su[k] = 0.5 * (p[i, j-1] - p[i, j]) * dx + D_e * u_star[i, j+1]
    Sv[k] = 0.5 * (p[i+1, j] - p[i, j]) * dy + D_n * v_star[i-1, j]

    # Under-Relaxation, Solving Equation System
    Ap = Ap / alpha
    Su[k] = Su[k] + (1 - alpha) * Ap * u_star[i, j]
    Sv[k] = Sv[k] + (1 - alpha) * Ap * v_star[i, j]

    A[ptr[k]:ptr[k+1]] = [Aw, Ap, As]
    A_diagonal[k] = Ap

    #bottom left corner on CSR Format
    i = n_y
    j = 1
    k = Lij(i, j)

    F_e = dy * u_face[i, j]
    F_w = dy * u_face[i, j-1]
    F_n = dx * v_face[i-1, j]
    F_s = dx * v_face[i, j]

    Ae =   D_e + max(0.0,-F_e)
    Aw = 2*D_w + max(0.0, F_w)
    An =   D_n + max(0.0,-F_n)
    As = 2*D_s + max(0.0, F_s)
    Ap = Aw + Ae + An + As + (F_e - F_w) + (F_n - F_s)

    Su[k] = 0.5 * (p[i, j] - p[i, j+1]) * dx + D_w * u_star[i, j-1]
    Sv[k] = 0.5 * (p[i, j] - p[i-1, j]) * dy + D_s * v_star[i+1, j]

    # Under-Relaxation, Solving Equation System
    Ap = Ap / alpha
    Su[k] = Su[k] + (1 - alpha) * Ap * u_star[i, j]
    Sv[k] = Sv[k] + (1 - alpha) * Ap * v_star[i, j]

    A[ptr[k]:ptr[k+1]] = [An, Ap, Ae]
    A_diagonal[k] = Ap

    #bottom right corner on CSR Format
    i = n_y
    j = n_x
    k = Lij(i, j)

    F_e = dy * u_face[i, j]
    F_w = dy * u_face[i, j-1]
    F_n = dx * v_face[i-1, j]
    F_s = dx * v_face[i, j]

    Ae = 2*D_e + max(0.0,-F_e)
    Aw =   D_w + max(0.0, F_w)
    An =   D_n + max(0.0,-F_n)
    As = 2*D_s + max(0.0, F_s)
    Ap = Aw + Ae + An + As + (F_e - F_w) + (F_n - F_s)

    Su[k] = 0.5*(p[i,j - 1] - p[i,j])*dx + D_e * u_star[i, j+1]
    Sv[k] = 0.5*(p[i,j] - p[i - 1,j])*dy + D_s * v_star[i+1, j]

    # Under-Relaxation, Solving Equation System
    Ap = Ap / alpha
    Su[k] = Su[k] + (1 - alpha) * Ap * u_star[i, j]
    Sv[k] = Sv[k] + (1 - alpha) * Ap * v_star[i, j]

    A[ptr[k]:ptr[k+1]] = [An, Aw, Ap]
    A_diagonal[k] = Ap

    return A, Su, Sv


def solve(val, col, ptr, b):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import bicgstab

    # bicgstab(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
    # rtol, atol : float, optional
    #   Parameters for the convergence test.
    #   For convergence, norm(b - A @ x) <= max(rtol*norm(b), atol) should be satisfied.
    #   The default is atol=0. and rtol=1e-5.
    # Retrun info : integer
    #   Provides convergence information:
    #   0 : successful exit 
    #  >0 : convergence to tolerance not achieved, number of iterations
    #  <0 : parameter breakdown

    A = csr_matrix((val, col, ptr), shape=(nxy, nxy))
    x, exit_code = bicgstab(A, b)

    print(exit_code)

    return x

# def solve(u,u_star,A_p,A_e,A_w,A_n,A_s,source_x,alpha,epsilon,max_inner_iteration,l2_norm):
#     for n in range(1,max_inner_iteration+1):

#         l2_norm=0
#         for i in range(1,n_y+1):
#             for j in range(1,n_x+1):
#                 u[i,j]= alpha*(A_e[i,j]*u[i,j+1] + A_w[i,j]*u[i,j-1] + A_n[i,j]*u[i-1,j] + A_s[i,j]*u[i+1,j] + source_x[i,j])/A_p[i,j] + (1-alpha)*u_star[i,j]
#                 l2_norm+=(u[i,j] - alpha*(A_e[i,j]*u[i,j+1] + A_w[i,j]*u[i,j-1] + A_n[i,j]*u[i - 1,j] + A_s[i,j]*u[i+1,j] +source_x[i,j])/A_p[i,j] -  (1-alpha)*u_star[i,j])**2

#         for i in range(1,n_y+1):
#             for j in range(1,n_x+1):
#                 l2_norm+=(u[i,j] - alpha*(A_e[i,j]*u[i,j+1] + A_w[i,j]*u[i,j-1] + A_n[i,j]*u[i - 1,j] + A_s[i,j]*u[i+1,j] +source_x[i,j])/A_p[i,j] -  (1-alpha)*u_star[i,j])**2


#         if(n==1):
#             norm=math.sqrt(l2_norm)


#         l2_norm=math.sqrt(l2_norm)
#         if(l2_norm<epsilon):
#             #print("Converged in ",n, " iterations")
#             break

#     return u,norm


def face_velocity(u, v, u_face, v_face, p, Ap, alpha):

    # uface velocity
    for i in range(1, n_y+1):
        for j in range(1, n_x):
            k = Lij(i, j)

            u_face[i, j] = 0.5 * (u[i, j] + u[i, j+1])
                         + 0.25 * alpha * (p[i, j+1] - p[i, j-1]) * dy / Ap[k]
                         + 0.25 * alpha * (p[i, j+2] - p[i,j]) * dy / Ap[k+1]
                         - 0.5 * alpha * (1/Ap[k] + 1/Ap[k+1]) * (p[i, j+1] - p[i, j]) * dy

    # vface velocity
    for i in range(2, n_y+1):
        for j in range(1, n_x+1):
            k = Lij(i, j)

            v_face[i-1, j] = 0.5 * (v[i, j] + v[i-1, j])
                           + 0.25 * alpha * (p[i-1, j] - p[i+1, j]) * dy / Ap[k]
                           + 0.25 * alpha * (p[i-2, j] - p[i,j]) * dy / Ap[k - n_x]
                           - 0.5 * alpha * (1/Ap[k] + 1/Ap[k - n_x]) * (p[i-1, j] - p[i, j]) * dy

    return u_face,v_face


def pressure_correction_link_coefficients(u,u_face,v_face,Ap_p,Ap_e,Ap_w,Ap_n,Ap_s,source_p,A_p,A_e,A_w,A_n,A_s,alpha_uv):

    #interior cells
    for i in range(2,n_y):
        for j in range(2,n_x):

            Ap_e[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j + 1])*(dy**2)
            Ap_w[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j - 1])*(dy**2)
            Ap_n[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i - 1,j])*(dx**2)
            Ap_s[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i + 1,j])*(dx**2)
            Ap_p[i,j]=Ap_e[i,j] + Ap_w[i,j] + Ap_n[i,j] + Ap_s[i,j]

            source_p[i,j]=-(u_face[i,j] - u_face[i,j - 1])*dy - (v_face[i - 1,j] - v_face[i,j])*dx


    #top
    i=1
    for j in range(2,n_x):
        
        Ap_e[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j + 1])*(dy**2)
        Ap_w[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j - 1])*(dy**2)
        Ap_n[i,j]=0
        Ap_s[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i + 1,j])*(dx**2)
        Ap_p[i,j]=Ap_e[i,j] + Ap_w[i,j] + Ap_n[i,j] + Ap_s[i,j]

        source_p[i,j]=-(u_face[i,j] - u_face[i,j - 1])*dy - (v_face[i - 1,j] - v_face[i,j])*dx

    #left
    j=1
    for i in range(2,n_y):
        Ap_e[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j + 1])*(dy**2)
        Ap_w[i,j]=0
        Ap_n[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i - 1,j])*(dx**2)
 vb         Ap_s[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i + 1,j])*(dx**2)
        Ap_p[i,j]=Ap_e[i,j] + Ap_w[i,j] + Ap_n[i,j] + Ap_s[i,j]

        source_p[i,j]=-(u_face[i,j] - u_face[i,j - 1])*dy - (v_face[i - 1,j] - v_face[i,j])*dx


    #right
    j=n_x
    for i in range(2,n_y):
        Ap_e[i,j]=0
        Ap_w[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j - 1])*(dy**2)
        Ap_n[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i - 1,j])*(dx**2)
        Ap_s[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i + 1,j])*(dx**2)
        Ap_p[i,j]=Ap_e[i,j] + Ap_w[i,j] + Ap_n[i,j] + Ap_s[i,j]

        source_p[i,j]=-(u_face[i,j] - u_face[i,j - 1])*dy - (v_face[i - 1,j] - v_face[i,j])*dx

    #bottom
    i=n_y
    for j in range(2,n_x):
        Ap_e[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j + 1])*(dy**2)
        Ap_w[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j - 1])*(dy**2)
        Ap_n[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i - 1,j])*(dx**2)
        Ap_s[i,j]=0
        Ap_p[i,j]=Ap_e[i,j] + Ap_w[i,j] + Ap_n[i,j] + Ap_s[i,j]

        source_p[i,j]=-(u_face[i,j] - u_face[i,j - 1])*dy - (v_face[i - 1,j] - v_face[i,j])*dx

    #top left corner
    i=1
    j=1

    Ap_e[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j + 1])*(dy**2)
    Ap_w[i,j]=0
    Ap_n[i,j]=0
    Ap_s[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i + 1,j])*(dx**2)
    Ap_p[i,j]=Ap_e[i,j] + Ap_w[i,j] + Ap_n[i,j] + Ap_s[i,j]

    source_p[i,j]=-(u_face[i,j] - u_face[i,j - 1])*dy - (v_face[i - 1,j] - v_face[i,j])*dx
    
    #top right corner
    i=1
    j=n_x
    Ap_e[i,j]=0
    Ap_w[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j - 1])*(dy**2)
    Ap_n[i,j]=0
    Ap_s[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i + 1,j])*(dx**2)
    Ap_p[i,j]=Ap_e[i,j] + Ap_w[i,j] + Ap_n[i,j] + Ap_s[i,j]

    source_p[i,j]=-(u_face[i,j] - u_face[i,j - 1])*dy - (v_face[i - 1,j] - v_face[i,j])*dx
    
    #bottom left corner
    i=n_y
    j=1
    Ap_e[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j + 1])*(dy**2)
    Ap_w[i,j]=0
    Ap_n[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i - 1,j])*(dx**2)
    Ap_s[i,j]=0
    Ap_p[i,j]=Ap_e[i,j] + Ap_w[i,j] + Ap_n[i,j] + Ap_s[i,j]

    source_p[i,j]=-(u_face[i,j] - u_face[i,j - 1])*dy - (v_face[i - 1,j] - v_face[i,j])*dx
    
    #bottom right corner
    i=n_y
    j=n_x
    Ap_e[i,j]=0
    Ap_w[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i,j - 1])*(dy**2)
    Ap_n[i,j]=0.5*alpha_uv*(1/A_p[i,j] + 1/A_p[i - 1,j])*(dx**2)
    Ap_s[i,j]=0
    Ap_p[i,j]=Ap_e[i,j] + Ap_w[i,j] + Ap_n[i,j] + Ap_s[i,j]

    source_p[i,j]=-(u_face[i,j] - u_face[i,j - 1])*dy - (v_face[i - 1,j] - v_face[i,j])*dx

    return Ap_p,Ap_e,Ap_w,Ap_n,Ap_s,source_p

def correct_pressure(p_star,p,p_prime,alpha_p):

    p_star=p+alpha_p*p_prime

    #BC

    #top wall
    p_star[0,1:n_x+1]=p_star[1,1:n_x+1]
    #left wall
    p_star[1:n_y+1,0]=p_star[1:n_y+1,1]
    #right wall
    p_star[1:n_y+1,n_x+1]=p_star[1:n_y+1,n_x]
    #bottom wall
    p_star[n_y+1,1:n_x+1]=p_star[n_y,1:n_x+1]

    #top left corner
    p_star[0,0]=(p_star[1,2]+p_star[0,1]+p_star[1,0])/3

    #top right corner
    p_star[0,n_x+1]=(p_star[0,n_x]+p_star[1,n_x]+p_star[1,n_x+1])/3

    #bottom left corner
    p_star[n_y+1,0]=(p_star[n_y,0]+p_star[n_y,1]+p_star[n_y+1,1])/3

    #bottom right corner
    p_star[n_y+1,n_x+1]=(p_star[n_y,n_x+1]+p_star[n_y+1,n_x]+p_star[n_y,n_x])/3



    return p_star

def correct_cell_center_velocity(u,v,u_star,v_star,p_prime,A_p,alpha_uv):

    #u velocity
    #interior cells
    for i in range(1,n_y+1):
        for j in range(2,n_x ):
            u_star[i,j]= u[i,j] + 0.5*alpha_uv*(p_prime[i,j-1]-p_prime[i,j+1])*dy/A_p[i,j]

    #left
    j=1
    for i in range(1,n_y+1):
        u_star[i,j]=u[i,j] + 0.5*alpha_uv*(p_prime[i,j] - p_prime[i,j+1])*dy/A_p[i,j]

    #right
    j=n_x
    for i in range(1,n_y+1):
        u_star[i,j]=u[i,j] + 0.5*alpha_uv*(p_prime[i,j-1] - p_prime[i,j])*dy/A_p[i,j]


    #v velocity
    for i in range(2,n_y):
        for j in range(1,n_x+1):
            v_star[i,j]=v[i,j] + 0.5*alpha_uv*(p_prime[i+1,j]-p_prime[i-1,j])*dx/A_p[i,j]

    #top
    i=1
    for j in range(1,n_x + 1):
        v_star[i,j]=v[i,j] + 0.5*alpha_uv*(p_prime[i + 1,j] - p_prime[i,j])*dx/A_p[i,j]

    #bottom
    i=n_y
    for j in range(1,n_x + 1):
        v_star[i,j]=v[i,j] + 0.5*alpha_uv*(p_prime[i,j] - p_prime[i - 1,j])*dx/A_p[i,j]

    return  u_star,v_star

def correct_face_velocity(u_face,v_face,p_prime,A_p,alpha_uv):


    for i in range(1,n_y+1):
        for j in range(1,n_x):
            u_face[i,j]=u_face[i,j]+ 0.5*alpha_uv*(1/A_p[i,j]+1/A_p[i,j+1])*(p_prime[i,j]-p_prime[i,j+1])*dy

    for i in range(2,n_y+1):
        for j in range(1,n_x+1):
            v_face[i-1,j]=v_face[i-1,j] +  0.5*alpha_uv*(1/A_p[i,j]+1/A_p[i-1,j])*(p_prime[i,j]-p_prime[i-1,j])*dx

    return u_face,v_face

def post_processing(u_star,v_star,p_star,X,Y,x,y):

    #u velocity contours
    plt.figure(1)
    plt.contourf(X,Y,np.flipud(u_star),levels=50,cmap='jet')
    plt.colorbar()
    plt.title('U contours')
    plt.show()

    #v velocity contours
    plt.figure(2)
    plt.contourf(X,Y,np.flipud(v_star),levels=50,cmap='jet')
    plt.colorbar()
    plt.title('V contours' )
    plt.show()

    #pressure contours
    plt.figure(3)
    plt.contourf(X,Y,np.flipud(p_star),levels=50,cmap='jet')
    plt.colorbar()
    plt.title('P contours')
    plt.show()

    #u centerline velocity
    plt.figure(4)
    plt.plot(1-y,u_star[:,round(n_x/2)])
    plt.xlabel('y')
    plt.ylabel('u')
    plt.title('U centerline velocity')
    plt.show()

    #v centerline velocity
    plt.figure(5)
    plt.plot(x,v_star[round(n_y/2),:])
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title('V centerline velocity')
    plt.show()



#Declaring primitive variables
u=np.zeros((n_y+2,n_x+2),dtype=np.float64)
u_star=np.zeros((n_y+2,n_x+2),dtype=np.float64)

v=np.zeros((n_y+2,n_x+2),dtype=np.float64)
v_star=np.zeros((n_y+2,n_x+2),dtype=np.float64)

p_star=np.zeros((n_y+2,n_x+2),dtype=np.float64)
p=np.zeros((n_y+2,n_x+2),dtype=np.float64)
p_prime=np.zeros((n_y+2,n_x+2),dtype=np.float64)

#Momentum link coeffficients
A_p=np.ones((n_y+2,n_x+2),dtype=np.float64)
A_e=np.ones((n_y+2,n_x+2),dtype=np.float64)
A_w=np.ones((n_y+2,n_x+2),dtype=np.float64)
A_s=np.ones((n_y+2,n_x+2),dtype=np.float64)
A_n=np.ones((n_y+2,n_x+2),dtype=np.float64)

#Pressure correction link coeffficients
Ap_p=np.ones((n_y+2,n_x+2),dtype=np.float64)
Ap_e=np.ones((n_y+2,n_x+2),dtype=np.float64)
Ap_w=np.ones((n_y+2,n_x+2),dtype=np.float64)
Ap_s=np.ones((n_y+2,n_x+2),dtype=np.float64)
Ap_n=np.ones((n_y+2,n_x+2),dtype=np.float64)

#Declaring source terms
source_x=np.zeros((n_y+2,n_x+2),dtype=np.float64)
source_y=np.zeros((n_y+2,n_x+2),dtype=np.float64)
source_p=np.zeros((n_y+2,n_x+2),dtype=np.float64)

#Declaring face velocities
u_face=np.zeros((n_y+2,n_x+1),dtype=np.float64)
v_face=np.zeros((n_y+1,n_x+2),dtype=np.float64)


#Momentum link coeffficients (DENSE)
ncells = n_y * n_x
A = np.zeros((ncells, ncells), dtype=np.int64)
uk = np.zeros((ncells), dtype=np.float64)
vk = np.zeros((ncells), dtype=np.float64)
Su = np.zeros((ncells), dtype=np.float64)
Sv = np.zeros((ncells), dtype=np.float64)

Ap = np.zeros((ncells, ncells), dtype=np.float64)
pk = np.zeros((ncells), dtype=np.float64)
Sp = np.zeros((ncells), dtype=np.float64)



#Grid
x=np.array([0.0],dtype=np.float64)
y=np.array([0.0],dtype=np.float64)

x=np.append(x,np.linspace(dx/2,1-dx/2,n_x))
x=np.append(x,[1.0])

y=np.append(y,np.linspace(dy/2,1-dy/2,n_y))
y=np.append(y,[1.0])

X,Y=np.meshgrid(x,y)


#BC
u[0,1:n_x+1]=1
u_star[0,1:n_x+1]=1
u_face[0,1:n_x]=1


l2_norm_x=0
alpha_uv=0.7
epsilon_uv=1e-3
max_inner_iteration_uv=50

l2_norm_y=0

l2_norm_p=0
max_inner_iteration_p=200
dummy_alpha_p=1
epsilon_p=1e-4
alpha_p=0.2

max_outer_iteration=200

exit()

for n in range(1,max_outer_iteration+1):

    A_p,A_e,A_w,A_n,A_s,source_x,source_y=momentum_link_coefficients(u_star,u_face,v_face,p,source_x,source_y,A_p,A_e,A_w,A_n,A_s)

    u,l2_norm_x=solve(u,u_star,A_p,A_e,A_w,A_n,A_s,source_x,alpha_uv,epsilon_uv,max_inner_iteration_uv,l2_norm_x)

    v,l2_norm_y=solve(v,v_star,A_p,A_e,A_w,A_n,A_s,source_y,alpha_uv,epsilon_uv,max_inner_iteration_uv,l2_norm_y)

    u_face,v_face=face_velocity(u,v,u_face,v_face,p,A_p,alpha_uv)

    Ap_p,Ap_e,Ap_w,Ap_n,Ap_s,source_p=pressure_correction_link_coefficients(u,u_face,v_face,Ap_p,Ap_e,Ap_w,Ap_n,Ap_s,source_p,A_p,A_e,A_w,A_n,A_s,alpha_uv)

    p_prime,l2_norm_p=solve(p_prime,p_prime,Ap_p,Ap_e,Ap_w,Ap_n,Ap_s,source_p,dummy_alpha_p,epsilon_p,max_inner_iteration_p,l2_norm_p)

    p_star=correct_pressure(p_star,p,p_prime,alpha_p)
  
    u_star,v_star=correct_cell_center_velocity(u,v,u_star,v_star,p_prime,A_p,alpha_uv)

    u_face,v_face=correct_face_velocity(u_face,v_face,p_prime,A_p,alpha_uv)

    p=np.copy(p_star)

    print(l2_norm_x,l2_norm_y,l2_norm_p)

    if (l2_norm_x < 1e-4 and l2_norm_y < 1e-4 and l2_norm_p<1e-4):
        print("Converged !")
        break

# post_processing(u_star,v_star,p_star,X,Y,x,y)
