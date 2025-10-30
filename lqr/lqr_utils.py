import os
import sys

root_path = os.path.abspath('..')

if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
import torch as th
# from lpe.lpe.utils import Transformer
import matplotlib.pyplot as plt


# model_name = "gelu-2l"
# device = th.device("cuda" if th.cuda.is_available() else "cpu")
# model = Transformer.from_pretrained(model_name).to(device)

def linearize(tfs, T, m, X_nom):
    """
    Linearize nonlinear dynamics f around nominal trajectory (X_nom, U_nom=0).

    Args:
        f: dynamics function f(x,u) -> x_next
        X_nom: nominal states (T+1, n)
        U_nom: nominal controls (T, m)

    Returns:
        A: linearized A matrices (T, n, n)
        B: linearized B matrices (T, n, m)
    """
    U_nom = th.zeros([T, m], device=X_nom.device)
    n = X_nom.shape[1]

    A = th.zeros((T, n, n), dtype=X_nom.dtype, device=X_nom.device)
    B = th.zeros((T, n, m), dtype=X_nom.dtype, device=X_nom.device)

    for t in range(T):
        x = X_nom[t].detach().requires_grad_(True)
        u = U_nom[t].detach().requires_grad_(True)
        f_eval = tfs[t](x,u)
        # f_eval = tfs(x,u)
        # print(f"feval shape: {f_eval.shape}")
        # x = x[:,-1,:]

        for i in range(n):
            grad_x = th.autograd.grad(f_eval[...,i], x, retain_graph=True, create_graph=False)[0]
            grad_u = th.autograd.grad(f_eval[...,i], u, retain_graph=True, create_graph=False)[0]
            # print(f"grad_x.shape: {grad_x.shape}")
            A[t, i] = grad_x
            B[t, i] = grad_u

    return A, B


def time_varying_lqr(A, B, Q, R, S_T):
    """
    Solve the time-varying LQR problem given linearized dynamics.

    Args:
        A: (T, n, n)
        B: (T, n, m)
        Q: (T, n, n)
        R: (T, m, m)
        Qf: (n, n)

    Returns:
        K: (T, m, n) feedback gains
    """
    T, n, m = B.shape

    S = th.zeros((T+1, n, n), dtype=A.dtype, device=A.device)
    K = th.zeros((T, m, n), dtype=A.dtype, device=A.device)


    S[T] = S_T

    for t in reversed(range(T)):
        At = A[t]
        Bt = B[t]
        Qt = Q[t]
        Rt = R[t]

        # Sk = Ak^T [Sk+1 − Sk+1 Bk (BkT Sk+1 Bk + Rk)^-1 BkT Sk+1]Ak + Qk
        P = (Bt.transpose(-2, -1) @ S[t+1] @ Bt + Rt).to(A.device) # = BkT Sk+1 Bk + Rk
        F = (Bt.transpose(-2, -1) @ S[t+1] @ At).to(A.device) # = BkT Sk+1 Ak  
        G = (Qt + At.transpose(-2, -1) @ S[t+1] @ At).to(A.device) # = Ak^T Sk+1 Ak + Qk

        P_inv = th.linalg.inv(P)
        K[t] = P_inv @ F

        S[t] = G - F.transpose(-2, -1) @ P_inv @ F

    return K

def time_varying_lqr_w_target(A, B, Q, R, S_T, x_tar):
    """
    Solve time-varying LQR with target final state x_tar.

    J = (x_T - x_tar)^T S_T (x_T - x_tar) + sum_t (x_t^T Q_t x_t + u_t^T R_t u_t)
    """
    T, n, m = B.shape

    S = th.zeros((T + 1, n, n), dtype=A.dtype, device=A.device)
    s = th.zeros((T + 1, n), dtype=A.dtype, device=A.device)
    K = th.zeros((T, m, n), dtype=A.dtype, device=A.device)
    kff = th.zeros((T, m), dtype=A.dtype, device=A.device)

    # Terminal conditions
    S[T] = S_T
    s[T] = -S_T @ x_tar

    # Backward recursion
    for t in reversed(range(T)):
        At, Bt, Qt, Rt = A[t], B[t], Q[t], R[t]
        P = Bt.T @ S[t+1] @ Bt + Rt
        F = Bt.T @ S[t+1] @ At
        G = Qt + At.T @ S[t+1] @ At
        P_inv = th.linalg.inv(P)

        K[t] = P_inv @ F
        kff[t] = P_inv @ (Bt.T @ s[t+1])

        S[t] = G - F.T @ P_inv @ F
        s[t] = At.T @ (s[t+1] - S[t+1] @ Bt @ kff[t])

    return K, kff

def transformerBlockControl(tf, x, u):
    return tf(x)[:,-1,:] + u
    # print(f"x.shape: {x.shape}")
    # return tf(x.unsqueeze(0).unsqueeze(0)) + u


def find_random_target(model, x0):
    x = x0
    for block in model.blocks:
        x = block(x)
    return x

def llama_block_wrapper(block, attention_mask, position_ids, x):
    x = x.unsqueeze(0).unsqueeze(0)
    return block(x, attention_mask, position_ids)[0]

def normed_error(x,x_lin):
    return th.norm(x - x_lin)/th.norm(x)


def dimnormed_error(x,x_lin, d):
    return th.norm(x - x_lin)/d

def unnormed_error(x,x_lin):
    return th.norm(x - x_lin)

def print_matrix(M):
    file_name = "matrix.txt"

    with open(file_name, 'w') as file:
        '''
        assume shape (T, n, n)
        '''
        for t in range(M.shape[0]):
            file.write("[")
            for j in range(M.shape[1]):
                row = ""
                for i in range(M.shape[1]):
                    row = row + str(M[t,i,j].item()) + " "
                row = row + "\n"
                file.write(row)
            file.write("]\n\n")

# Nonlinear dynamics example
# def pendulum_dynamics(x, u, dt=0.05):
#     """
#     Discrete-time nonlinear pendulum dynamics.

#     Args:
#         x: (2,) state tensor: [theta, omega]
#         u: (1,) control tensor: torque
#         dt: timestep size

#     Returns:
#         x_next: (2,) next state
#     """
#     theta, omega = x[0], x[1]
#     torque = u[0]

#     theta_next = theta + dt * omega
#     omega_next = omega + dt * (-th.sin(theta) - 0.1 * omega + torque)

#     return th.stack([theta_next, omega_next])



# n = 2
# m = 1
# T = 1000

# # Nominal trajectory
# X_nom = th.zeros((T+1, n), device=device)
# X_nom[:,0] = 0.5
# U_nom = th.zeros((T, m), device=device)
# U_nom[:,0] = 0.48
 
# # Linearize dynamics around nominal
# A, B = linearize(pendulum_dynamics,T,m,X_nom)

# # Define quadratic cost matrices
# Q = th.eye(n).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
# R = th.eye(m).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 0
# Qf = 10000 * th.eye(n).to(A.device)

# # Solve LQR on linearized system
# K = time_varying_lqr(A, B, Q, R, Qf)

# print("Feedback gains K shape:", K.shape)
# print("K[0]:", K[0])

# X = th.zeros_like(X_nom)
# U = th.zeros_like(U_nom)
# X[0] = th.tensor([0, 0])
# for i in range(T):
#     U[i] = U_nom[i]-K[i]@(X[i]-X_nom[i])
#     X[i+1] = pendulum_dynamics(X[i], U[i])


# plt.plot(range(T+1),X[:,0].cpu())
# plt.plot(range(T+1),X_nom[:,0].cpu())
# plt.savefig("lqr_test.png")
