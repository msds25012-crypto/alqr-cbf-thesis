import torch as th
import matplotlib.pyplot as plt
import os

def linearize(tfs, T, m, X_nom):
    """
    Linearize nonlinear dynamics f around nominal trajectory (X_nom, U_nom=0).

    Args:
        f: dynamics function f(x,u) -> x_next
        X_nom: nominal states (T+1, k, n)
        U_nom: nominal controls (T, m)

    Returns:
        A: linearized A matrices (T, n, n)
        B: linearized B matrices (T, n, m)
    """
    U_nom = th.zeros([T, m], device=X_nom.device)
    n = X_nom.shape[-1]
    # print(X_nom.shape)

    A = th.zeros((T, n, n), dtype=X_nom.dtype, device=X_nom.device)
    # B = th.zeros((T, n, m), dtype=X_nom.dtype, device=X_nom.device)

    for t in range(T):
        x = X_nom[t].detach().requires_grad_(True)
        u = U_nom[t].detach().requires_grad_(True)

        def f_last(x, u):
            return tfs[t](x,u)[..., -1, :]

        # Compute Jacobians:
        Jx = th.autograd.functional.jacobian(lambda x_: f_last(x_, u), x, create_graph=False, vectorize=True)   # shape: [n, *x.shape]
        # Ju = th.autograd.functional.jacobian(lambda u_: f_last(x, u_), u, create_graph=False, vectorize=True)   # shape: [n, *u.shape]
        A[t] = Jx[...,-1,:]
        # B[t] = Ju

    return A

import torch as th

def linearize_vram_efficient(tfs, T, m, X_nom):
    """
    Memory-efficient linearization using row-by-row gradient computation.

    Args:
        tfs: list of dynamics functions
        T: time horizon
        m: control dim
        X_nom: nominal states (T+1, k, n)

    Returns:
        A: (T, n, n)
    """
    device = X_nom.device
    dtype = X_nom.dtype
    n = X_nom.shape[-1]

    U_nom = th.zeros((T, m), device=device, dtype=dtype)
    A = th.zeros((T, n, n), device=device, dtype=dtype)

    for t in range(T):
        x = X_nom[t].detach().requires_grad_(True)
        u = U_nom[t].detach().requires_grad_(True)

        def f_last(x, u):
            return tfs[t](x, u)[..., -1, :]  # shape (..., n)

        y = f_last(x, u)  # forward pass once

        # Compute Jacobian row-by-row
        for i in range(n):
            grad_outputs = th.zeros_like(y)
            grad_outputs[..., i] = 1.0

            grad_x = th.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=grad_outputs,
                retain_graph=True,   # reuse graph for next row
                create_graph=False
            )[0]

            A[t, i] = grad_x[..., -1, :]  # match your original indexing

        # Free graph explicitly (important for VRAM)
        del y, x, u

    return A

def mean_linearize(tfs, T, m, X_nom):
    """
    Linearize nonlinear dynamics f around nominal trajectory (X_nom, U_nom=0).

    Args:
        f: dynamics function f(x,u) -> x_next
        X_nom: nominal states (T+1, B, k, n)
        U_nom: nominal controls (T, m)

    Returns:
        A: linearized A matrices (T, n, n)
        B: linearized B matrices (T, n, m)
    """
    U_nom = th.zeros([T, m], device=X_nom.device)
    n = X_nom.shape[-1]
    print(X_nom.shape)

    A = th.zeros((T, n, n), dtype=X_nom.dtype, device=X_nom.device)

    for t in range(T):
        x = X_nom[t].detach().requires_grad_(True)
        u = U_nom[t].detach().requires_grad_(True)

        def f_last(x, u):
            return tfs[t](x,u)[..., -1, :]

        # Compute Jacobians:
        Jx = th.autograd.functional.jacobian(lambda x_: f_last(x_, u), x, create_graph=False, vectorize=True)   # shape: [n, *x.shape]
        # print(Jx.shape)
        A[t] = th.mean(Jx[...,-1,:], dim=1)

    return A

def time_varying_lqr_noB(A, Q, R, S_T):
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
    T, n, m = A.shape

    S = th.zeros((T+1, n, n), dtype=A.dtype, device=A.device)
    K = th.zeros((T, m, n), dtype=A.dtype, device=A.device)


    S[T] = S_T

    for t in reversed(range(T)):
        At = A[t]
        Qt = Q[t]
        Rt = R[t]

        # Sk = Ak^T [Sk+1 − Sk+1 Bk (BkT Sk+1 Bk + Rk)^-1 BkT Sk+1]Ak + Qk
        P = (S[t+1] + Rt).to(A.device) # = BkT Sk+1 Bk + Rk
        F = (S[t+1] @ At).to(A.device) # = BkT Sk+1 Ak  
        G = (Qt + At.transpose(-2, -1) @ S[t+1] @ At).to(A.device) # = Ak^T Sk+1 Ak + Qk

        P_inv = th.linalg.inv(P)
        K[t] = P_inv @ F

        S[t] = G - F.transpose(-2, -1) @ P_inv @ F

    return K

def time_varying_lqr_noB_mem_efficient(A, Q, R, S_T):
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
    T, n, m = A.shape

    S = th.zeros((T+1, n, n), dtype=A.dtype)
    K = th.zeros((T, m, n), dtype=A.dtype, device=A.device)


    S[T] = S_T

    for t in reversed(range(T)):
        At = A[t]
        Qt = Q[t]
        Rt = R[t]
        Stp1 = S[t+1].to(A.device)

        # Sk = Ak^T [Sk+1 − Sk+1 Bk (BkT Sk+1 Bk + Rk)^-1 BkT Sk+1]Ak + Qk
        P = (Stp1 + Rt).to(A.device) # = BkT Sk+1 Bk + Rk
        F = (Stp1 @ At).to(A.device) # = BkT Sk+1 Ak  
        G = (Qt + At.transpose(-2, -1) @ Stp1 @ At).to(A.device) # = Ak^T Sk+1 Ak + Qk

        P_inv = th.linalg.inv(P)
        K[t] = P_inv @ F

        S[t] = (G - F.transpose(-2, -1) @ P_inv @ F).to('cpu')

    return K


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

def transformerBlockControl(tf, x, u):
    # print(f"if ousdfdsdfd: {x.shape}")
    x_next = tf(x)
    # x_next[:,-1,:] = x_next[:,-1,:] + u # 4.40.2
    x_next[...,-1,:] = x_next[...,-1,:] + u
    return x_next


def find_random_target(model, x0):
    x = x0
    for block in model.blocks:
        x = block(x)
    return x

def llama_block_wrapper(block, attention_mask, position_ids, x):
    x = x.unsqueeze(0)
    # x = x
    return block(x, attention_mask, position_ids)[0]

def new_llama_block_wrapper(block, attention_mask, position_ids, position_embeddings, x): # 4.57
    x = x.unsqueeze(0)
    return block(x, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]