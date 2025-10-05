import mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from timeit import default_timer as timer
import numpy as np
import gpt_config as config
from functools import partial
import jax.numpy as jnp
import jax 

import torch as th

def runOffline(x0, target):
    """
    Runs one MPC update using the current state, input, and foot positions.

    Args:
        x0: Current system state vector.

    Returns:
        A tuple (X, U, V) representing the computed state trajectory, control sequence,
        and auxiliary variable trajectory.
    """
    #compensate for the time delay
    #get forward kinematics for foot position
    # print("starting")
    cost = partial(config.cost, config.N)
    work = partial(optimizers.mpc, cost, config.dynamics, config.hessian_approx, False) # set limited memory to false
    _solve = jax.jit(work)
    

    # self.X0 = self.X0.at[:,:13+self.config.n_joints].set(reference[:,:13+self.config.n_joints])

    reference = jnp.zeros([config.N+1, config.n])
    reference.at[0].set(x0)
    # print(f"shapey shape: {reference[0].shape}")
    for i,block in enumerate(config.tfs_jax):
        ri_1 = reference[i].squeeze()
        ri_reshape = ri_1[None, None,...]
        # print(f"rishap = {ri_reshape.shape}")
        sqeee = block(ri_reshape).squeeze()
        # print(f"squeee: {sqeee.shape}")
        reference.at[i+1].set(sqeee)
    
    _cost = partial(config.cost,config.N, config.W, reference)
    _dynamics = config.dynamics
    model_evaluator = partial(optimizers.model_evaluator_helper, _cost, _dynamics,x0)
    jitted_model_evaluator = jax.jit(model_evaluator)

    _exit = False
    max_iter = 2 # was 100
    # max_iter = 0 # was 100
    last_cost = 1e10
    i = 0
    # output = []
    # output.append((self.X0))

    print(f"x0 shape: {x0.shape}")

    


    Xf = jnp.zeros([config.N+1, config.n])
    Uf = jnp.zeros([config.N, config.m])
    Vf = jnp.zeros([config.N+1, config.n]) #idk what this is
    # print("starting the loop")
    while not _exit:
        start = timer()

        X, U, V = _solve(
            reference,
            # parameter,
            # target,
            None, # parameter
            config.W,
            x0,
            Xf,
            Uf,
            Vf
            )
        # print(f"post solve iteration: {i}")

        X.block_until_ready()

        Xf = X
        Uf = U
        Vf = V

        # output.append((self.X0))
        # print(f"X post solve: {X}")
        # print(f"U post solve: {U}")

        g, c = jitted_model_evaluator(X,U)

        stop = timer()

        l2_cost = np.sum(g*g)

        if i == 0:
            print("{:<10} {:<20} {:<20} {:<20}".format("Iter", "Cost", "Constraint", "Time Elapsed"))
        print("{:<10d} {:<20.5f} {:<20.5f} {:<20.5f}".format(i, l2_cost, np.sum(c*c), stop-start))
        i += 1

        if i > max_iter:
            # print("exit because of max iter")
            _exit = True
        if last_cost - l2_cost < 1e-3 and np.sum(c*c) < 1e-5:
            print("exit because converged")
            _exit = True
        last_cost = l2_cost

    return Xf,Uf


# x0 = onehot @ config.model.embed.W_E
# x0 = x0 + config.model.pos_embed(config.input)

print(f"input = {config.input}")
print(f"target = {config.target}")

onehot = th.nn.functional.one_hot(config.input, num_classes=config.model.embed.d_vocab).float().to(config.device)

onehot.requires_grad_(True)
x = onehot @ config.model.embed.W_E

x = x + config.model.pos_embed(config.input)
# print(f"x0 outside: {x}")

assert(np.array_equal(np.asarray(config.p0), x.cpu().detach().numpy()))

xpre = config.model.ln_final(x)
logits_pre = config.model.unembed(xpre).squeeze(1)
ypre = logits_pre.argmax(-1)
print(f"incorrectly unembedded input: {ypre}")

# print(f"x0: {x}")

for i, block in enumerate(config.model.blocks):
    x = block(x)

x = config.model.ln_final(x[:,-1].unsqueeze(1))

logits = config.model.unembed(x).squeeze(1)
y = logits.argmax(-1)

# print(f"unsteered output = {y}")
# print(f"true output: {config.target}")

# X,U = runOffline(config.p0.squeeze(0), config.target)
X,U = runOffline(config.p0.squeeze(0).squeeze(0), config.target)

# print(f"Target token: {config.target}")

# print(f"input: {config.input}")

# print(f"X = {X.shape}")
# print(f"U = {U.shape}")
# p0 = config.ln(config.p0)
# logits_p = p0 @ config.W_U_jn
# y_pin = logits_p.argmax(-1)
# print(f"p_in: {y_pin}")

# print(f"X0: {X[0, :]}")
X0 = config.ln(X[0, :])
logits_0 = X0 @ config.W_U_jn
y_in = logits_0.argmax(-1)
print(f"\"steered\" (incorrectly unembedded) input: {y_in}")

Xf = config.ln(X[-1, :])
logits = Xf @ config.W_U_jn
y_jx = logits.argmax(-1)
print(f"steered output: {y_jx}")
# # print(f"perturbations: {U}")

# x_test = onehot @ config.model.embed.W_E

# x_test = x_test + config.model.pos_embed(config.input)
# for i, block in enumerate(config.model.blocks):
#     x_test = block(x_test) + th.tensor(np.asarray(U[i])).to(config.device)

# x_test = config.model.ln_final(x_test[:,-1].unsqueeze(1))

# logits = config.model.unembed(x_test).squeeze(1)
# y_test = logits.argmax(-1)
# print(f"y_test: {y_test}")

# p_test = config.p0

# for tf in config.tfs_jax:
#     p_test = tf(p_test)
#     # print("i like doggies")
# # print(f"x shape pre ln: {x.shape}")


# # ln_w = jnp.asarray(conmodel.ln_final.w.detach())
# # ln_eps = jnp.asarray(model.ln_final.ln_eps)
# # ln_b = jnp.asarray(model.ln_final.b.detach())
# p_test = jnp.squeeze(config.ln(p_test))
# # print(f"that late p: {p.shape}")


# # W_U_jn = jnp.asarray(model.unembed.W_U[None].detach())
# logits = p_test @ config.W_U_jn
# y_jx = logits.argmax(-1)

# print(f"y_jax: {y_jx}")
