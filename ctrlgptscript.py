import mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from timeit import default_timer as timer
import numpy as np
import gpt_config as config
from functools import partial
import jax.numpy as jnp
import jax 

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
    print("starting")
    cost = partial(config.cost, config.N)
    work = partial(optimizers.mpc, cost, config.dynamics, config.hessian_approx, False) # set limited memory to false
    _solve = jax.jit(work)
    

    # self.X0 = self.X0.at[:,:13+self.config.n_joints].set(reference[:,:13+self.config.n_joints])

    _cost = partial(config.cost,config.N, config.W,config.target)
    _dynamics = config.dynamics
    model_evaluator = partial(optimizers.model_evaluator_helper, _cost, _dynamics,x0)
    jitted_model_evaluator = jax.jit(model_evaluator)

    _exit = False
    max_iter = 100
    last_cost = 1e10
    i = 0
    # output = []
    # output.append((self.X0))

    Xf = jnp.zeros([config.N+1, config.n])
    Uf = jnp.zeros([config.N, config.m])
    Vf = jnp.zeros([config.N+1, config.n]) #idk what this is
    print("starting the loop")
    while not _exit:
        start = timer()

        X, U, V = _solve(
            target,
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

        g, c = jitted_model_evaluator(X,U)

        stop = timer()

        l2_cost = np.sum(g*g)

        if i == 0:
            print("{:<10} {:<20} {:<20} {:<20}".format("Iter", "Cost", "Constraint", "Time Elapsed"))
        print("{:<10d} {:<20.5f} {:<20.5f} {:<20.5f}".format(i, l2_cost, np.sum(c*c), stop-start))
        i += 1

        if i > max_iter:
            print("exit because of max iter")
            _exit = True
        if last_cost - l2_cost < 1e-3 and np.sum(c*c) < 1e-5:
            print("exit because converged")
            _exit = True
        last_cost = l2_cost

    return Xf,Uf


# x0 = onehot @ config.model.embed.W_E
# x0 = x0 + config.model.pos_embed(config.input)

X,U = runOffline(config.p0.squeeze(0).squeeze(0), config.target)

print(f"X: {X}")
print(f"U: {U}")

Xf = jnp.squeeze(config.ln(X[-1, :]),0)
logits = Xf @ config.W_U_jn
y_jx = logits.argmax(-1)
print(f"final token: {y_jx}")