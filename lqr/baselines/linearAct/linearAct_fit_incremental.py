import torch as th
from data_handling_linearAct import LayerActivationCollector, PartialLinearActApplier, save_linearact_state


def fit_linear_map_sorted(A: th.Tensor, B: th.Tensor):

    A = th.sort(A, dim=0).values #(N, n) 
    B = th.sort(B, dim=0).values

    meanA = A.mean(dim=0)
    meanB = B.mean(dim=0)

    varA = ((A - meanA) ** 2).mean(dim=0).clamp_min(1e-8)
    covAB = ((A - meanA) * (B - meanB)).mean(dim=0)

    omega = covAB / varA #(n, )
    beta = meanB - omega * meanA
    return omega, beta


@th.no_grad()
def fit_linearact_incremental(
    model,
    tokenizer,
    prompts_src,
    prompts_tgt,
    num_samples: int,
    out_state_name: str,
    batch_size: int = 32,
    seed: int = 0,
    use_support: bool = True, # q_0_100
):
    """
    causal fitting:
      layer 0: fit on raw model
      layer 1: fit after applying layer 0 map
      layer 2: fit after applying layer 1 map
      ...
    """
    device = next(model.parameters()).device
    collector = LayerActivationCollector(model, tokenizer)

    T = len(model.model.layers)
    n = model.model.embed_tokens.embedding_dim

    omega = th.ones((T, n), device=device)
    beta = th.zeros((T, n), device=device)

    # support for q_0_100 (transport only within observed source range)
    min_src = th.full((T, n), float("-inf"), device=device)
    max_src = th.full((T, n), float("inf"), device=device)

    for layer_idx in range(T):
        print(f"\n[FIT] Layer {layer_idx}/{T-1}")

        # apply maps for layers < layer_idx
        applier = None
        if layer_idx > 0:
            applier = PartialLinearActApplier(
                model=model,
                omega=omega,
                beta=beta,
                min_src=min_src,
                max_src=max_src,
                max_layer=layer_idx,
                strength=1.0,
                use_support=use_support,
            )

        A = collector.collect_layer_samples(
            prompts=prompts_src,
            num_samples=num_samples,
            layer_idx=layer_idx,
            batch_size=batch_size,
            seed=seed,
            apply_partial=applier,
            pooling_op="mean",
        ).to(device)

        B = collector.collect_layer_samples(
            prompts=prompts_tgt,
            num_samples=num_samples,
            layer_idx=layer_idx,
            batch_size=batch_size,
            seed=seed + 1,
            apply_partial=applier,
            pooling_op="mean",
        ).to(device)

        # record support on *source* for mitigation (q_0_100)
        min_src[layer_idx] = A.min(dim=0).values
        max_src[layer_idx] = A.max(dim=0).values

        w, b = fit_linear_map_sorted(A, B)
        omega[layer_idx] = w
        beta[layer_idx] = b

        print(f"  omega mean={omega[layer_idx].mean().item():.4f}, beta mean={beta[layer_idx].mean().item():.4f}")

    state = {
        "omega": omega.detach().cpu(),
        "beta": beta.detach().cpu(),
        "min_src": min_src.detach().cpu(),
        "max_src": max_src.detach().cpu(),
        "meta": {
            "num_samples": num_samples,
            "batch_size": batch_size,
            "seed": seed,
            "use_support": use_support,
        },
    }
    save_linearact_state(out_state_name, state)
