hook_configs = {
    "linearact":{
        "hook_type": "linear_ot",
        "quantiles_src": "q_0_100",
    },
    # "meanact": {
    #     "hook_type": "mean_ot",
    #     "quantiles_src": "q_all",
    # },
    "pid": {
        "hook_type": "mean_ot_pid",
        "quantiles_src": "q_all",
    },
}

model_configs = {
    "llama1b": {
        "model_path": "meta-llama/Llama-3.2-1B",
        "module_patterns": [
            "model.layers.*.mlp.up_proj",
            "model.layers.*.mlp.down_proj",
            "model.layers.*.mlp.gate_proj",
        ],
    },
    # "gemma2b": {
    #     "model_path": "meta-llama/Llama-3.2-1B",
    #     "module_patterns": [
    #         ".*post_attention_layernorm",
    #         ".*post_feedforward_layernorm",
    #     ],
    # },
    # "qwen3b": {
    #     "model_path": "Qwen/Qwen2.5-3B",
    #     "module_patterns": [
    #         "model.layers.*.mlp.up_proj",
    #         "model.layers.*.mlp.down_proj",
    #         "model.layers.*.mlp.gate_proj",
    #     ],
    # },
    # "llama8b": {
    #     "model_path": "meta-llama/Meta-Llama-3-8B",
    #     "module_patterns": [
    #         "model.layers.*.mlp.up_proj",
    #         "model.layers.*.mlp.down_proj",
    #         "model.layers.*.mlp.gate_proj",
    #     ],
    # },
    # "gemma9b": {
    #     "model_path": "google/gemma-2-9b",
    #     "module_patterns": [
    #         ".*post_attention_layernorm",
    #         ".*post_feedforward_layernorm",
    #     ],
    # },
    # "qwen14b": {
    #     "model_path": "Qwen/Qwen2.5-14B",
    #     "module_patterns": [
    #         "model.layers.*.mlp.up_proj",
    #         "model.layers.*.mlp.down_proj",
    #         "model.layers.*.mlp.gate_proj",
    #     ],
    # },
}
