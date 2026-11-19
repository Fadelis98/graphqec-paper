from typing import Any, Dict, List, Tuple

import torch
from torch.utils.flop_counter import FlopCounterMode

from graphqec.benchmark.utils import format_number, format_table, timed


def count_parameters(
    model: torch.nn.Module, depth: int = None
) -> Tuple[int, Dict[str, int], Dict[str, Dict[str, Any]]]:
    """
    Count the total parameters and trainable parameters in a model.
    Additionally, return parameter details for each submodule up to the specified depth.

    Args:
        model (torch.nn.Module): The model to analyze.
        depth (int, optional): Maximum depth to traverse submodules. 
                               If None, traverse all submodules.

    Returns:
        Tuple[int, Dict[str, int], Dict[str, Dict[str, Any]]]:
            - Total number of parameters in the model.
            - Dictionary with total, trainable, and non-trainable parameter counts.
            - Nested dictionary with parameter details for each submodule.
    """

    def _count_recursive(module: torch.nn.Module, current_depth: int) -> Dict[str, Any]:
        if depth is not None and current_depth > depth:
            return {}

        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        for p in module.parameters(recurse=True):
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
            else:
                non_trainable_params += p.numel()

        submodule_details = {}
        for name, submodule in module.named_children():
            submodule_counts = _count_recursive(submodule, current_depth + 1)
            if submodule_counts:
                submodule_details[name] = submodule_counts

        result = {
            "class_name": type(module).__name__,
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable_params,
        }
        if submodule_details:
            result["submodules"] = submodule_details

        return result

    param_details = _count_recursive(model, current_depth=0)

    total_params = param_details["total"]
    trainable_params = param_details["trainable"]
    non_trainable_params = param_details["non_trainable"]

    return (
        total_params,
        {"trainable": trainable_params, "non_trainable": non_trainable_params},
        param_details.get("submodules", {}),
    )

def generate_parameter_table(
    submodule_details: Dict[str, Dict[str, Any]], 
    headers: List[str] = ["Module", "Class", "Total", "Trainable", "Non-Trainable"]
) -> str:
    """
    Generate a formatted table from submodule parameter details.

    Args:
        submodule_details (Dict[str, Dict[str, Any]]): Nested dictionary of parameter details.
        headers (List[str]): Table headers.

    Returns:
        str: Formatted table as a string.
    """
    data = []

    def _flatten_submodules(prefix: str, details: Dict[str, Any]):
        """
        Flatten nested submodule details into a list of rows.
        """
        row = [
            prefix,
            details["class_name"],
            format_number(details["total"]),
            format_number(details["trainable"]),
            format_number(details["non_trainable"]),
        ]
        data.append(row)

        if "submodules" in details:
            for name, subdetails in details["submodules"].items():
                _flatten_submodules(f"{prefix}.{name}" if prefix else name, subdetails)

    # Start flattening from the top-level submodules
    for name, details in submodule_details.items():
        _flatten_submodules(name, details)

    # Format the table
    return format_table(data, headers)

def analyze_layer(layer: torch.nn.Module, *inputs, show_layer_detail: bool = False) -> Dict[str, Any]:
    """
    Analyze a layer's FLOPs and parameters.
    Returns a dictionary with analysis results.
    """
    # Count parameters
    total_params, param_details, submodule_details  = count_parameters(layer, depth=3)
    param_table = generate_parameter_table(submodule_details)
    if show_layer_detail:
        print(param_table)
    
    # Count FLOPs
    flop_counter = FlopCounterMode(display=show_layer_detail, depth=3)
    with flop_counter as counter:
        _ = layer(*inputs)
    
    return {
        "total_params": total_params,
        "trainable_params": param_details["trainable"],
        "non_trainable_params": param_details["non_trainable"],
        "flops": counter.get_total_flops(),
        "params_str": format_number(total_params),
        "flops_str": format_number(counter.get_total_flops()),
    }

def compare_layers(
    layers, 
    data_constructor=None, 
    loss_fn=None, 
    num_runs=10, 
    num_warmup=5, 
    device="cuda", 
    dtype=None,
    show_detailed_timings=False,
    show_stats=True,
    show_layer_detail=False,
    return_results = False,
):
    """
    Compare the forward and backward behavior of multiple layers or benchmark a single layer.
    
    Args:
        layers: A list of layers to compare, or a single layer
        data_constructor: Function that returns tuple of (inputs, target) where inputs can be a single tensor or tuple
        loss_fn: Loss function to use for backward pass
        num_runs: Number of timing runs to perform
        num_warmup: Number of warm-up runs to perform
        device: Device to run on ("cuda" or "cpu")
        dtype: Data type to use for the layers and inputs
        show_detailed_timings: Whether to show detailed timing tables
        show_stats: Whether to show parameters and FLOPs statistics
        return_results: Whether to return a tuple that contains the results

    example usage:

    >>> def data_constructor():
    >>>     x = torch.randn((32, 128, 256))
    >>>     target = torch.randn((32, 128, 256))
    >>>     return x, target

    >>> # Create some example layers with different configurations
    >>> layer1 = torch.nn.Linear(256, 256)
    >>> layer2 = torch.compile(layer1, mode="max-autotune")
    >>> layer3 = torch.nn.Sequential(
    >>>     torch.nn.Linear(256, 512),
    >>>     torch.nn.ReLU(),
    >>>     torch.nn.Linear(512, 256)
    >>> )

    >>> # Compare the layers
    >>> results = compare_layers(
    >>>     [layer1, layer2, layer3],
    >>>     data_constructor=data_constructor,
    >>>     loss_fn=torch.nn.MSELoss(),
    >>>     num_warmup=5,
    >>>     num_runs=50,
    >>>     device="cuda",
    >>>     dtype=torch.float16,
    >>>     show_detailed_timings=True
    >>> )
    """
    if data_constructor is None:
        raise ValueError("data_constructor must be provided.")
    
    # Convert single layer to list for uniform processing
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    
    dtype = dtype or torch.get_default_dtype()
    layers = [layer.to(device=device, dtype=dtype) for layer in layers]
    
    # Analyze layers first if stats are requested
    layer_stats = []
    if show_stats:
        inputs, _ = data_constructor()
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        inputs = tuple(x.to(device=device, dtype=dtype if x.is_floating_point() else None) for x in inputs)
        for layer in layers:
            stats = analyze_layer(layer, *inputs, show_layer_detail=show_layer_detail)
            layer_stats.append(stats)

    def run_layer(layer, inputs, target):
        with torch.no_grad():
            _, forward_time = timed(lambda: layer(*inputs) if isinstance(inputs, tuple) else layer(inputs))
        
        if loss_fn is not None:
            output = layer(*inputs) if isinstance(inputs, tuple) else layer(inputs)
            if isinstance(output, tuple):
                output = output[0]
            loss = loss_fn(output, target)
            _, backward_time = timed(lambda: loss.backward())
        else:
            backward_time = 0.0
            
        return forward_time, backward_time
    
    forward_data = []
    backward_data = []
    
    # Run timing tests
    for i in range(num_warmup + num_runs):
        inputs, target = data_constructor()
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        inputs = tuple(x.to(device=device, dtype=dtype if x.is_floating_point() else None) for x in inputs)
        target = target.to(device=device, dtype=dtype)
        
        forward_row = [f"{i + 1}*" if i < num_warmup else i + 1 - num_warmup]
        backward_row = [f"{i + 1}*" if i < num_warmup else i + 1 - num_warmup]
        
        for layer in layers:
            forward_time, backward_time = run_layer(layer, inputs, target)
            forward_row.append(forward_time)
            if loss_fn is not None:
                backward_row.append(backward_time)
        
        forward_data.append(forward_row)
        if loss_fn is not None:
            backward_data.append(backward_row)

    # Calculate averages
    timing_times = forward_data[num_warmup:]
    forward_avgs = [sum(t[i+1] for t in timing_times) / num_runs for i in range(len(layers))]
    
    backward_avgs = []
    if loss_fn is not None:
        timing_times = backward_data[num_warmup:]
        backward_avgs = [sum(t[i+1] for t in timing_times) / num_runs for i in range(len(layers))]
    else:
        backward_avgs = [0.0] * len(layers)

    # Print detailed timing tables if requested
    if show_detailed_timings:
        forward_headers = ["Run"] + [f"Layer {i+1} (ms)" for i in range(len(layers))]
        print("\nForward Pass Times:")
        print(format_table(forward_data, forward_headers))
        
        if loss_fn is not None:
            print("\nBackward Pass Times:")
            backward_headers = ["Run"] + [f"Layer {i+1} (ms)" for i in range(len(layers))]
            print(format_table(backward_data, backward_headers))

    # Prepare and print summary table
    summary_data = []
    headers = ["Metric"]
    for i in range(len(layers)):
        headers.append(f"Layer {i+1}")

    # Add timing data
    summary_data.append(["Forward (ms)"] + [f"{avg:.2f}" for avg in forward_avgs])
    if loss_fn is not None:
        summary_data.append(["Backward (ms)"] + [f"{avg:.2f}" for avg in backward_avgs])
        summary_data.append(["Total (ms)"] + [f"{f+b:.2f}" for f, b in zip(forward_avgs, backward_avgs)])

    # Add stats if requested
    if show_stats:
        summary_data.extend([
            ["Parameters"] + [stats["params_str"] for stats in layer_stats],
            ["FLOPs"] + [stats["flops_str"] for stats in layer_stats],
            ["FLOPs/ms"] + [format_number(stats["flops"] / fwd) for stats, fwd in zip(layer_stats, forward_avgs)]
        ])
        if len(layers) > 1:
            summary_data.extend([
                ["Speed vs L1"] + ["1.00x"] + [f"{forward_avgs[0] / avg:.2f}x" for avg in forward_avgs[1:]],
                ["FLOPs ratio"] + ["1.00x"] + [f"{stats['flops'] / layer_stats[0]['flops']:.2f}x" for stats in layer_stats[1:]]
            ])

    print("\nSummary:")
    print(format_table(summary_data, headers))

    if return_results:
        return (
            list(zip(forward_avgs, backward_avgs)), 
            layer_stats if show_stats else None
        )