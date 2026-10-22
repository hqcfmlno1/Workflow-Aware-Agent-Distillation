"""Explicit replica initialization and gradient reduction, without DDP hooks."""


def initialize_replica(decoder):
    import torch
    import torch.distributed as dist

    named = [(n, p) for n, p in decoder.named_parameters() if p.requires_grad]
    signature = [(n, tuple(p.shape), str(p.dtype)) for n, p in named]
    signatures = [None] * dist.get_world_size()
    dist.all_gather_object(signatures, signature)
    if not named or any(item != signature for item in signatures):
        raise ValueError("Trainable replica parameter signatures differ")
    with torch.no_grad():
        for _, parameter in named:
            dist.broadcast(parameter, src=0)


def average_gradients(parameters):
    import torch
    import torch.distributed as dist

    invalid = torch.tensor(
        int(any(p.grad is None or p.grad.is_sparse for p in parameters)),
        device=parameters[0].device,
    )
    dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    if bool(invalid):
        raise RuntimeError("Missing or sparse adapter gradients on a replica")
    for parameter in parameters:
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(dist.get_world_size())
