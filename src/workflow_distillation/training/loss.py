"""Equal-trajectory reduction using the verified chunked vocabulary CE."""


def batch_loss(hidden, head, labels, groups, denominators, coefficients):
    from workflow_distillation.weighted_sft_loss import chunked_hidden_loss

    size = hidden.shape[0]
    if not (
        1 <= size <= 2
        and size == len(denominators) == len(coefficients)
        and labels.shape == groups.shape == hidden.shape[:2]
    ):
        raise ValueError("Expected one or two aligned trajectories")
    return sum(
        chunked_hidden_loss(
            hidden[i : i + 1],
            head,
            labels[i : i + 1],
            groups[i : i + 1],
            denominators[i],
            coefficient,
            chunk_size=64,
        )
        for i, coefficient in enumerate(coefficients)
    )
