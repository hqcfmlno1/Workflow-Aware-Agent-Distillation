"""Mean cross-entropy over the supervised assistant tokens in each trajectory."""

from workflow_distillation.lora_sft.loss import group_loss_parameters

ASSISTANT_WEIGHTS = (0, 1, 1, 1, 1)


def loss_parameters(groups):
    return group_loss_parameters(groups, ASSISTANT_WEIGHTS, length_alpha=0)
