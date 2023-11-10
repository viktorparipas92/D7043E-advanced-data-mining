import torch
from torch import Tensor


def dice_coefficient(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
) -> float:
    """Average of Dice coefficient for all batches, or for a single mask"""
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    not_reduce_batch_first = (input.dim() == 2 or not reduce_batch_first)
    sum_dim = (-1, -2) if not_reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coefficient(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
) -> float:
    """Average of Dice coefficient for all classes"""
    return dice_coefficient(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> float:
    """Dice loss (objective to minimize) between 0 and 1"""
    coefficient = (
        multiclass_dice_coefficient if multiclass else dice_coefficient
    )
    return 1 - coefficient(input, target, reduce_batch_first=True)