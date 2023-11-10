import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dice_score import dice_coefficient, multiclass_dice_coefficient


@torch.inference_mode()
def evaluate(
    network: torch.nn.Module,
    validation_data: DataLoader,
    device: torch.device,
    automatic_mixed_precision: bool = False,
):
    network.eval()

    number_of_validation_batches = len(validation_data)
    dice_score = 0

    with torch.autocast(device.type, enabled=automatic_mixed_precision):
        for batch in tqdm(
            validation_data,
            total=number_of_validation_batches,
            desc='Validation round',
            unit='batch',
            leave=False,
        ):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(
                device=device,
                dtype=torch.float32,
                memory_format=torch.channels_last,
            )
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_predicted = network(image)

            if network.n_classes == 1:
                assert (
                    mask_true.min() >= 0 and mask_true.max() <= 1,
                    'True mask indices should be in [0, 1]',
                )
                mask_predicted = (F.sigmoid(mask_predicted) > 0.5).float()
                dice_score += dice_coefficient(
                    mask_predicted, mask_true, reduce_batch_first=False
                )
            else:
                assert (
                    (
                        mask_true.min() >= 0
                        and mask_true.max() < network.n_classes
                    ),
                    'True mask indices should be in [0, n_classes[',
                )
                mask_true = (
                    F.one_hot(mask_true, network.n_classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )
                mask_predicted = (
                    F.one_hot(mask_predicted.argmax(dim=1), network.n_classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coefficient(
                    mask_predicted[:, 1:],
                    mask_true[:, 1:],
                    reduce_batch_first=False,
                )

    network.train()
    return dice_score / max(number_of_validation_batches, 1)