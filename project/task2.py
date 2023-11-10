#!/usr/bin/env python
# coding: utf-8

# # Image segmentation with U-Net
# Inspired by [this repository](https://)

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Setting up

# ## Import dependencies

# In[2]:


from model import UNet

import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss


# ## Setting the GPU device

# In[3]:


device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


# # Defining helper functions

# ## Loading the data

# ### Creating dataset from given directories

# In[4]:


def _create_dataset(
    image_directory: Path,
    mask_directory: Path,
    image_scale: float,
):
    try:
        dataset = CarvanaDataset(image_directory, mask_directory, image_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(image_directory, mask_directory, image_scale)

    return dataset


# ### Splitting dataset into training and validation sets

# In[5]:


def _split_dataset(dataset, validation_percentage: float):
    num_validation = int(len(dataset) * validation_percentage)
    num_training = len(dataset) - num_validation
    training_set, validation_set = random_split(
        dataset,
        [num_training, num_validation],
        generator=torch.Generator().manual_seed(0),
    )
    return training_set, validation_set


# ### Creating data loaders

# In[6]:


def _create_data_loaders(training_set, validation_set, batch_size):
    loader_arguments = dict(
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    training_loader = DataLoader(
        training_set, shuffle=True, **loader_arguments
    )
    validation_loader = DataLoader(
        validation_set, shuffle=False, drop_last=True, **loader_arguments
    )
    return training_loader, validation_loader


# ## Checkpointing

# In[7]:


image_directory = Path('./data/images/')
mask_directory = Path('./data/masks/')
checkpoint_directory = Path('./checkpoints/')


# In[8]:


def save_checkpoint(model, dataset, checkpoint_directory, epoch):
    Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    state_dict['mask_values'] = dataset.mask_values
    torch.save(state_dict, str(checkpoint_directory / f'checkpoint_epoch{epoch}.pth'))
    logging.info(f'Checkpoint {epoch} saved!')


# ## Training

# ### Determining the loss for a batch

# In[9]:


def _get_batch_loss(model, batch, criterion, automatic_mixed_precision):
    images, true_masks = batch['image'], batch['mask']
    invalid_shape_message = (
        f'Network has been defined with {model.n_channels} input channels, '
        f'but loaded images have {images.shape[1]} channels. Please check that '
        'the images are loaded correctly.'
    )
    assert images.shape[1] == model.n_channels, invalid_shape_message
    images = images.to(
        device=device, dtype=torch.float32, memory_format=torch.channels_last
    )
    true_masks = true_masks.to(device=device, dtype=torch.long)
    with torch.autocast(
        device.type if device.type != 'mps' else 'cpu',
        enabled=automatic_mixed_precision,
    ):
        predicted_masks = model(images)
        if model.n_classes == 1:
            loss = criterion(predicted_masks.squeeze(1), true_masks.float())
            loss += dice_loss(
                F.sigmoid(predicted_masks.squeeze(1)),
                true_masks.float(),
                multiclass=False,
            )
        else:
            loss = criterion(predicted_masks, true_masks)
            loss += dice_loss(
                F.softmax(predicted_masks, dim=1).float(),
                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )

        return images, loss, true_masks, predicted_masks


# ### Logging

# In[10]:


def _initialize_logging(
    epochs,
    batch_size,
    learning_rate,
    validation_percentage,
    save_checkpoint,
    image_scale,
    num_training,
    num_validation,
    automatic_mixed_precision,
):
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment_config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'val_percent': validation_percentage,
        'save_checkpoint': save_checkpoint,
        'img_scale': image_scale,
        'automatic_mixed_precision': automatic_mixed_precision,
    }
    experiment.config.update(experiment_config)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {num_training}
        Validation size: {num_validation}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {image_scale}
        Mixed Precision: {automatic_mixed_precision}
    ''')


# ### Evaluation

# In[11]:


def _evaluate(
    validation_loader,
    epoch,
    num_training,
    batch_size,
    global_step,
    scheduler,
    optimizer,
    images,
    true_masks,
    predicted_masks,
    automatic_mixed_precision,
    experiment,
):
    division_step = (num_training // (5 * batch_size))
    if division_step > 0:
        if global_step % division_step == 0:
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score = evaluate(model, validation_loader, device, automatic_mixed_precision)
            scheduler.step(val_score)

            logging.info(f'Validation Dice score: {val_score}')
            try:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': val_score,
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(predicted_masks.argmax(dim=1)[0].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch,
                    **histograms
                })
            except:
                pass


# # Putting it together

# ## Parameters

# In[12]:


EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
VALIDATION_PERCENTAGE = 0.1
SAVE_CHECKPOINT = True
IMAGE_SCALE = 0.5
AUTOMATIC_MIXED_PRECISION = False
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999
GRADIENT_CLIPPING = 1.0

BILINEAR = True
N_CHANNELS = 3  # for RGB images
N_CLASSES = 2  # the number of probabilities you want to get per pixel

STATE_DICT_PATH = ''
LOAD_STATE_DICT = False


# ## Loading the model

# In[13]:


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.info(f'Using device {device}')

model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=BILINEAR)
model = model.to(memory_format=torch.channels_last)

logging.info(
    f'Network:\n'
     f'\t{model.n_channels} input channels\n'
     f'\t{model.n_classes} output channels (classes)\n'
     f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
)


# In[14]:


if LOAD_STATE_DICT:
    state_dict = torch.load(STATE_DICT_PATH, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {STATE_DICT_PATH}')


# In[15]:


model.to(device=device)


# ## Loading the data

# In[16]:


dataset = _create_dataset(image_directory, mask_directory, IMAGE_SCALE)


# In[17]:


training_set, validation_set = _split_dataset(dataset, VALIDATION_PERCENTAGE)
num_training = len(training_set)
num_validation = len(validation_set)


# In[18]:


training_loader, validation_loader = _create_data_loaders(
    training_set, validation_set, BATCH_SIZE
)


# ## Configuring the training

# In[19]:


experiment = _initialize_logging(
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    VALIDATION_PERCENTAGE,
    SAVE_CHECKPOINT,
    IMAGE_SCALE,
    num_training,
    num_validation,
    AUTOMATIC_MIXED_PRECISION,
)


# In[20]:


optimizer = optim.RMSprop(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    momentum=MOMENTUM,
    foreach=True,
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'max', patience=5
)
grad_scaler = torch.cuda.amp.GradScaler(enabled=AUTOMATIC_MIXED_PRECISION)
criterion = torch.nn.CrossEntropyLoss() if model.n_classes > 1 else torch.nn.BCEWithLogitsLoss()


# ## Train the model

# In[21]:


global_step = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    with tqdm(
        total=num_training, desc=f'Epoch {epoch}/{EPOCHS}', unit='img'
    ) as progress_bar:
        for batch in training_loader:
            images, loss, true_masks, predicted_masks = _get_batch_loss(
                model, batch, criterion, AUTOMATIC_MIXED_PRECISION
            )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            progress_bar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()
            experiment.log({'train loss': loss.item(), 'step': global_step, 'epoch': epoch})
            progress_bar.set_postfix(**{'loss (batch)': loss.item()})

            _evaluate(
                validation_loader,
                epoch,
                num_training,
                BATCH_SIZE,
                global_step,
                scheduler,
                optimizer,
                images,
                true_masks,
                predicted_masks,
                AUTOMATIC_MIXED_PRECISION,
                experiment,
            )

    if SAVE_CHECKPOINT:
        save_checkpoint(model, dataset, checkpoint_directory, epoch)


# 
