import gc
import os
from datetime import datetime
from typing import Tuple
import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loader import ButterFly

# Internal functions
from nnet import UNet, summary
from aug_loader import AugButterFly, CachedAugButterfly
from loss import DiceLoss

ex = Experiment('augmentation_unet')


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


@ex.config
def config():
    # Model params
    init_features = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logdir = f'runs/augment_unet{init_features}-' + datetime.now().strftime('%y%m%d-%H%M%S')

    # Batch-size
    batch_size = 32
    gradient_acc_step = 4
    path = "./metadata.json"
    train_aug_path = "./data/augment_leedsbutterfly/train/metadata.json"
    test_aug_path = "./data/augment_leedsbutterfly/test/metadata.json"

    # Epoch information
    iterations = 2000 * gradient_acc_step
    resume_iteration = None
    checkpoint_interval = None
    validation_interval = 100 * gradient_acc_step
    learning_rate_decay_steps = 100 * gradient_acc_step
    learning_rate_decay_rate = 0.98

    learning_rate = 1e-3
    clip_gradient_norm = 3
    ex.observers.append(FileStorageObserver.create(logdir))


@ex.capture
def make_dataloader(path, train_aug_path, test_aug_path, batch_size):
    train_dataset = CachedAugButterfly(train_aug_path)
    val_trans_dataset = CachedAugButterfly(test_aug_path)
    val_original_dataset = ButterFly(metadata_path=path, group='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_trans_loader = DataLoader(
        val_trans_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_original_loader = DataLoader(
        val_original_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_loader, val_trans_loader, val_original_loader


@ex.capture
def make_model(
    logdir,
    device,
    resume_iteration,
    learning_rate,
    learning_rate_decay_steps,
    learning_rate_decay_rate,
    init_features
) -> Tuple[int, UNet, torch.optim.Adam, StepLR, DiceLoss]:
    model = UNet(init_features=init_features)
    model.to(device)

    if resume_iteration is None:
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    summary(model)
    scheduler = StepLR(
        optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate
    )
    dice_loss = DiceLoss()
    dice_loss.to(device)
    return resume_iteration, model, optimizer, scheduler, dice_loss


@ex.automain
def train(
    logdir,
    device,
    iterations,
    resume_iteration,  # checkpoint_interval,
    clip_gradient_norm,
    validation_interval,
    gradient_acc_step
):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # NOTE: Init core component
    train_loader, val_trans_loader, val_original_loader = make_dataloader()
    resume_iteration, model, optimizer, scheduler, dice_loss = make_model()

    # NOTE: Code suggestion enforce
    assert isinstance(train_loader, DataLoader), ""
    assert isinstance(val_trans_loader, DataLoader), ""
    assert isinstance(val_original_loader, DataLoader), ""
    assert isinstance(model, UNet), ""
    assert isinstance(optimizer, torch.optim.Adam), ""
    assert isinstance(scheduler, StepLR), ""
    assert isinstance(dice_loss, DiceLoss), ""

    # NOTE: multiple GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # NOTE: Start train
    loop = tqdm(range(resume_iteration + 1, iterations + 1), desc="Iterations...")
    model.train()
    for i, batch in zip(loop, cycle(train_loader)):
        imag = batch['trans_image'].to(device)
        label = batch['trans_mask'].to(device)

        pred = model(imag)
        optimizer.zero_grad()

        loss = dice_loss.forward(pred, label)
        loss.backward()

        if (i % gradient_acc_step == 0):
            optimizer.step()
            scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        writer.add_scalar('train/loss', loss.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                _loss = []
                for val_batch in val_original_loader:
                    imag = val_batch['image'].to(device)
                    label = val_batch['mask'].to(device)
                    pred = model(imag)
                    loss = dice_loss.forward(pred, label).to('cpu')
                    _loss.append(loss.item())
                    pass
                writer.add_scalar('test/orignal/loss', np.mean(_loss), global_step=i)

                torch.cuda.empty_cache()
                _ = gc.collect()

                _loss = []
                for val_batch in val_trans_loader:
                    imag = val_batch['trans_image'].to(device)
                    label = val_batch['trans_mask'].to(device)
                    pred = model(imag)
                    loss = dice_loss.forward(pred, label).to('cpu')
                    _loss.append(loss.item())
                    pass
                writer.add_scalar('test/transform/loss', np.mean(_loss), global_step=i)

                torch.cuda.empty_cache()
                _ = gc.collect()

            model.train()
            pass
        pass

    # NOTE: Save model
    print(f"Save model for inference")
    if isinstance(model, nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), os.path.join(logdir, f"model-state-dict.pt"))
