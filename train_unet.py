import os
from datetime import datetime
from typing import Tuple
import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Internal functions
from nnet import UNet, summary
from loader import ButterFly
from loss import DiceLoss

ex = Experiment('baseline_unet')


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


@ex.config
def config():
    # Model params
    init_features = 2
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logdir = f'runs/baseline_unet{init_features}-' + datetime.now(
    ).strftime('%y%m%d-%H%M%S')

    # Batch-size
    batch_size = 128
    path = "./metadata.json"

    # Epoch information
    iterations = 500
    resume_iteration = None
    checkpoint_interval = None
    validation_interval = 50
    learning_rate_decay_steps = 50
    learning_rate_decay_rate = 0.98

    learning_rate = 1e-4
    clip_gradient_norm = 3
    ex.observers.append(FileStorageObserver.create(logdir))


@ex.capture
def make_dataloader(path, batch_size):
    train_dataset = ButterFly(metadata_path=path, group='train')

    val_dataset = ButterFly(metadata_path=path, group='test')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_loader, val_loader


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
        optimizer,
        step_size=learning_rate_decay_steps,
        gamma=learning_rate_decay_rate
    )
    dice_loss = DiceLoss()
    return resume_iteration, model, optimizer, scheduler, dice_loss


@ex.automain
def train(
    logdir,
    device,
    iterations,
    resume_iteration,
    checkpoint_interval,
    clip_gradient_norm,
    validation_interval,
):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # NOTE: Init core component
    train_loader, val_loader = make_dataloader()
    resume_iteration, model, optimizer, scheduler, dice_loss = make_model()

    # NOTE: Code suggestion enforce
    assert isinstance(train_loader, DataLoader), ""
    assert isinstance(val_loader, DataLoader), ""
    assert isinstance(model, UNet), ""
    assert isinstance(optimizer, torch.optim.Adam), ""
    assert isinstance(scheduler, StepLR), ""
    assert isinstance(dice_loss, DiceLoss), ""

    # NOTE: Start train
    loop = tqdm(range(resume_iteration + 1, iterations + 1), desc="Epoch...")
    model.train()
    for i, batch in zip(loop, cycle(train_loader)):
        imag = batch['image'].to(device)
        label = batch['mask'].to(device)

        pred = model(imag)
        loss = dice_loss.forward(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        writer.add_scalar('train/loss', loss.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                _loss = []
                for val_batch in val_loader:
                    imag = val_batch['image'].to(device)
                    label = val_batch['mask'].to(device)
                    pred = model(imag)
                    loss = dice_loss.forward(pred, label)
                    _loss.append(loss.item())
                    pass
                writer.add_scalar('test/loss', np.mean(_loss), global_step=i)
                pass
            model.train()
