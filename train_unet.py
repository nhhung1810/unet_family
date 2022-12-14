import os
from datetime import datetime
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
    logdir = 'runs/baseline_unet-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Batch-size
    batch_size = 2
    path = "./metadata.json"

    # Epoch information
    iterations = 100
    resume_iteration = None
    checkpoint_interval = 10
    validation_interval = 10
    learning_rate_decay_steps = 10
    learning_rate_decay_rate = 0.98

    learning_rate = 0.0006
    clip_gradient_norm = 3
    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(
    logdir,
    device,
    iterations,
    resume_iteration,
    checkpoint_interval,
    path,
    batch_size,
    learning_rate,
    learning_rate_decay_steps,
    learning_rate_decay_rate,
    clip_gradient_norm,
    validation_interval,
):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = 'train', 'test'

    train_dataset = ButterFly(metadata_path=path, group=train_groups)

    # val_dataset = ButterFly(metadata_path=path, group=train_groups)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    # val_loader = DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    # )

    model = UNet(init_features=2)
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

    loop = tqdm(range(resume_iteration + 1, iterations + 1), desc="Epoch...")
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

        # if i % validation_interval == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         for key, value in evaluate(validation_dataset, model).items():
        #             writer.add_scalar(
        #                 'validation/' + key.replace(' ', '_'),
        #                 np.mean(value),
        #                 global_step=i
        #             )
        #     model.train()

        # if i % checkpoint_interval == 0:
        #     # torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
        #     model.to("cpu")
        #     torch.save(
        #         model.state_dict(), os.path.join(logdir, f'model-{i}.pt')
        #     )
        #     model.to(device)
        #     torch.save(
        #         optimizer.state_dict(),
        #         os.path.join(logdir, 'last-optimizer-state.pt')
        #     )
