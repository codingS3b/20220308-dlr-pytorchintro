import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
## for the code to work, install torchvision
## $ python -m pip install --user -I --no-deps torchvision
import torchvision
import pytorch_lightning as pl
from torchvision import datasets, transforms
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

# Fixing random seeds for numpy, torch for obtaining
# reproducible results when executing the script multiple
# times
pl.seed_everything(seed=42)


def load_data(
    somepath,
    norm_loc=(0.1307,),  ## mu of normal dist to normalize by
    norm_scale=(0.3081,),  ## sigma of normal dist to normalize by
    train_kwargs={"batch_size": 64},
    test_kwargs={"batch_size": 1000},
    use_cuda=torch.cuda.device_count() > 0,
):
    """load MNIST data and return train/test loader object"""

    transform_ = transforms.Compose(
        # TODO where do the magic numbers come from?
        [transforms.ToTensor(), transforms.Normalize(norm_loc, norm_scale)]
    )

    train_dataset = datasets.MNIST(
        somepath, download=True, transform=transform_, train=True
    )
    test_dataset = datasets.MNIST(
        somepath, download=True, transform=transform_, train=False
    )

    if use_cuda:
        train_kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": True})
        test_kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": False})

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    return train_loader, test_loader


class MyNetwork(pl.LightningModule):
    """a very basic relu neural network involving conv, dense, max_pool and dropout layers"""

    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # x.shape = (batchsize, 10)
        output = F.log_softmax(x, dim=1)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)

        loss = F.nll_loss(prediction, y)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=1.e-3)

        return optimizer


def main(somepath="./pytorch-data"):
    """load the data set and run a random init CNN on it"""

    # is a GPU available?
    cuda_present = torch.cuda.is_available()
    ndevices = torch.cuda.device_count()
    use_cuda = cuda_present and ndevices > 0

    train_loader, test_loader = load_data(somepath, use_cuda=use_cuda)
    model = MyNetwork()

    max_nepochs = 3
    log_interval = 100

    init_params = list(model.parameters())[0].clone().detach()
    model.train(True)
    logfolder = Path("lightning_outputs")
    if not logfolder.is_dir():
        logfolder.mkdir()

    ckpt_callback = ModelCheckpoint(
        filename='{epoch:03.0f}-{train_loss:.3f}',
        save_last=True,
        save_top_k=1,
        monitor="train_loss",
        every_n_epochs=1
    )

    trainer = pl.Trainer(
        default_root_dir=logfolder,
        max_epochs=max_nepochs,
        log_every_n_steps=log_interval,
        accelerator="gpu" if use_cuda else "cpu",
        devices=ndevices if use_cuda else 1,
        callbacks=[
            ckpt_callback
        ]
    )

    trainer.fit(model, train_dataloaders=train_loader)

    final_params = list(model.parameters())[0].clone().detach()
    assert not torch.allclose(init_params, final_params)

    # when to reload chkp, e.g. for doing inference
    model_from_ckpt = MyNetwork.load_from_checkpoint(
        ckpt_callback.last_model_path
    )
    loaded_params = list(model_from_ckpt.parameters())[0]
    assert torch.allclose(loaded_params, final_params)


if __name__ == "__main__":
    main()
    print("Ok. Checkpoint on training with lightning reached.")

    # this logs to tensorboard automatically, access is via

    # tensorboard --logdir lightning_outputs