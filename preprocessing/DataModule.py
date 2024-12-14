import pytorch_lightning as pl
from torch.utils.data import DataLoader
import constants as cst


class DataModule(pl.LightningDataModule):
    """ Splits the datasets in TRAIN, VALIDATION. """

    def   __init__(self, train_set, val_set, batch_size, test_batch_size,  num_workers=16):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        if train_set.data.device.type != cst.DEVICE:       #this is true only when we are using a GPU but the data is still on the CPU
            self.pin_memory = True
        else:
            self.pin_memory = False
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
