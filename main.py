# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from edsr_torch_data_utils import TrainDatasets, ValDatasets, display_transform
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from EDSR_model import edsr_model_pytorch


def train_and_test_pytorch_vdsr(
    batch_size=4,
    upscale_factor=4,
    crop_size=128,
    train_dir='',
    val_dir='',
):
    print('-' * 15 + '> start loading data... ', '[ {} ]'.format(datetime.now()))

    # dataloader and remember pytorch needs to write the dataloader file
    train_set = TrainDatasets(data_dir=train_dir, crop_size=crop_size, upscale_factor=upscale_factor)
    val_set = ValDatasets(data_dir=val_dir, crop_size=crop_size, upscale_factor=upscale_factor)
    # then we need to feed the data into DataLoader

    print('-' * 15 + '> data loading......')

    return train_set, val_set


if __name__ == '__main__':
    data_train, data_val = train_and_test_pytorch_vdsr(train_dir='/Users/neonrocks/PycharmProjects/SRGAN/datasets/train',
                                                       val_dir='/Users/neonrocks/PycharmProjects/SRGAN/datasets/val')
    data = data_val.__getitem__(0)
    print(data)
