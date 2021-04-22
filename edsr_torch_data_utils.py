"""
    date:       2021/4/21 11:02 上午
    written by: neonleexiang
"""
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_img_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calc_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


'''
    class torchvision.transforms.Compose(transforms)
    将多个transform组合起来使用。

    transforms.Compose([
     transforms.CenterCrop(10),
     transforms.ToTensor(),
    ])

'''


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor(),
    ])


def display_transform():
    """

    :return:
    """
    '''
        resize to 400 for better img plotting
    '''
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor(),
    ])


'''
    when we training, we use crop_size to create our model but when we evaluate,
    we should notice the size of the images should be as the same size as the input of network layer
'''


class TrainDatasets:
    def __init__(self, data_dir, crop_size, upscale_factor):
        super(TrainDatasets, self).__init__()
        self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_img_file(x)]
        self.crop_size = calc_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(self.crop_size)
        self.lr_transform = train_lr_transform(self.crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_img = self.hr_transform(Image.open(self.img_filenames[index]))
        lr_img = self.lr_transform(hr_img)
        return lr_img, hr_img

    def __len__(self):
        return len(self.img_filenames)


'''
    the default setting of the size of the val dataset is min(w, h), 
    but the input size of the network layer is the crop size, 
    so we should make sure it has the same size of the layer input.
'''


class ValDatasets:
    def __init__(self, data_dir, crop_size, upscale_factor):
        super(ValDatasets, self).__init__()
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_img_file(x)]
        self.hr_transform = train_hr_transform(self.crop_size)
        self.lr_transform = train_lr_transform(self.crop_size, upscale_factor)

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        # hr_img = Image.open(self.img_filenames[index])
        # w, h = hr_img.size
        # crop_size = calc_valid_crop_size(min(w, h), self.upscale_factor)
        # crop_size = self.crop_size
        # lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        # hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        # hr_img = CenterCrop(crop_size)(hr_img)
        # lr_img = lr_scale(hr_img)
        # hr_restore_img = hr_scale(lr_img)
        # return ToTensor()(lr_img), ToTensor()(hr_restore_img), ToTensor()(hr_img)
        '''
            setting image by using the same method as TrainDataset to avoid the error with different size of the input
        '''
        hr_img = self.hr_transform(Image.open(self.img_filenames[index]))
        lr_img = self.lr_transform(hr_img)

        hr_restore_img = ToTensor()(Resize(self.crop_size, interpolation=Image.BICUBIC)(ToPILImage()(lr_img)))

        return lr_img, hr_restore_img, hr_img

    def __len__(self):
        return len(self.img_filenames)


class TestDatasets:
    def __init__(self, data_dir, upscale_factor):
        super(TestDatasets, self).__init__()
        self.lr_path = data_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = data_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [os.path.join(self.lr_path, x) for x in os.listdir(self.lr_path) if is_img_file(x)]
        self.hr_filenames = [os.path.join(self.hr_path, x) for x in os.listdir(self.hr_path) if is_img_file(x)]

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        '''
            hr_img is the ground true
            and hr_restore_img is using Resize method and set the interpolation to be bicubic method
        '''
        img_name = self.lr_filenames[index].split('/')[-1]
        lr_img = Image.open(self.lr_filenames[index])
        w, h = lr_img.size
        hr_img = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_img)
        return img_name, ToTensor()(lr_img), ToTensor()(hr_restore_img), ToTensor()(hr_img)

    def __len__(self):
        return len(self.lr_filenames)


if __name__ == '__main__':
    '''
        test dataset dir
    '''
    # test = TrainDatasets('datasets/val/', 128, 4)
    test = ValDatasets('datasets/val/', 128, 4)
    print(test.__getitem__(0)[0].shape)
    print(type(test.__getitem__(0)[1]))
    # print(test.__getitem__(0)[2].shape)
