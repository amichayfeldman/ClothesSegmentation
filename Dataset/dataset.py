import imgaug.augmenters as iaa
import h5py
import numpy as np
import cv2
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torch.utils.data import Dataset, DataLoader
import os


sometimes = lambda aug: iaa.Sometimes(0.3, aug)


class data_set(Dataset):
    def __init__(self, h5_file_path, img_w, img_h, mode):
        self.file = h5py.File(h5_file_path, 'r')
        self.mode = mode
        self.img_h, self.img_w = img_h, img_w
        self.mean_mat = np.tile(np.array([0.485, 0.456, 0.406]), [self.img_h, self.img_w, 1])
        self.std_mat = np.tile(np.array([0.229, 0.224, 0.225]), [self.img_h, self.img_w, 1])
        self.crop_w, self.crop_h = np.random.randint(200, max(img_h, img_w), size=(2,))

    def __len__(self):
        return len(self.file['data'])

    def change_crop_size(self):
        self.crop_w, self.crop_h = np.random.randint(200, max(self.img_h, self.img_w), size=(2,))

    def __getitem__(self, idx):
        img = self.file['data'][idx]
        img = cv2.resize(src=img, dsize=(self.img_w, self.img_h))
        gt_map = self.file['gt'][idx]
        gt_map = cv2.resize(src=gt_map, dsize=(self.img_w, self.img_h)).astype('int32')
        gt_map = cv2.flip(gt_map, 0)  # flip gt_cause origin DeepLab produces flipped output

        if self.mode == 'test':
            # normalized_img = img / img.max()
            # norm_img = np.divide(normalized_img - self.mean_mat, self.std_mat)
            norm_img = (2 * (img / 255)) - 1

            # gt_one_hot = (np.arange(self.num_of_classes) == gt_map[..., None]).astype(int)
            return {'image': np.moveaxis(norm_img, -1, 0),
                    # 'gt': np.moveaxis(gt_one_hot, -1, 0),
                    'gt_reg_map': gt_map,
                    'idx': idx}

        else:
            segmap = SegmentationMapsOnImage(gt_map, shape=img.shape)
            seq = iaa.Sequential([
                                #   iaa.CropToFixedSize(width=self.crop_w, height=self.crop_h),
                                  sometimes(iaa.OneOf([iaa.GaussianBlur((0, 3.0)),
                                                       iaa.AverageBlur(k=(2, 4)),
                                                       ])),
                                  sometimes(iaa.ChangeColorTemperature((1100, 10000))),
                                  sometimes(iaa.OneOf([iaa.Add((-10, 10), per_channel=0.5),
                                                       iaa.Multiply((0.85, 1.15), per_channel=0.5)])),
                                  # iaa.OneOf([iaa.GammaContrast((0.5, 2.0)),
                                  #            iaa.GammaContrast((0.5, 2.0), per_channel=True),
                                  #            iaa.LogContrast(gain=(0.6, 1.4))]),
                                  # sometimes(iaa.JpegCompression(compression=(80, 99))),
                                #   sometimes(iaa.CropAndPad(percent=(-0.5, 0.5))),
                                  sometimes(iaa.Grayscale(alpha=(0.0, 1.0)))
                                  ])
            # pdb.set_trace()
            augmented_img, augmented_map = seq(image=img.astype('uint8'), segmentation_maps=segmap)
            augmented_map = augmented_map.arr[..., 0]
            # flip - lr:
            if np.random.uniform() > 0.5:
                augmented_img = cv2.flip(augmented_img, 1)
                augmented_map = cv2.flip(augmented_map, 1)
            # normalized_img = np.true_divide(augmented_img, augmented_img.max())
            # norm_img = np.true_divide(normalized_img - self.mean_mat, self.std_mat)
            norm_img = (2 * (augmented_img / 255)) - 1
            # gt_one_hot = (np.arange(self.num_of_classes) == augmented_map[..., None]).astype(int)
            # pdb.set_trace()
            sample = {'image': np.moveaxis(norm_img, -1, 0),
                    #   'gt': np.moveaxis(gt_one_hot, -1, 0),
                      'gt_reg_map': augmented_map,
                      'idx': idx}
            return sample


def get_dataloaders(config):
    data_folder = config['Paths']['data_folder']
    batch_size = config['Params']['batch_size']
    img_h, img_w = config['Params']['target_size']

    train_dataset = data_set(h5_file_path=os.path.join(data_folder, 'train_data.h5'),
                             img_w=img_w, img_h=img_h, mode='train')
    val_dataset = data_set(h5_file_path=os.path.join(data_folder, 'val_data.h5'),
                           img_w=img_w, img_h=img_h, mode='val')
    test_dataset = data_set(h5_file_path=os.path.join(data_folder, 'test_data.h5'),
                            img_w=img_w, img_h=img_h, mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
