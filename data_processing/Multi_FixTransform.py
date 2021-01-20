#modified from https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
from torchvision import transforms
from data_processing.RandAugment import RandAugment
from data_processing.Image_ops import GaussianBlur
class Multi_Fixtransform(object):
    def __init__(self,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,normalize,
            aug_times,init_size=224):
        """
        :param size_crops: list of crops with crop output img size
        :param nmb_crops: number of output cropped image
        :param min_scale_crops: minimum scale for corresponding crop
        :param max_scale_crops: maximum scale for corresponding crop
        :param normalize: normalize operation
        :param aug_times: strong augmentation times
        :param init_size: key image size
        """
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        trans=[]
        #key image transform
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(init_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trans.append(self.weak)
        self.aug_times=aug_times
        trans_weak=[]
        trans_strong=[]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )

            strong = transforms.Compose([
            randomresizedcrop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            RandAugment(n=self.aug_times, m=10),
            transforms.ToTensor(),
            normalize
            ])
            weak=transforms.Compose([
            randomresizedcrop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
            trans_weak.extend([weak]*nmb_crops[i])
            trans_strong.extend([strong]*nmb_crops[i])
        trans.extend(trans_weak)
        trans.extend(trans_strong)
        self.trans=trans
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops
