
from torchvision.transforms import v2 as T
from typing import Tuple, List
import torch


class ThreeCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform, online_transform="train", input_height=224):
        self.transform = transform

        if type(online_transform) == str:

            if online_transform == "train":
                self.online_transform = T.Compose(
                    [T.RandomResizedCrop(input_height),
                     T.RandomHorizontalFlip(),
                     T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),]
                )
            elif online_transform == "train_tensor":
                self.online_transform = T.Compose(
                    [T.RandomResizedCrop(input_height),
                     T.RandomHorizontalFlip()]
                )
            elif online_transform == "val_tensor":
                self.online_transform = T.Compose(
                    [
                        T.Resize(int(input_height + 0.1 * input_height)),
                        T.CenterCrop(input_height),

                    ]
                )

            else:

                self.online_transform = T.Compose(
                    [
                        T.Resize(int(input_height + 0.1 * input_height)),
                        T.CenterCrop(input_height),
                        T.PILToTensor(),
                        T.ConvertImageDtype(dtype=torch.float32),
                        T.ConvertImageDtype(torch.float32),
                    ]
                )
        else:
            self.online_transform = online_transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.online_transform(x)]


class SupportViewPrototypeTransform:
    """Create two crops of the same image"""

    def __init__(self, transform, support_view_transform, n_support_views=5, online_transform="train", input_height=224):
        self.transform = transform
        self.support_view_transform = support_view_transform
        self.n_support_views = n_support_views

        if type(online_transform) == str:

            if online_transform == "train":
                self.online_transform = T.Compose(
                    [T.RandomResizedCrop(input_height),
                     T.RandomHorizontalFlip(), 
                     T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),]
                )
            elif online_transform == "train_tensor":
                self.online_transform = T.Compose(
                    [T.RandomResizedCrop(input_height),
                     T.RandomHorizontalFlip()]
                )
            elif online_transform == "val_tensor":
                self.online_transform = T.Compose(
                    [
                        T.Resize(int(input_height + 0.1 * input_height)),
                        T.CenterCrop(input_height),

                    ]
                )
            elif online_transform == "val_new":
                self.online_transform = T.Compose(
                    [
                        T.Resize(int(input_height + 0.1 * input_height)),
                        T.CenterCrop(input_height),
                        T.ToImage(),
                        T.ToDtype(torch.float32, scale=True), 
                    ]
                )

            else:

                self.online_transform = T.Compose(
                    [
                        T.Resize(int(input_height + 0.1 * input_height)),
                        T.CenterCrop(input_height),
                        T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),
                    ]
                )
        else:
            self.online_transform = online_transform

    def __call__(self, x):

        return [self.transform(x), self.transform(x)] + [self.transform(x) for _ in range(self.n_support_views)] + [self.online_transform(x)]


def get_simclr_transform():
    """
    Returns a SimCLR-style data augmentation transform for ImageNet using torchvision.transforms.v2.
    
    SimCLR augments each image with two different random augmentations.
    """
    # Define the Gaussian Blur transform
    # For ImageNet (224x224), kernel size is typically set to 23 as in the SimCLR paper
    gaussian_blur = T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    
    # Compose the SimCLR augmentation pipeline
    simclr_transform = T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([gaussian_blur], p=0.5),
        
        T.ToDtype(torch.float32, scale=True),  # Replaces ToTensor()
        T.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet mean
                    std=[0.229, 0.224, 0.225])    # ImageNet std
    ])
    
    return simclr_transform



def get_test_transform():
    """
    Returns a SimCLR-style data augmentation transform for ImageNet using torchvision.transforms.v2.
    
    SimCLR augments each image with two different random augmentations.
    """
    # Define the Gaussian Blur transform
    # For ImageNet (224x224), kernel size is typically set to 23 as in the SimCLR paper
    gaussian_blur = T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    
    # Compose the SimCLR augmentation pipeline
    simclr_transform = T.Compose([
        T.ToImage(),
        T.Resize(256),
        T.RandomHorizontalFlip(),
        T.CenterCrop(224),        
        T.ToDtype(torch.float32, scale=True),  # Replaces ToTensor()
        T.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet mean
                    std=[0.229, 0.224, 0.225])    # ImageNet std
    ])
    
    return simclr_transform

class SimCLRTrainTransform:
    '''
        three simple augmentations: random cropping followed by resize back to the original size,
        random color distortions, and random Gaussian blur

        see https://arxiv.org/pdf/2002.05709.pdf appendix A
    '''

    def __init__(self, strength: float = 1.0,
                 img_height: int = 224,
                 normalize: Tuple[List[float],
                                  List[float]] = None,
                 color_jitter=[0.4, 0.4, 0.4, 0.1]) -> None:
        '''
        strength : float
            strength of color distortion
        img_height : int
            height of the image
        normalize : Tuple[List[float], List[float]]
            mean and std of the dataset
        '''
        self.strength = strength
        self.img_height = img_height

        # Color distortion is composed by color jittering and color dropping
        color_jitter_transform = T.ColorJitter(
            *[strength*item for item in color_jitter])

        # Random crop and resize to 224x224, standard Inception-style random cropping
        # Random horizontal left to right flip p=0.5

        self.transforms_list = [
            T.RandomResizedCrop(size=img_height, scale=(0.2,
                                                        1.0),),
            T.RandomHorizontalFlip(p=0.5),
            
            T.RandomApply(torch.nn.ModuleList([
                color_jitter_transform,
            ]), p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ConvertImageDtype(dtype=torch.float32),
            T.ToTensor(),
         
            # self._get_gausian_blur(),

        ]

        if normalize is not None:
            self.set_normalize(normalize)

        else:
            self.transform = T.Compose(
                self.transforms_list
            )

    def set_normalize(self, normalize):
        self.transform = T.Compose(
            self.transforms_list + [T.Normalize(*normalize)]
        )

        print(f"Set normalize to {normalize}")

    def __call__(self, image):

        # width, height = image.size

        return self.transform(image)

    # unused
    def _get_gausian_blur(self,):
        '''
        - blur the image 0.5 of the time using a Gaussian kernel
        - sigma [0.1, 2.0]
        - kernel size is set to be 0.1 of the image height/width.
        '''

        # set kernel size to be 0.1 of the image height/width and make sure it is odd integer
        kernel = int(0.1 * self.img_height)  # int(0.1 * min(width, height))
        if kernel % 2 == 0:
            kernel += 1
        return T.RandomApply([T.GaussianBlur(
            kernel_size=kernel, sigma=(0.1, 2.0))], p=0.5)
class MedicalGreyScaleTrainTransform(SimCLRTrainTransform):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        normalize = kwargs.get("normalize", None)
        img_height = kwargs.get("img_height", 224)

        self.transforms_list = [


            T.PILToTensor(),
             T.ToDtype(torch.float32, scale=True),
          
            #T.RandomAffine(8, translate=(0.05, 0.05), scale=(0.9, 1.1)),

            T.RandomResizedCrop(size=img_height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),

            T.RandomHorizontalFlip(),
            #T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),


            T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.8),

      

        ]
        print("normalioze is ", normalize)

        if normalize is not None:
            self.set_normalize(normalize)
        else:
            self.transform = T.Compose(
                self.transforms_list
            )
class IsicTrainTransform:
    '''
        three simple augmentations: random cropping followed by resize back to the original size,
        random color distortions, and random Gaussian blur

        see https://arxiv.org/pdf/2002.05709.pdf appendix A
    '''

    def __init__(self, strength: float = 1.0,
                 img_height: int = 224,
                 normalize: Tuple[List[float],
                                  List[float]] = None,
                 color_jitter=[0.4, 0.4, 0.4, 0.1]) -> None:
        '''
        strength : float
            strength of color distortion
        img_height : int
            height of the image
        normalize : Tuple[List[float], List[float]]
            mean and std of the dataset
        '''
        self.strength = strength
        self.img_height = img_height

        # Color distortion is composed by color jittering and color dropping
        color_jitter_transform = T.ColorJitter(
            *[strength*item for item in color_jitter])

        # Random crop and resize to 224x224, standard Inception-style random cropping
        # Random horizontal left to right flip p=0.5

        self.transforms_list = [
            T.RandomResizedCrop(size=img_height, scale=(0.5,
                                                        1.0),),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),
            T.ConvertImageDtype(torch.float32),
            T.RandomApply(torch.nn.ModuleList([
                color_jitter_transform,
            ]), p=0.3),
            T.RandomRotation(180),
            # self._get_gausian_blur(),

        ]

        if normalize is not None:
            self.set_normalize(normalize)

        else:
            self.transform = T.Compose(
                self.transforms_list
            )

    def set_normalize(self, normalize):
        self.transform = T.Compose(
            self.transforms_list + [T.Normalize(*normalize)]
        )

        print(f"Set normalize to {normalize}")

    def __call__(self, image):

        # width, height = image.size

        return self.transform(image)

    # unused
    def _get_gausian_blur(self,):
        '''
        - blur the image 0.5 of the time using a Gaussian kernel
        - sigma [0.1, 2.0]
        - kernel size is set to be 0.1 of the image height/width.
        '''

        # set kernel size to be 0.1 of the image height/width and make sure it is odd integer
        kernel = int(0.1 * self.img_height)  # int(0.1 * min(width, height))
        if kernel % 2 == 0:
            kernel += 1
        return T.RandomApply([T.GaussianBlur(
            kernel_size=kernel, sigma=(0.1, 2.0))], p=0.5)


class CardiacTrainTransform:
    def __init__(self, strength: float = 1.0,
                 img_height: int = 128, ) -> None:
        '''
        strength : float
            strength of color distortion
        img_height : int
            height of the image
        normalize : Tuple[List[float], List[float]]
            mean and std of the dataset
        '''
        self.strength = strength
        self.img_height = img_height

        # Random crop and resize to 224x224, standard Inception-style random cropping
        # Random horizontal left to right flip p=0.5

        self.transforms_list = [
            T.RandomHorizontalFlip(),
            T.RandomRotation(45),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            T.RandomResizedCrop(
                size=img_height, scale=(0.2, 1), antialias=True),
            T.ConvertImageDtype(dtype=torch.float32),
        ]

        self.transform = T.Compose(
            self.transforms_list
        )

    def __call__(self, image):

        # width, height = image.size
        return self.transform(image)


class HistoPathoTrainTransform:
    def __init__(self, strength: float = 1.0,
                 img_height: int = 50, ) -> None:
        '''
        strength : float
            strength of color distortion
        img_height : int
            height of the image
        normalize : Tuple[List[float], List[float]]
            mean and std of the dataset
        '''
        self.strength = strength
        self.img_height = img_height

        # Random crop and resize to 224x224, standard Inception-style random cropping
        # Random horizontal left to right flip p=0.5

        self.transforms_list = [
            T.Resize(int(img_height + 0.15 * img_height)),
            T.CenterCrop(img_height),
            T.RandomHorizontalFlip(),  # Flips the image horizontally
            T.RandomVerticalFlip(),    # Flips the image vertically
            # Moderate color jittering
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.02),
            T.RandomRotation(degrees=10),  # Slight rotations without cropping
            T.RandomAffine(degrees=0, translate=(
                0.05, 0.05)),  # Small translations
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),
        ]

        self.transform = T.Compose(
            self.transforms_list
        )

    def __call__(self, image):

        # width, height = image.size
        return self.transform(image)


class SimCLRValTransformHistoPatho:

    def __init__(self,
                 ):

        transforms = [
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),

        ]

        self.transform = T.Compose(
            transforms
        )

    def __call__(self, image):
        return self.transform(image)


class SimCLRValTransformCard:
    '''
        see https://arxiv.org/pdf/2002.05709.pdf appendix A
    '''

    def __init__(self,
                 img_height: int = 128,
                 # [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
                 normalize: Tuple[List[float], List[float]] = None

                 ):
        self.img_height = img_height

        transforms = [
            T.Resize(int(img_height + 0.15 * img_height),antialias=True),
            T.CenterCrop(img_height),
            T.ConvertImageDtype(dtype=torch.float32),

        ]

        self.transform = T.Compose(
            transforms
        )

    def __call__(self, image):
        return self.transform(image)


class SimCLRValTransform:
    '''
        see https://arxiv.org/pdf/2002.05709.pdf appendix A
    '''

    def __init__(self,
                 img_height: int = 224,
                 # [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
                 normalize: Tuple[List[float], List[float]] = None

                 ):
        self.img_height = img_height

        transforms = [
            T.Resize(int(img_height + 0.15 * img_height)),
            T.CenterCrop(img_height),
            T.PILToTensor(),
             T.ToDtype(torch.float32, scale=True),
        ]
        if normalize is not None:
            transforms.append(T.Normalize(*normalize))
        else:
            print("No normalization applied")

        self.transform = T.Compose(
            transforms
        )

    def __call__(self, image):
        return self.transform(image)


class SimCLRFineTuneTransform:
    '''
        see https://arxiv.org/pdf/2002.05709.pdf appendix A
    '''

    def __init__(self,
                 img_height: int = 224,
                 # [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
                 normalize: Tuple[List[float], List[float]] = None

                 ):
        self.img_height = img_height

        transforms = [
            T.Resize(int(img_height + 0.15 * img_height)),
            T.CenterCrop(img_height),
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),
        ]
        if normalize is not None:
            transforms.append(T.Normalize(*normalize))
        else:
            print("No normalization applied")

        self.transform = T.Compose(
            transforms
        )

    def __call__(self, image):
        return self.transform(image)


class BaselineTrainTransform:

    def __init__(self,
                 img_height: int = 224,
                 normalize: Tuple[List[float], List[float]] =
                 None):
        self.img_height = img_height

        transforms = [
            T.RandomResizedCrop(size=img_height, scale=(0.08,
                                                        1.0), ratio=(0.75, 1.3333333333333333)),
            T.RandomHorizontalFlip(p=0.5),
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),
        ]
        if normalize is not None:
            transforms.append(T.Normalize(*normalize))

        self.transform = T.Compose(
            transforms
        )

    def __call__(self, image):

        return self.transform(image)


class SupportViewTransform:

    def __init__(self,
                 img_height: int = 224,
                 normalize: Tuple[List[float], List[float]] =
                 None):
        self.img_height = img_height

        transforms = [
            T.RandomResizedCrop(size=img_height, scale=(0.08,
                                                        0.7), ratio=(0.75, 1.3333333333333333)),
            T.RandomHorizontalFlip(p=0.5),
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),
        ]
        if normalize is not None:
            transforms.append(T.Normalize(*normalize))

        self.transform = T.Compose(
            transforms
        )

    def __call__(self, image):

        return self.transform(image)


class NViewsTransform:
    """Create N augmentations of the same image"""

    def __init__(self, transform, n_views=2):
        self.transform = transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.n_views)]
    
