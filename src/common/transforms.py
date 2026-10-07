"""D2 training augmentation; validation always uses the weight preset."""
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def augmented_transform(config):
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=tuple(config["crop_scale"]),
                                     ratio=tuple(config["crop_ratio"]),
                                     interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(p=config["horizontal_flip_probability"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
