"""Image preprocessing shared by training and inference."""

from torchvision import transforms

IMAGE_SIZE = 128
MEAN = (0.7229, 0.5555, 0.5390)
STD = (0.1875, 0.1964, 0.2101)


def build_transform(training: bool = False):
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(steps)
