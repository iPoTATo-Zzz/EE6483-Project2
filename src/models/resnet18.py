"""Frozen ImageNet ResNet-18 baseline; keep BatchNorm statistics fixed."""
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def baseline_transform():
    return ResNet18_Weights.IMAGENET1K_V1.transforms()
