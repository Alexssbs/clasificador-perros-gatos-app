# cnn_model.py
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 2),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def load_cnn_checkpoint(path, device="cpu"):
    """Carga el .pt de la CNN y devuelve el modelo listo para inferencia."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_to_idx"] != {"cat": 0, "dog": 1}:
        raise ValueError("El checkpoint no tiene las clases esperadas.")
    model = CatDogCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def cnn_preprocess(image_pil):
    """Preprocesamiento exacto del notebook: EXIF, RGB, resize 224x224, ToTensor.
    NO aplica Normalize (los píxeles quedan en [0,1])."""
    image = ImageOps.exif_transpose(image_pil).convert("RGB")
    if image.size != (224, 224):
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
    return ToTensor()(image).unsqueeze(0)