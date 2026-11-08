"""
Clasificador visual de residuos de EcoBot CDMX (Avance de proyecto 2).

CNN con transferencia de aprendizaje desde ImageNet. Arquitecturas evaluadas:
  * EfficientNet-B0 (Tan y Le, 2019)  – 5.3 M parámetros
  * ResNet50 (He et al., 2016)         – 25.6 M parámetros
Los pesos preentrenados se descargan de las versiones publicadas de la biblioteca timm
en GitHub, para que el mismo código funcione en Google Colab y en local.

El modelo reconoce el MATERIAL (6 clases de TrashNet). La categoría de la norma
NADF-024 la decide la base de conocimiento del chatbot a partir de ese material.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

CLASES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NOMBRE_ES = {"cardboard": "cartón", "glass": "vidrio", "metal": "metal", "paper": "papel",
             "plastic": "plástico", "trash": "basura no reciclable"}
# Material -> id de residuo en base_conocimiento.json (None = categoría genérica)
A_RESIDUO = {"cardboard": "cartón", "glass": "vidrio", "metal": "latas y metales", "paper": "papel",
             "plastic": "envases de plástico rígido", "trash": None}

PESOS = {
    "efficientnet_b0": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth",
    "resnet50": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth",
}
MEDIA, DESV = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TAM = 224

TRANSF_EVAL = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(TAM), transforms.ToTensor(), transforms.Normalize(MEDIA, DESV)])
TRANSF_AUMENTO = transforms.Compose([
    transforms.RandomResizedCrop(TAM, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.02), transforms.ToTensor(), transforms.Normalize(MEDIA, DESV)])


def crear_red(arquitectura: str, num_clases: int = len(CLASES), preentrenada: bool = True) -> nn.Module:
    """Crea la CNN con la cabeza de ImageNet reemplazada por una de `num_clases` salidas."""
    red = timm.create_model(arquitectura, pretrained=False)
    if preentrenada:
        sd = torch.hub.load_state_dict_from_url(PESOS[arquitectura], map_location="cpu", progress=False)
        red.load_state_dict(sd)
    red.reset_classifier(num_clases)
    return red


def congelar_hasta(red: nn.Module, arquitectura: str, entrenables: int) -> None:
    """Congela toda la red salvo los últimos `entrenables` bloques y la cabeza."""
    for p in red.parameters():
        p.requires_grad = False
    if arquitectura == "efficientnet_b0":
        bloques = list(red.blocks)[-entrenables:] if entrenables else []
        modulos = bloques + [red.conv_head, red.bn2, red.classifier]
    else:
        capas = [red.layer1, red.layer2, red.layer3, red.layer4]
        modulos = (capas[-entrenables:] if entrenables else []) + [red.fc]
    for m in modulos:
        for p in m.parameters():
            p.requires_grad = True


class ClasificadorResiduos:
    """Envoltura de inferencia que usa el chatbot."""

    def __init__(self, ruta_modelo: str | Path, dispositivo: str | None = None):
        ck = torch.load(ruta_modelo, map_location="cpu", weights_only=False)
        self.arquitectura, self.clases = ck["arquitectura"], ck["clases"]
        self.umbral = ck.get("umbral", 0.6)
        self.red = crear_red(self.arquitectura, len(self.clases), preentrenada=False)
        self.red.load_state_dict(ck["estado"])
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.red.to(self.dispositivo).eval()

    @torch.no_grad()
    def predecir(self, imagen) -> tuple[str, float, list[tuple[str, float]]]:
        """Recibe una ruta o una imagen PIL; devuelve (clase, probabilidad, top-3)."""
        if not isinstance(imagen, Image.Image):
            imagen = Image.open(imagen)
        x = TRANSF_EVAL(imagen.convert("RGB")).unsqueeze(0).to(self.dispositivo)
        p = torch.softmax(self.red(x), dim=1)[0].cpu().numpy()
        orden = np.argsort(p)[::-1]
        top = [(self.clases[i], float(p[i])) for i in orden[:3]]
        return top[0][0], top[0][1], top
