"""Este é o código central da segmentação"""

import os
import glob
import cv2
import argparse

import torch
import torch.nn.functional as F

import util.io

from torchvision.transforms import Compose
from raiz.models import ViTSegmentation
from raiz.transforms import Resize, NormalizeImage, PrepareForNet


def run(input_path, output_path, model_path, model_type="ViT_hybrid", optimize=True):
    """Kernel do algoritmo
        input_path (str): caminho de entrada
        output_path (str): caminho de saida
        model_path (str): caminho que salvo variaveis do modelo
    """
    print("initialize")

    # seleciona o device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    net_w = net_h = 480

    # carregando o modelo
    if model_type == "ViT_large":
        model = ViTSegmentation(
            150,
            path=model_path,
            backbone="vitl16_384",
        )
    elif model_type == "ViT_hybrid":
        model = ViTSegmentation(
            150,
            path=model_path,
            backbone="vitb_rn50_384",
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' ta errado, use: --model_type [ViT_large|ViT_hybrid]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # recolhe o input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # cria a pasta de saida
    os.makedirs(output_path, exist_ok=True)

    print("Comecando o processamento")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # entrada
        img = util.io.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # processamento
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            out = model.forward(sample)

            prediction = torch.nn.functional.interpolate(
                out, size=img.shape[:2], mode="bicubic", align_corners=False
            )
            prediction = torch.argmax(prediction, dim=1) + 1
            prediction = prediction.squeeze().cpu().numpy()

        # resultado
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_segm_img(filename, img, prediction, alpha=0.5)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input", help="pasta de entrada"
    )

    parser.add_argument(
        "-o", "--output_path", default="output_semseg", help="pasta de saida"
    )

    parser.add_argument(
        "-m",
        "--model_weights",
        default=None,
        help="Caminho dos pesos do modelo",
    )

    # 'ViT_large', 'ViT_hybrid'
    parser.add_argument("-t", "--model_type", default="ViT_hybrid", help="model type")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.set_defaults(optimize=True)

    args = parser.parse_args()

    default_models = {
        "ViT_large": "weights/UltimosPesos.pt",
        "ViT_hybrid": "weights/UltimosPesos.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # configuração adicional do torch
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # gerando os mapas de atenção
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )
