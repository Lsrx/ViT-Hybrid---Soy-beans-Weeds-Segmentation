# ViT-Hybrid---Soy-beans-Weeds-Segmentation

## Segmentação de Ervas no Cultivo da Soja Utilizando o ViT

### Setup 

1) Baixe e instale os pesos de treinamento na pasta de pesos:
  
2) Instale os arquivos necessários: 

    ```shell
    pip install -r requirements.txt
    ```

   The code was tested with Python 3.7, PyTorch 1.8.0, OpenCV 4.5.1, and timm 0.4.5

### Uso

1) Coloque as imagens a serem segmentadas em "input".

2) Rode o algoritmo:

    ```shell
    python run_segmentation.py
    ```

3) Os resultados vão para a pasta `output_semseg`.

OBS: Para obter os pesos do último treinamento realizar o download da seguinte pasta do drive, e colocar em uma pasta com nome "weights": 

https://drive.google.com/drive/folders/10moyrAqmw_dVCGHfwS9irqgHXngjbTmR?usp=sharing


### Agradecimentos

Meu algoritmo utilizou o modus operandi do seguinte projeto: [timm](https://github.com/rwightman/pytorch-image-models) and [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding). Um grande agradecimento por fazer esse projeto público.
