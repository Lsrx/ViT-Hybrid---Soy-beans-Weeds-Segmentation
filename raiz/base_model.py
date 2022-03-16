import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Carregamento de modelo por arquivo (parei de usar o keras)

        Args:
            path (str): caminho do arquivo
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
