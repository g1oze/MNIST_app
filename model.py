import numpy as np
import torch
from torch import nn


class Predictor:
    def __init__(self, weights_path: str = "model_weights.pth", device: str = "cpu") -> None:
        self.weights_path = weights_path
        self.device = torch.device(device)
        self.model = self._build_model().to(self.device)

    @staticmethod
    def _build_model() -> nn.Module:
        return nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 10),
        )

    def _load_weights(self) -> None:
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, data):
        self._load_weights()
        x = self._prepare_input(data)

        with torch.no_grad():
            logits = self.model(x)
            predictions = torch.argmax(logits, dim=1)

        return predictions.cpu().numpy()

    def predict_proba(self, data):
        self._load_weights()
        x = self._prepare_input(data)

        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def _prepare_input(self, data) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            x = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            x = data.float()
        else:
            x = torch.tensor(data, dtype=torch.float32)

        return x.reshape(-1, 28 * 28).to(self.device)
