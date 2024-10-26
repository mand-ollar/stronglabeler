"""
Strong label the audio.
All the tensor or array is run on PyTorch.
"""

import math

import matplotlib.pyplot as plt
import torch
import torchaudio


class StrongLabeler:

    sr: int = 16000

    def __init__(
        self,
    ) -> None:

        self.data: torch.Tensor
        self.abs_norm_data: torch.Tensor
        self.outline: torch.Tensor
        self.step: int
        self.threshold: float
        self.st_labels: list[int]
        self.en_labels: list[int]

    def __get_outline(
        self,
    ) -> None:
        """
        Make the audio signal into positive and normalize by the maximum value.

        data: torch.Tensor with shape (length,)
        """

        data_variance: float = torch.var(self.data).item()
        self.step = int(self.data.shape[0] * data_variance / 2)

        abs_data: torch.Tensor = torch.abs(self.data)
        self.abs_norm_data = abs_data / torch.max(abs_data)

        n_chunks: int = math.ceil(self.abs_norm_data.shape[0] / self.step)

        self.outline = torch.zeros_like(self.abs_norm_data)

        for i in range(n_chunks):
            self.outline[i * self.step : (i + 1) * self.step] = torch.max(
                self.abs_norm_data[i * self.step : (i + 1) * self.step]
            )

        for i in range(n_chunks):
            if i == n_chunks - 1:
                continue
            start = i * self.step
            end = (i + 1) * self.step
            self.outline[start:end] = torch.linspace(
                self.outline[start], self.outline[end], steps=self.step
            )

    def __get_label(
        self,
    ) -> None:
        self.st_labels = []
        self.en_labels = []

        min_gap: int = int(0.5 * self.sr)

        for i, data in enumerate(self.outline):
            if i == 0:
                if self.outline[i] > self.threshold:
                    self.st_labels.append(i)
                continue
            if i == self.outline.shape[0] - 1:
                if self.outline[i] > self.threshold:
                    self.en_labels.append(i)
                continue

            if self.outline[i - 1] < self.threshold and data > self.threshold:
                self.st_labels.append(i)
            if self.outline[i - 1] > self.threshold and data < self.threshold:
                self.en_labels.append(i)

        for st_label, en_label in zip(self.st_labels[1:], self.en_labels[:-1]):
            if st_label - en_label < min_gap:
                self.st_labels.remove(st_label)
                self.en_labels.remove(en_label)

    def label(
        self,
        data: torch.Tensor,
        threshold: float = 0.1,
    ) -> tuple[list[int], list[int]]:
        self.data = data
        self.threshold = threshold

        self.__get_outline()
        self.__get_label()

        return self.st_labels, self.en_labels
