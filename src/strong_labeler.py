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
        self.min_gap_sec: float
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
        # self.step = int(self.data.shape[0] * data_variance / 5)
        self.step = int(0.05 * self.sr)

        abs_data: torch.Tensor = torch.abs(self.data)
        self.abs_norm_data = abs_data / torch.max(abs_data)

        n_chunks: int = math.ceil(self.abs_norm_data.shape[0] / self.step)

        self.outline = torch.zeros_like(self.abs_norm_data)

        for i in range(n_chunks):
            self.outline[i * self.step : (i + 1) * self.step] = torch.max(
                self.abs_norm_data[i * self.step : (i + 1) * self.step]
            )

        for i in range(n_chunks - 1):
            start = i * self.step
            end = min((i + 1) * self.step, self.abs_norm_data.shape[0])
            self.outline[start:end] = torch.linspace(
                self.outline[start], self.outline[end], steps=self.step
            )

    def __get_label(
        self,
    ) -> None:
        self.st_labels = []
        self.en_labels = []

        min_gap: int = int(self.min_gap_sec * self.sr)

        above_threshold = self.outline > self.threshold
        changes = torch.diff(above_threshold.int())

        self.st_labels = (changes == 1).nonzero(as_tuple=True)[0].tolist()
        self.en_labels = (changes == -1).nonzero(as_tuple=True)[0].tolist()

        # Handle edge cases
        if above_threshold[0]:
            self.st_labels.insert(0, 0)
        if above_threshold[-1]:
            self.en_labels.append(len(self.outline) - 1)

        for st_label, en_label in zip(self.st_labels[1:], self.en_labels[:-1]):
            if st_label - en_label < min_gap:
                self.st_labels.remove(st_label)
                self.en_labels.remove(en_label)

        for st_label, en_label in zip(self.st_labels, self.en_labels):
            if en_label - st_label < 0.1 * self.sr:
                self.st_labels.remove(st_label)
                self.en_labels.remove(en_label)

    def label(
        self,
        data: torch.Tensor,
        min_gap_sec: float = 0.5,
        threshold: float = 0.1,
    ) -> tuple[list[int], list[int]]:
        self.data = data
        self.min_gap_sec = min_gap_sec
        self.threshold = threshold

        self.__get_outline()
        self.__get_label()

        return self.st_labels, self.en_labels
