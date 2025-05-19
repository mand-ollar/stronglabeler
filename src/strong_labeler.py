"""
Strong label the audio.
All the tensor or array is run on PyTorch.
"""

import math

import torch


class StrongLabeler:
    def __init__(self, sr: int, outline_step: float = 0.01) -> None:
        self.data: torch.Tensor
        self.abs_norm_data: torch.Tensor
        self.outline: torch.Tensor
        self.step: int
        self.min_target_sec: float
        self.min_gap_sec: float
        self.threshold: float
        self.st_labels: list[int]
        self.en_labels: list[int]

        self.sr: int = sr
        self.outline_step: float = outline_step

    def __get_outline(self) -> None:
        """
        Make the audio signal into positive and normalize by the maximum value.

        data: torch.Tensor with shape (length,)
        """

        # data_variance: float = torch.var(self.data).item() + 1e-6
        # self.step = int(self.data.shape[0] * data_variance / 5)

        self.step = int(self.outline_step * self.sr)

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
            self.outline[start:end] = torch.linspace(self.outline[start], self.outline[end], steps=self.step)

    def __get_label(self) -> None:
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

        st_labels: list[int] = []
        en_labels: list[int] = []

        for st_label, en_label in zip(self.st_labels, self.en_labels):
            if en_label - st_label < self.min_target_sec * self.sr:
                continue
            # elif abs(self.data[st_label:en_label].abs().max() - self.data.abs().max()) > 1e-6:
            #     continue
            else:
                st_labels.append(st_label)
                en_labels.append(en_label)

        self.st_labels = st_labels
        self.en_labels = en_labels

        if self.st_labels and self.en_labels:
            st_labels = [self.st_labels[0]]
            en_labels = []

            for st_label, en_label in zip(self.st_labels[1:], self.en_labels[:-1]):
                if st_label - en_label < min_gap:
                    continue
                else:
                    st_labels.append(st_label)
                    en_labels.append(en_label)

            en_labels.append(self.en_labels[-1])

            self.st_labels = st_labels
            self.en_labels = en_labels

    def label(
        self,
        data: torch.Tensor,
        min_target_sec: float,
        min_gap_sec: float,
        threshold: float,
    ) -> tuple[list[int], list[int]]:
        self.data = data
        self.min_target_sec = min_target_sec
        self.min_gap_sec = min_gap_sec
        self.threshold = threshold

        self.__get_outline()
        self.__get_label()

        return self.st_labels, self.en_labels
