import torch
import torch.nn as nn
import torchaudio
import math


def linear_softmax_pooling(x: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, time_steps, num_class)
    return (x ** 2).sum(1) / x.sum(1)


class Crnn6(nn.Module):
    def __init__(self, sample_rate: int, num_class: int):
        super().__init__()
        self.melspec_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            win_length=40 * sample_rate // 1000,
            hop_length=20 * sample_rate // 1000,
            n_fft=2048,
            n_mels=64,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     sample_rate: int, audio sampling rate
        #     num_class: int, the number of output classes
        ##############################

        # 网络结构
        self.bn = nn.BatchNorm2d(64)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # 新增一层
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # 再新增一层
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # 再再新增一层
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.bigru = nn.GRU(input_size=512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_class, bias=True)

    def detection(self, x: torch.Tensor):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        batch_size, time_steps, num_freq = x.shape
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = x.transpose(1, 3)  # [B, F, T, 1]
        x = self.bn(x)
        x = x.transpose(1, 3)  # [B, 1, T, F]
        x = self.cnn(x)
        x = x.transpose(1, 2)  # [B, T, C, F]
        x = x.reshape(batch_size, time_steps, -1)  # [B, T, C*F]
        x, _ = self.bigru(x)
        x = torch.sigmoid(self.fc(x))  # [B, T, num_class]
        return x
    
    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)
        x = x.transpose(1, 2)
        frame_prob = self.detection(x)  # (batch_size, time_steps, num_class)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, num_class)
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }
    
