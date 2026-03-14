# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor
import librosa


class SALMONNDataset(Dataset):
    def __init__(self, ann_path, whisper_path):
        super().__init__()

        self.annotation = json.load(open(ann_path, "r"))["annotation"]

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        return {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        # 读取音频
        audio, sr = sf.read(ann["path"])
        if len(audio.shape) == 2:  # 立体声转单声道
            audio = audio[:, 0]
        
        dropout_rate = 0.1  # 初始化 dropout rate
        audio_length = len(audio)
            
        # 基于 index 设置随机种子，确保可重复性
        rng = random.Random(index)
            
        # 使用滑动窗口范围进行采样
        pool_size = 10  # 每个滑动窗口的大小
        sampled_indices = []
        for i in range(0, audio_length, pool_size):
            # 从范围 i 到 i + pool_size 随机采样一个点
            start = i
            end = min(i + pool_size, audio_length)
            if rng.random() > dropout_rate and start < end:  # 根据 dropout rate 判断是否保留
                sampled_indices.append(rng.randint(start, end - 1))
            
        # 根据采样的索引获取音频片段
        audio = audio[sampled_indices]
        
        # 添加扩展音频
        if "expand_wav" in ann:
            for p in ann["expand_wav"]:
                expand_audio, _ = sf.read(p)
                if len(expand_audio.shape) == 2:
                    expand_audio = expand_audio[:, 0]
                sil = np.zeros(1600, dtype=float)
                audio = np.concatenate((audio, sil, expand_audio), axis=0)
        
        # 如果音频长度小于采样率（1秒），用静音填充
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        
        # 截断音频至最多 30 秒
        audio = audio[:30 * sr]

        # 处理波形数据生成谱图
        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        
        # 获取标签和任务信息
        text = ann["text"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        # 返回样本
        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": text,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }