#-*- coding:utf-8 -*-
# import uvicorn
# from fastapi import FastAPI
# from fastapi.responses import FileResponse
import uuid


import matplotlib.pyplot as plt

import os
import json
import math

import scipy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
#from transforms import Translate

from scipy.io.wavfile import write
import argparse
parser = argparse.ArgumentParser(description='查看传参')
parser.add_argument("--text",type=str,default="小宇宙中只剩下漂流瓶和生态球。漂流瓶隐没于黑暗里，在一千米见方的宇宙中，只有生态球里的小太阳发出一点光芒。在这个小小的生命世界中，几只清澈的水球在零重力环境中静静地飘浮着，有一条小鱼从一只水球中蹦出，跃入另一只水球，轻盈地穿游于绿藻之间。在一小块陆地上的草丛中，有一滴露珠从一片草叶上脱离，旋转着飘起，向太空中折射出一缕晶莹的阳光。")
parser.add_argument("--character",type=int,default=17)   #魈
args = parser.parse_args()


class vits:
    def __init__(self):
        self.load_moudle()

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        cc = torch.LongTensor(text_norm)
        return cc

    def load_moudle(self):
        self.hps = utils.get_hparams_from_file("./configs/Genshin.json")

        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,  #
            **self.hps.model)
        _ = self.net_g.eval()

        _ = utils.load_checkpoint("./checkpoints/G_213000.pth", self.net_g, None)  # G_389000.pth

    def get_re(self, text,character):
        stn_tst = self.get_text(text)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            sid = torch.LongTensor([character])  # 指定第几个人说话
            audio = \
                self.net_g.infer(x_tst, x_tst_lengths,sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1.2)[0][0, 0].data.cpu().float().numpy()
            uui = uuid.uuid1()
            scipy.io.wavfile.write("./record/%s.wav" % uui, self.hps.data.sampling_rate, audio)
            return './record/%s.wav' % uui

    def voice_conversion(self,character):
        dataset = TextAudioSpeakerLoader(self.hps.data.validation_files, self.hps.data)
        collate_fn = TextAudioSpeakerCollate()
        loader = DataLoader(dataset, num_workers=8, shuffle=False,
                            batch_size=1, pin_memory=True,
                            drop_last=True, collate_fn=collate_fn)
        data_list = list(loader)
        x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x for x in data_list[0]]
        print(sid_src.numpy())   #香菱 “我的餐室”
        with torch.no_grad():
            sid_tgt = torch.LongTensor([character])  # 后说话人
            audio = self.net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][0, 0].data.cpu().float().numpy()
            uui = uuid.uuid1()
            scipy.io.wavfile.write("./record/%s.wav" % uui, self.hps.data.sampling_rate, audio)
            return './record/%s.wav' % uui

vits = vits()
# app = FastAPI()
#
#
# @app.get('/run')
# def run_moudle(text):
#     print(text)
#     ava = vits.get_re(text)
#     return FileResponse(ava)
#
#
# uvicorn.run(app, host='0.0.0.0', port=8003)
text = args.text
character = args.character
ava = vits.get_re(text,character)