from functools import reduce
from typing import Literal

import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

from soundstream.encoder import Encoder
from soundstream.decoder import Decoder


class SoundStream(nn.Module):
    def __init__(self, n_q, codebook_size, D, C, strides=(2, 4, 5, 8)):
        super(SoundStream, self).__init__()

        # The temporal resampling ratio between input waveform and embeddings.
        # Not used in here, but helpful for consumers.
        self.M = reduce(lambda a, b: a * b, strides)

        self.encoder = Encoder(C=C, D=D, strides=strides)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q,
            codebook_size=codebook_size,
            dim=D,
            kmeans_init=True,
            kmeans_iters=100,
            threshold_ema_dead_code=2
        )
        self.decoder = Decoder(C=C, D=D, strides=strides)

    def forward(
            self,
            x,
            mode: Literal['end-to-end', 'encode', 'decode'] = 'end-to-end',
        ):
        # x: batch_size x 1 x (T / 1)
        # e: batch_size x (T / M) x D --- where M is product of all numbers in `strides` tuple
        # o: batch_size x 1 x (T / 1)

        if mode == 'end-to-end':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            o = self.decoder(quantized.permute((0,2,1)))
            return o
        
        if mode == 'encode':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            return quantized
        
        if mode == 'decode':
            o = self.decoder(x.permute((0,2,1)))
            return o
