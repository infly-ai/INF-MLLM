#coding=utf-8

import math
import torch 
import torch.nn as nn


class Pooler(nn.Module):
    def __init__(self, dim_in, dim_out, pool_out_size):
        super().__init__()
        self.pool_h, self.pool_w = pool_out_size, pool_out_size

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )
                
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, t, f, v, d = x.shape
        s = int(math.sqrt(v -1))
        assert t == 1 and f == 1 
        x = x[:, :, :, 1:, :]           # remove cls_token
        x = x.reshape(b, t, f, s, s, d)

        if s % self.pool_h == 0 and s % self.pool_w == 0:
            x = x.reshape(b, t, f, self.pool_h, s//self.pool_h, self.pool_w, s//self.pool_w, d)
            x = x.permute([0, 1, 2, 3, 5, 7, 4, 6]).reshape(b, t, f, self.pool_h * self.pool_w, d, -1).mean(-1)
            x = self.mlp(x)                 # [b, t, f, h*w, d]
            x = x.flatten(0, 2)
        #else:
        #    x = x.flatten(0, 2).permute(0, 3, 1, 2)
        #    x = torch.nn.functional.adaptive_avg_pool2d(x, (self.pool_h, self.pool_w))
        #    x = x.permute(0, 2, 3, 1).flatten(1, 2)
        #    x = self.mlp(x)                 # [b, t, f, h*w, d]
        else:
            raise ValueError()

        return x.unsqueeze(1)
