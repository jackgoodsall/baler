# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.nn import functional as F


import torch.utils.data
from torch.nn import functional as F
from torch.autograd import Function
from ..modules import helper


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """

    def __init__(
        self,
        ch,
        inverse=False,
        beta_min=1e-6,
        gamma_init=0.1,
        reparam_offset=2**-18,
    ):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.device = helper.get_device()
        self.reparam_offset = torch.tensor([reparam_offset], device=self.device)

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2) ** 0.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch, device=self.device) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch, device=self.device)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class AE(nn.Module):
    # This class is a modified version of the original class by George Dialektakis found at
    # https://github.com/Autoencoders-compression-anomaly/Deep-Autoencoders-Data-Compression-GSoC-2021
    # Released under the Apache License 2.0 found at https://www.apache.org/licenses/LICENSE-2.0.txt
    # Copyright 2021 George Dialektakis

    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(AE, self).__init__(*args, **kwargs)

        self.activations = {}

        # encoder
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, z_dim)
        # decoder
        self.de1 = nn.Linear(z_dim, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    # Implementation of activation extraction using the forward_hook method

    def get_hook(self, layer_name):
        def hook(model, input, output):
            self.activations[layer_name] = output.detach()

        return hook

    def get_layers(self) -> list:
        return [self.en1, self.en2, self.en3, self.de1, self.de2, self.de3]

    def store_hooks(self) -> list:
        layers = self.get_layers()
        hooks = []
        for i in range(len(layers)):
            hooks.append(layers[i].register_forward_hook(self.get_hook(str(i))))
        return hooks

    def get_activations(self) -> dict:
        for kk in self.activations:
            self.activations[kk] = F.leaky_relu(self.activations[kk])
        return self.activations

    def detach_hooks(self, hooks: list) -> None:
        for hook in hooks:
            hook.remove()


class BIGGER_AE(nn.Module):
    """Moddified Version of AE for compatability with 
    higher dimensional spaces"""

    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(BIGGER_AE, self).__init__(*args, **kwargs)

        self.activations = {}

        # encoder
        self.en1 = nn.Linear(n_features, 2048)
        self.en2 = nn.Linear(2048, 512)
        self.en3 = nn.Linear(512, 256)
        self.en4 = nn.Linear(256, z_dim)
        # decoder
        self.de1 = nn.Linear(z_dim, 256)
        self.de2 = nn.Linear(256, 512)
        self.de3 = nn.Linear(512, 2048)
        self.de4 = nn.Linear(2048, n_features)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return F.leaky_relu(self.en4(h3))

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class BIGGER_AE_Dropout_BN(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(BIGGER_AE_Dropout_BN, self).__init__(*args, **kwargs)

        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 2048, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048,dtype=torch.float64),
            nn.Linear(2048, 512, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512,dtype=torch.float64),
            nn.Linear(512, 256, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256,dtype=torch.float64),
            nn.Linear(256, z_dim, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim,dtype=torch.float64)
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 256, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256, dtype=torch.float64),
            nn.Linear(256, 512, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512, dtype=torch.float64),
            nn.Linear(512, 2048, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048, dtype=torch.float64),
            nn.Linear(2048, n_features, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(n_features, dtype=torch.float64),
            nn.LeakyReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class TransformerAE_two(nn.Module):
    """Transformer Autoencoder with a final linear output layer."""

    def __init__(
        self,
        n_features,
        z_dim,
        encoder_h_dim: list = [512, 256, 128],
        decoder_h_dim: list = [128, 256, 512],
        nheads=2,
        latent_dim=40,
        activation=torch.nn.functional.gelu,
    ):
        super(TransformerAE_two, self).__init__()
        in_dim = n_features
        out_dim = n_features    
        self.in_dim = n_features
        self.out_dim = n_features
        self.latent_dim = latent_dim

        self.encoder_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                batch_first=True,
                norm_first=True,
                d_model=i,
                activation=activation,
                dim_feedforward=i,
                nhead=nheads,
            )
            for i in ([in_dim] + encoder_h_dim[:] + [latent_dim])
        ])

        self.encoder_linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(i[0]),
                nn.Linear(i[0], i[-1]),
                nn.GELU(),
            )
            for i in zip([in_dim] + encoder_h_dim, encoder_h_dim + [latent_dim])
        ])

        self.decoder_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                batch_first=True,
                norm_first=True,
                d_model=i,
                activation=activation,
                dim_feedforward=i,
                nhead=nheads,
            )
            for i in ([latent_dim] + decoder_h_dim)
        ])

        self.decoder_linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(i[0]),
                nn.Linear(i[0], i[-1]),
                nn.GELU(),
            )
            for i in zip([latent_dim] + decoder_h_dim[:-1], decoder_h_dim)
        ])

  
        self.final_output_layer = nn.Linear(decoder_h_dim[-1], out_dim)

    def encode(self, x: torch.Tensor):
        for i in range(len(self.encoder_linear_layers)):
            x = self.encoder_transformer_layers[i](x)
            x = self.encoder_linear_layers[i](x)
        x = self.encoder_transformer_layers[-1](x)
        return x

    def decode(self, x: torch.Tensor):
        for i in range(len(self.decoder_linear_layers)):
            x = self.decoder_transformer_layers[i](x)
            x = self.decoder_linear_layers[i](x)
        x = self.decoder_transformer_layers[-1](x)
        x = self.final_output_layer(x)  
        return x

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x = self.decode(z)
        return x


class ResidualConnectionAE(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(ResidualConnectionAE, self).__init__()

        self.n_features = n_features
        self.z_dim = z_dim

        self.e1 = nn.Sequential(
            nn.Linear(n_features, 2048),
            nn.LeakyReLU(),
        )
        self.e2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
        )
        self.e3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
        )
        self.e4 =  nn.Sequential(
            nn.Linear(256, z_dim),
            nn.LeakyReLU(),
        )

        self.d1 = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(),
        )
        self.d2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
        )
        self.d3 = nn.Sequential(
            nn.Linear(512, 2048),
            nn.LeakyReLU(),
        )
        self.d4 = nn.Sequential(
            nn.Linear(2048, n_features)
        )

    def encode(self, x):
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        return x

    def decode(self, z: torch.Tensor) ->  torch.Tensor:
        z = self.d1(z)
        z = self.d2(z)
        z = self.d3(z)
        z = self.d4(z)
        return z

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)

        z1  = x3 + self.d1(x4)
        z2 = x2 + self.d2(z1)
        z3 = x1 + self.d3(z2)
        z4 = self.d4(z3)

        return z4