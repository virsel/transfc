import lightning as L

import torch.nn as nn
import torch

from config import Config


def get_model(cfg):
    model = WaveModel(cfg)
    layers = list(model.modules())[1:]
    for layer in layers[:-1]:
        layer.register_forward_hook(forward_hook)
        # print(layer)
    return  model
    
# Define the forward hook function
def forward_hook(module, input, output):
    module.out = output  # Store output in the module itself

class FlattenConsecutive(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        # forward to torch batchnorm, expects:
        # Input: (N,C) or (N,C,L), where N is the batch size,
        # C is the number of features or channels, and L is the sequence length
        # output: (N, C, L)

        N, L, C = x.shape
        try:
            x = x.view(N, L // self.n, C * self.n)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {N, L//self.n, C*self.n}")
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out


class BatchNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def forward(self, x, target=None):
        # calculate the forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)  # batch mean
            xvar = x.var(dim, keepdim=True)  # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar
        return self.out

    def _parameters(self):
        return [self.gamma, self.beta]


class WaveModel(L.LightningModule):
    def __init__(self, cfg: Config, n_classes=5):
        super().__init__()

        # Define the layers
        self.n_hidden = cfg.n_hidden
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.embedding.weight.data = (
            self.embedding.weight.data / (cfg.n_embd * cfg.n_consecutive) ** 0.5
        )
        self.flatten1 = FlattenConsecutive(cfg.n_consecutive)
        self.linear1 = nn.Linear(cfg.n_embd * cfg.n_consecutive, cfg.n_hidden, bias=False)
        self.linear1.weight.data = (
            self.linear1.weight.data * 5 / 3 / (cfg.n_hidden * cfg.n_consecutive) ** 0.5
        )
        self.batch_norm1 = BatchNorm1d(cfg.n_hidden)
        self.tanh1 = nn.Tanh()

        self.flatten2 = FlattenConsecutive(cfg.n_consecutive)
        self.linear2 = nn.Linear(cfg.n_hidden * cfg.n_consecutive, cfg.n_hidden, bias=False)
        self.linear2.weight.data = (
            self.linear2.weight.data * 5 / 3 / (cfg.n_hidden * cfg.n_consecutive) ** 0.5
        )
        self.batch_norm2 = BatchNorm1d(cfg.n_hidden)
        self.tanh2 = nn.Tanh()

        n_consecutive = 2
        self.flatten3 = FlattenConsecutive(n_consecutive)
        self.linear3 = nn.Linear(cfg.n_hidden * n_consecutive, cfg.n_hidden, bias=False)
        self.linear3.weight.data = self.linear3.weight.data * 5 / 3 / (cfg.n_hidden) ** 0.5
        self.batch_norm3 = BatchNorm1d(cfg.n_hidden)
        self.tanh3 = nn.Tanh()

        self.output_linear = nn.Linear(cfg.n_hidden, n_classes)
        self.output_linear.weight.data = self.output_linear.weight.data * 0.1

    def forward(self, x):
        # Step 1: Embedding layer
        x = self.embedding(x)

        # Step 2: First flattening, linear, batch normalization, and activation layers
        x = self.flatten1(x)
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.tanh1(x)

        # Step 3: Second flattening, linear, batch normalization, and activation layers
        x = self.flatten2(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = self.tanh2(x)

        # Step 4: Third flattening, linear, batch normalization, and activation layers
        x = self.flatten3(x)
        x = self.linear3(x)
        x = self.batch_norm3(x)
        x = self.tanh3(x)

        # Step 5: Output linear layer
        x = self.output_linear(x)

        return x

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.cross_entropy(output, target.view(-1))
        # lossi.append(loss.log10().item())
        # with torch.no_grad():
        # lr = optimizer.param_groups[0]['lr']
        # ud.append([((lr*p.grad).std() / p.output.std()).log10().item() for p in model.parameters()])
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.cross_entropy(output, target.view(-1))
        self.log("val_loss", loss, on_step=True, on_epoch=True,prog_bar=True, sync_dist=True)
        return loss
