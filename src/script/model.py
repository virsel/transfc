import lightning as L

import torch.nn as nn
import torch
import torch.nn.functional as F

from config import Config, HyperParams


def get_model(params: HyperParams):
    return TransfEncModel(params)

def get_trainable_layers(parent_module, parent_name='root'):
    res = {}
    modules = list(parent_module.named_children())
    if len(modules) == 0 and any(param.requires_grad for param in parent_module.parameters()):
        # Base case: If no children and has trainable params, return itself
        res[parent_name] = parent_module
    else:
        for name, module in modules:
            # Construct the full module name by appending current name to parent's
            full_name = f'{parent_name}.{name}' if parent_name else name
            # Recursively get trainable layers
            sub_layers = get_trainable_layers(module, full_name)
            if len(sub_layers) == 0 and any(param.requires_grad for param in module.parameters()):
                # If the module has parameters but no sub-layers returned, add the module
                res[full_name] = module
            res.update(sub_layers)
    return res

def get_activations(parent_module, parent_name='root'):
    res = {}
    modules = list(parent_module.named_children())
    if len(modules) == 0 and isinstance(parent_module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
        # Base case: If no children and has trainable params, return itself
        res[parent_name] = parent_module
    else:
        for name, module in modules:
            # Construct the full module name by appending current name to parent's
            full_name = f'{parent_name}.{name}' if parent_name else name
            # Recursively get trainable layers
            sub_layers = get_activations(module, full_name)
            if len(sub_layers) == 0 and isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
                # If the module has parameters but no sub-layers returned, add the module
                res[full_name] = module
            res.update(sub_layers)
    return res

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, params: HyperParams):
        super().__init__()
        self.key = nn.Linear(params.n_embd, params.n_embd // params.n_head, bias=False)
        self.query = nn.Linear(params.n_embd, params.n_embd // params.n_head, bias=False)
        self.value = nn.Linear(params.n_embd, params.n_embd // params.n_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(params.context_length, params.context_length)))
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, params: HyperParams):
        super().__init__()
        self.heads = nn.ModuleList([Head(params) for _ in range(params.n_head)])
        self.proj = nn.Linear(params.n_embd, params.n_embd)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, params: HyperParams):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(params)
        self.ffwd = FeedFoward(params.n_embd)
        self.ln1 = nn.LayerNorm(params.n_embd)
        self.ln2 = nn.LayerNorm(params.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransfEncModel(L.LightningModule):
    def __init__(self, params: HyperParams, n_classes=5):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.lossi = []

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(params.vocab_size, params.n_embd)
        self.position_embedding_table = nn.Embedding(params.context_length, params.n_embd)
        self.blocks = nn.Sequential(*[Block(params) for _ in range(1)])
        self.ln_f = nn.LayerNorm(params.n_embd)  # final layer norm
        self.lm_head = nn.Linear(params.context_length * params.n_embd, n_classes)

        # register forward hook
        self._register_forward_hooks()

    def _register_forward_hooks(self):
        # Define the forward hook function
        activations = self.get_activations().values()
        for act in activations:
            act.register_forward_hook(forward_hook)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        B, T, C = x.shape
        x = x.view(B, T * C)  # (B,T,C) -> (B, T*C)
        logits = self.lm_head(x)  # (B,n_classes)

        return logits

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        opt = self.optimizers()
        opt.zero_grad()
        output = self(inputs)
        loss = torch.nn.functional.cross_entropy(output, target.view(-1))
        # Call backward with retain_graph=True
        self.manual_backward(loss, retain_graph=True)
        # Ensure that logger has the log_ud method
        self.training_step_log()
        opt.step()
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def training_step_log(self):
        if hasattr(self.logger, 'log_ud'):
            self.logger.log_ud()
        if hasattr(self.logger, 'log_activation_out'):
            self.logger.log_activation_out()

    def on_fit_end(self) -> None:
        return super().on_fit_end()

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.cross_entropy(output, target.view(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def get_trainable_layers(self):
        return get_trainable_layers(self)

    def get_activations(self):
        return get_activations(self)

def forward_hook(module, input, output):
    module.out = output  # Store output in the module itself
    module.out.retain_grad()  # Ensure that the output gradients are stored