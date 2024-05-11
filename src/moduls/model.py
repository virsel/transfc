import math

from .custom_logging import Logger

import lightning as L

import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

from .config import HyperParams
from .utils import act_possible
import torchmetrics


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
    if len(modules) == 0 and isinstance(parent_module, act_possible):
        # Base case: If no children and has trainable params, return itself
        res[parent_name] = parent_module
    else:
        for name, module in modules:
            # Construct the full module name by appending current name to parent's
            full_name = f'{parent_name}.{name}' if parent_name else name
            # Recursively get trainable layers
            sub_layers = get_activations(module, full_name)
            if len(sub_layers) == 0 and isinstance(module, act_possible):
                # If the module has parameters but no sub-layers returned, add the module
                res[full_name] = module
            res.update(sub_layers)
    return res

def get_elements(parent_module, parent_name='root'):
    res = {}
    modules = list(parent_module.named_children())
    if len(modules) == 0:
        # Base case: If no children and has trainable params, return itself
        res[parent_name] = parent_module
    else:
        for name, module in modules:
            # Construct the full module name by appending current name to parent's
            full_name = f'{parent_name}.{name}' if parent_name else name
            # Recursively get trainable layers
            sub_layers = get_elements(module, full_name)
            if len(sub_layers) == 0:
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
        # self.register_buffer('tril', torch.tril(torch.ones(params.context_length, params.context_length)))
        self.dropout = nn.Dropout(params.dropout)
    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     # Using Glorot uniform initialization
    #     nn.init.xavier_uniform_(self.query, gain=nn. init. calculate_gain('relu'))
    #     nn.init.xavier_uniform_(self.key, gain=nn. init. calculate_gain('relu'))
    #     nn.init.xavier_uniform_(self.value, gain=nn. init. calculate_gain('relu'))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
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
    def __init__(self, params: HyperParams):
        super().__init__()
        # Define individual layers
        self.l1 = nn.Linear(params.n_embd, params.n_hidden)
        self.batchnorm1 = nn.BatchNorm1d(params.context_length)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(params.dropout)
        self.l2 = nn.Linear(params.n_hidden, params.n_embd)
        self.batchnorm2 = nn.BatchNorm1d(params.context_length)
        self.elu2 = nn.ELU()

    def forward(self, x):
        # Manually apply each layer
        x = self.l1(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)
        x = self.elu1(x)
        x = self.l2(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)
        return x

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, params: HyperParams):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.lnorm1 = nn.LayerNorm(params.n_embd)
        self.dropout1 = nn.Dropout(p=params.dropout)
        self.sa = MultiHeadAttention(params)
        self.lnorm2 = nn.LayerNorm(params.n_embd)
        self.ffwd = FeedFoward(params)
        self.dropout2 = nn.Dropout(p=params.dropout)

    def forward(self, x):
        # self-attention
        _x = x
        x = self.sa(x)
        x = self.dropout1(x)
        x = self.lnorm1(_x + x)

        # feedforward
        _x = x
        x = self.ffwd(x)
        x = self.dropout2(x)
        x = self.lnorm2(_x + x)
        return x

class TransfEncModel(L.LightningModule):
    def __init__(self, params: HyperParams, n_classes=5):
        super().__init__()
        self.save_hyperparameters()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.lossi = []
        self.custom_logger: Logger = None
        self.params = params

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(params.vocab_size, params.n_embd)
        self.token_embedding_table.weight = nn.Parameter(self.token_embedding_table.weight // (params.n_embd ** 0.5), requires_grad=True)

        # Create sinusoidal positional embedding
        self.position_embedding_table = self.create_sinusoidal_positional_embedding(params.context_length, params.n_embd)

        self.blocks = nn.Sequential(*[Block(params) for _ in range(self.params.n_blocks)])
        self.lnorm_f = nn.LayerNorm(params.n_embd)  # final layer norm
        self.l_out = nn.Linear(params.n_embd, n_classes)
        self.l_out.weight = nn.Parameter(self.l_out.weight * 0.1, requires_grad=True)

        # register forward hook
        self._register_forward_hooks()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=n_classes, average='macro')
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=n_classes, average='macro')
        self.predictions = []
        self.labels = []

    def _register_forward_hooks(self):
        # Define the forward hook function
        activations = self.get_activations().values()
        for act in activations:
            act.register_forward_hook(forward_hook)
        for l in self.get_trainable_layers().values():
            l.register_forward_hook(forward_hook)

    def create_sinusoidal_positional_embedding(self, seq_len, embd, n = 10000):
        """Create sinusoidal positional embedding matrix as per Vaswani et al."""

        if embd % 2 != 0:
            raise ValueError(
                "Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(
                    embd))

        T = seq_len
        d = embd  # d_model=head_num*d_k, not d_q, d_k, d_v

        positions = torch.arange(0, T).unsqueeze_(1)
        embeddings = torch.zeros(T, d)

        denominators = torch.pow(n, 2 * torch.arange(0, d // 2) / d)  # 10000^(2i/d_model), i is the index of embedding
        embeddings[:, 0::2] = torch.sin(positions / denominators)  # sin(pos/10000^(2i/d_model))
        embeddings[:, 1::2] = torch.cos(positions / denominators)  # cos(pos/10000^(2i/d_model))

        return nn.Parameter(embeddings, requires_grad=False)

    def set_custom_logger(self, logger: Logger):
        self.custom_logger = logger

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.params.lr, weight_decay=0.01)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table  # (1, T, C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.lnorm_f(x)  # (B,T,C)
        # B, T, C = x.shape
        # x = x.view(B, T * C)  # (B,T,C) -> (B, T*C)
        logits = self.l_out(x)  # (B,n_classes)
        logits = logits[:, -1, :]  # becomes (B, C)

        return logits

    def set_val_data_loader(self, val_dataloader):
        self.val_dataloader = val_dataloader

    def on_train_start(self) -> None:
        self.custom_logger.log_model_arch()
        self.custom_logger.log_params()
        # Compute validation loss before any weights are updated
        if self.val_dataloader:
            val_loss = self.compute_val_loss()  # You need to implement this method
            self.logger.experiment.add_scalar("val_loss", val_loss, self.global_step)

    def compute_val_loss(self):
        # Function to compute validation loss before any weights are updated
        val_loss = 0.0
        num_batches = 0
        for batch in self.val_dataloader:
            inputs, target = batch
            output = self(inputs)
            loss = torch.nn.functional.cross_entropy(output, target.view(-1))
            val_loss += loss.item()
            num_batches += 1
        val_loss /= num_batches
        return val_loss

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



    def on_train_epoch_end(self) -> None:
        if self.custom_logger:
            self.custom_logger.log_out_on_epoch()

    def training_step_log(self):
        if self.custom_logger:
            self.custom_logger.log_ud()
            self.custom_logger.log_activation_out_sat()

    def on_fit_end(self) -> None:
        return super().on_fit_end()

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.cross_entropy(output, target.view(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.predictions.extend(preds.cpu().numpy())
        self.labels.extend(y.cpu().numpy())

        # Update metrics
        self.accuracy.update(preds, y)
        self.precision.update(preds, y)
        self.recall.update(preds, y)
        return {'test_loss': F.cross_entropy(logits, y)}

    def on_test_epoch_end(self):
        # Log metrics to TensorBoard
        self.log('test_accuracy', self.accuracy.compute(), on_epoch=True)
        self.log('test_precision', self.precision.compute(), on_epoch=True)
        self.log('test_recall', self.recall.compute(), on_epoch=True)

        # Classification report
        report = classification_report(self.labels, self.predictions, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):  # Skip the summary metrics
                for metric_name, score in metrics.items():
                    self.log(f'test_{label}_{metric_name}', score)


    def get_trainable_layers(self):
        return get_trainable_layers(self)

    def get_activations(self):
        return get_activations(self)

    def get_elements(self):
        return get_elements(self)

def forward_hook(module, input, output):
    if module.training:
        module.out = output # Store output in the module itself
        module.out.retain_grad()  # Ensure that the output gradients are stored