import os
import logging
from torch import nn


def set_logging():
    # set logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
    # Disable c10d logging
    logging.getLogger('c10d').setLevel(logging.ERROR)


from lightning.pytorch.loggers import TensorBoardLogger


class Logger(TensorBoardLogger):
    def __init__(self, model, *args, **kwargs):
        super().__init__(".", name="lightning_logs", *args, **kwargs)
        self._log_graph = True
        self.model = model
        self.experiment.add_custom_scalars(self._layout())

    def _layout(self):
        activation_params = []
        discrepancy_params = []
        for i, item in enumerate(self.model.get_activations().items()):
            name, layer = item
            if isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
                # Adding activation stats layout
                # activation_params.append(f'act_{name}_out')
                activation_params.append(f'z_act_{name.replace(".", "_")}_out_sat')

        for name, p in self.model.named_parameters():
            if p.ndim == 2:
                # Assuming you only log update discrepancies for 2D params
                formatted_name = 'z_ud_' + name.replace('.', '_')
                discrepancy_params.append(formatted_name)

        layout = {
            "Layer Metrics": {
                "Activation Out": ["Multiline", activation_params],
                "Update Discrepancy by Layer": ["Multiline", discrepancy_params]
            }
        }
        return layout

    # logging
    def log_ud(self):
        # Get the current learning rate
        lr = self.model.optimizers().param_groups[0]['lr']

        # Iterate through named parameters to calculate and log metrics
        for name, p in self.model.named_parameters():
            if p.ndim == 2 and p.grad is not None:
                # Calculate the standard deviation of the gradients adjusted by the learning rate
                grad_std = (lr * p.grad).std()
                # Calculate the standard deviation of the parameter values
                param_std = p.data.std()
                # Calculate the Update Discrepancy (ud) metric and take the log10
                metric = (grad_std / param_std).log10().item()
                # Create a formatted name that corresponds to the naming convention in the TensorBoard layout
                formatted_name = 'z_ud_' + name.replace('.', '_')
                # Log the metric using the formatted name
                self.model.log(formatted_name, metric, on_step=False, on_epoch=True, sync_dist=True)

    def log_activation_out(self):
        for i, item in enumerate(self.model.get_activations().items()):
            name, layer = item
            # if isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            #     t = layer.out  # Make sure outputs are stored during forward pass
            #     self.log(f'act_{name}_out', t.mean().item(), on_step=False, on_epoch=True)
            if isinstance(layer, nn.ReLU):
                t = layer.out  # Make sure outputs are stored during forward pass
                saturation = (t < 0.05).float().mean() * 100
                self.model.log(f'z_act_{name.replace(".", "_")}_out_sat', saturation, on_step=False, on_epoch=True, sync_dist=True)
            if isinstance(layer, nn.Tanh):
                t = layer.out  # Make sure outputs are stored during forward pass
                saturation = (t.abs() > 0.025).float().mean() * 100
                self.model.log(f'z_act_{name.replace(".", "_")}_out_sat', saturation, on_step=False, on_epoch=True, sync_dist=True)
