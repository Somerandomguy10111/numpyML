from typing import Callable, Optional, Iterable
import torch
from torch import device, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from .descent_stategy import Descent, Adam
from .monitor import Report, Monitor

# ---------------------------------------------------------

LossFunction = Callable[[Tensor, Tensor], Tensor]


class Trainer:
    def __init__(self, descent : Descent = Adam(),
                 torch_device: device = device("cuda" if torch.cuda.is_available() else "cpu"),
                 monitor : Optional[Monitor] = None):
        self.descent : Descent = descent
        self.device: torch.device = torch_device
        self.monitor : Monitor = monitor


    def train(self, model : Module, loss_fn : LossFunction, dataset : Dataset, batch_size : int = 64) -> Report:
        def calculate_loss(output, target) -> Tensor:
            output, target = output.to(self.device), target.to(self.device)
            loss = loss_fn(output, target)
            report.state_loss_map[i] = loss.item()
            return loss

        def backpropagate(loss : Tensor):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model = self.convert_to_training(model=model)
        optimizer = self.get_optimizer(params=model.parameters())
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        report = Report(seed=torch.seed())

        for (i, (x, y)) in enumerate(data_loader):
            current_loss = calculate_loss(output=model(x),target=y)
            backpropagate(current_loss)
            print(f'Loss is currently {current_loss.item()} at iteration {i}')
        return report


    def convert_to_training(self, model : Module) -> Module:
        model = model.to(device=self.device)
        model.train()
        print(f'Note: Model mode was automatically set to train and moved to device \"{self.device}\"')
        return model

    def get_optimizer(self, params : Iterable[Tensor]):
        return self.descent.get_optimizer(params=params)
