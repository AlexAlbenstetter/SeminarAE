import collections
import copy
import typing
import torch
from river import base

from IncrementalDL.OnlineTorch.base import PyTorch2RiverBase, RollingPyTorch2RiverBase

class PyTorch2RiverClassifier(PyTorch2RiverBase, base.Classifier):

    def __init__(self,
                 build_fn,
                 loss_fn: torch.nn.modules.loss._Loss,
                 optimizer_fn: typing.Type[torch.optim.Optimizer],
                 learning_rate=1e-3,
                 **net_params,
                 ):
        self.classes = collections.Counter()
        self.n_classes = 1
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            learning_rate=learning_rate,
            **net_params
        )

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Classifier:
        self.classes.update([y])

        # check if model is initialized
        if self.net is None:
            self._init_net(len(list(x.values())))

        # check last layer
        if len(self.classes) != self.n_classes:
            self.n_classes = len(self.classes)
            layers = list(self.net.children())
            i = -1
            layer_to_convert = layers[i]
            while not hasattr(layer_to_convert, 'weight'):
                layer_to_convert = layers[i]
                i -= 1

            removed = list(self.net.children())[:i + 1]
            new_net = removed
            new_layer = torch.nn.Linear(in_features=layer_to_convert.in_features,
                                        out_features=self.n_classes)
            # copy the original weights back
            with torch.no_grad():
                new_layer.weight[:-1, :] = layer_to_convert.weight
                new_layer.weight[-1:, :] = torch.mean(layer_to_convert.weight, 0)
            new_net.append(new_layer)
            if i + 1 < -1:
                for layer in layers[i + 2:]:
                    new_net.append(layer)
            self.net = torch.nn.Sequential(*new_net)
            self.optimizer = self.optimizer_fn(self.net.parameters(), self.learning_rate)

        # training process
        proba = {c: 0.0 for c in self.classes}
        proba[y] = 1.0
        x = list(x.values())
        y = list(proba.values())

        x = torch.Tensor([x])
        y = torch.Tensor([y])
        self._learn_one(x=x, y=y)
        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if self.net is None:
            self._init_net(len(list(x.values())))
        x = torch.Tensor(list(x.values()))
        yp = self.net(x).detach().numpy()
        proba = {c: 0.0 for c in self.classes}
        for idx, val in enumerate(self.classes):
            proba[val] = yp[idx]
        return proba


class RollingPyTorch2RiverClassifer(RollingPyTorch2RiverBase, base.Classifier):
    def __init__(self,
                 build_fn,
                 loss_fn: torch.nn.modules.loss._Loss,
                 optimizer_fn: typing.Type[torch.optim.Optimizer],
                 window_size=1,
                 learning_rate=1e-3,
                 **net_params,
                 ):
        self.classes = collections.Counter()
        self.n_classes = 1
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            window_size=window_size,
            learning_rate=learning_rate,
            **net_params
        )

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Classifier:
        self.classes.update([y])

        # check if model is initialized
        if self.net is None:
            self._init_net(len(list(x.values())))

        # check last layer
        if len(self.classes) != self.n_classes:
            self.n_classes = len(self.classes)
            layers = list(self.net.children())
            i = -1
            layer_to_convert = layers[i]
            while not hasattr(layer_to_convert, 'weight'):
                layer_to_convert = layers[i]
                i -= 1

            removed = list(self.net.children())[:i + 1]
            new_net = removed
            new_layer = torch.nn.Linear(in_features=layer_to_convert.in_features,
                                        out_features=self.n_classes)
            # copy the original weights back
            with torch.no_grad():
                new_layer.weight[:-1, :] = layer_to_convert.weight
                new_layer.weight[-1:, :] = torch.mean(layer_to_convert.weight, 0)
            new_net.append(new_layer)
            if i + 1 < -1:
                for layer in layers[i + 2:]:
                    new_net.append(layer)
            self.net = torch.nn.Sequential(*new_net)
            self.optimizer = self.optimizer_fn(self.net.parameters(), self.learning_rate)

        # training process
        self._x_window.append(list(x.values()))
        proba = {c: 0.0 for c in self.classes}
        proba[y] = 1.0
        y = list(proba.values())

        if len(self._x_window) == self.window_size:
            x = torch.Tensor([self._x_window.values])
            [y.append(0.0) for i in range(self.n_classes - len(y))]
            y = torch.Tensor([y])
            self._learn_batch(x=x, y=y)
        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if self.net is None:
            self._init_net(len(list(x.values())))
        if len(self._x_window) == self.window_size:
            l = copy.deepcopy(self._x_window.values)
            l.append(list(x.values()))
            x = torch.Tensor([l])
            yp = self.net(x).detach().numpy()
            proba = {c: 0.0 for c in self.classes}
            for idx, val in enumerate(self.classes):
                proba[val] = yp[0][idx]
        else:
            proba = {c: 0.0 for c in self.classes}
        return proba