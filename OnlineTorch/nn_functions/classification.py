from torch import nn


def build_torch_mlp_classifier(n_features):
    net = nn.Sequential(
        nn.Linear(n_features, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
        nn.Sigmoid()
    )
    return net
