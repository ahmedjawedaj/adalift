import torch


def pytest_sessionstart() -> None:
    torch.set_num_threads(2)
