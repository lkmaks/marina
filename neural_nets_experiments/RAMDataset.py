from torch.utils.data import DataLoader, Dataset


def todevice(obj, device):
    if isinstance(obj, int) or isinstance(obj, float):
        return obj
    elif isinstance(obj, tuple):
        return tuple(todevice(x, device) for x in obj)
    else:
        return obj.to(device)

class RAMDataset(Dataset):
    def __init__(self, dataset, device='cpu'):
        self.dataset = dataset
        self.len = len(self.dataset)
        self.elems = [todevice(dataset[i], device) for i in range(self.len)]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.elems[item]
