# Defines some util functions
import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

class ContrastiveBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, pos_window, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.pos_window = pos_window
        self.drop_last = drop_last
        self.n = len(self.data_source)

    def __iter__(self):
        for i in range(self.n // self.batch_size):
            x = torch.randint(low=0, high=self.n-1, size=(self.batch_size//2,),
                              dtype=torch.int64)
            y = x + torch.randint(low=-self.pos_window, high=self.pos_window, size=(self.batch_size//2,),
                                  dtype=torch.int64)
            y = torch.clamp(y, 0, self.n-1)
            z = x.tolist() + y.tolist()

            yield z

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        else:
            return (self.n + self.batch_size - 1) // self.batch_size