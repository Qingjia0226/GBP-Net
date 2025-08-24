from __future__ import print_function, division
import numpy as np
import torch

__all__ = [
    'collate_fn_train',
    'collate_fn_test']

####################################################################
# Collate Functions
####################################################################


def collate_fn_train(batch):
    return TrainBatch(batch)


def collate_fn_test(batch):
    return TestBatch(batch)

####################################################################
# Custom Batch Class
####################################################################


class TrainBatch:
    def __init__(self, batch):
        self._handle_batch(*zip(*batch))

    def _handle_batch(self, pos, out_input, out_target_l,out_target_s, out_weight,out_volume):
        self.pos = pos
        self.out_input = torch.from_numpy(np.stack(out_input, 0))
        self.out_volume = torch.from_numpy(np.stack(out_volume, 0))

        out_target_ll = [None]*len(out_target_l[0])
        ow = [[None]*len(out_weight[0][x]) for x in range(len(out_weight[0]))]

        for i in range(len(out_target_l[0])):
            out_target_ll[i] = np.stack([out_target_l[x][i]
                                        for x in range(len(out_target_l))], 0)
            out_target_ll[i] = torch.from_numpy(out_target_ll[i])


        out_target_ss = [None] * len(out_target_s[0])


        for i in range(len(out_target_s[0])):
            out_target_ss[i] = np.stack([out_target_s[x][i]
                                        for x in range(len(out_target_s))], 0)
            out_target_ss[i] = torch.from_numpy(out_target_ss[i])



        # each target can have multiple loss/weights
        for i in range(len(out_weight[0])):
            for j in range(len(out_weight[0][i])):
                ow [i][j] = np.stack(
                    [out_weight[x][i][j] for x in range(len(out_weight))], 0)
                ow [i][j] = torch.from_numpy(ow [i][j])

        self.out_target_l = out_target_ll
        self.out_target_s = out_target_ss
        self.out_weight = ow 

    # custom memory pinning method on custom type
    def pin_memory(self):
        self._pin_batch()
        return self

    def _pin_batch(self):
        self.out_input = self.out_input.pin_memory()
        self.out_volume = self.out_volume.pin_memory()
        for i in range(len(self.out_target_l)):
            self.out_target_l[i] = self.out_target_l[i].pin_memory()
        for i in range(len(self.out_target_s)):
            self.out_target_s[i] = self.out_target_s[i].pin_memory()

        for i in range(len(self.out_weight)):
            for j in range(len(self.out_weight[i])):
                self.out_weight[i][j] = self.out_weight[i][j].pin_memory()


class TrainBatchRecon(TrainBatch):
    def _handle_batch(self, pos, out_input, out_target, out_weight, out_recon):
        super()._handle_batch(pos, out_input, out_target, out_weight)
        self.out_recon = torch.from_numpy(np.stack(out_recon, 0))

    # custom memory pinning method on custom type
    def pin_memory(self):
        self._pin_batch()
        self.out_recon = self.out_recon.pin_memory()
        return self


class TrainBatchReconOnly:
    def __init__(self, batch):
        self._handle_batch(*zip(*batch))

    def _handle_batch(self, pos, out_input, out_recon):
        self.pos = pos
        self.out_input = torch.from_numpy(np.stack(out_input, 0))
        self.out_recon = torch.from_numpy(np.stack(out_recon, 0))

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.out_input = self.out_input.pin_memory()
        self.out_recon = self.out_recon.pin_memory()
        return self


class TestBatch:
    def __init__(self, batch):
        pos, out_input, out_volume = zip(*batch)
        self.pos = pos
        self.out_input = torch.from_numpy(np.stack(out_input, 0))
        self.out_volume = torch.from_numpy(np.stack(out_volume, 0))

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.out_input = self.out_input.pin_memory()
        self.out_volume = self.out_volume.pin_memory()
        return self
