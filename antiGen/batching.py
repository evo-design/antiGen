"""Length-bucketed batching, for training one model on several phylogenies at once.

``MutationDataset`` yields one ``(aa_idx (L,), mat (L, E))`` pair per example and relies
on torch's default collate, which ``torch.stack``s and therefore needs a single ``L`` per
batch. When several source trees — different proteins, different lengths — are trained
jointly, we keep both the model and the loss exact by never mixing lengths inside a batch:
bucket the indices by length, shuffle within each bucket, cut into batches, then shuffle
the batch *order* so consecutive optimizer steps still interleave sources.

The alternative, padding plus an attention mask, would change the numbers the existing
single-tree runs produce: the flattened ``L*E`` softmax in ``eval_utils.compute_probs``
would spread mass over pad cells, and ``ESM2FitnessPredictor._forward_impl`` branches on
the padded length, sending short rows down the sliding-window path. Bucketing avoids all
of that.
"""

from collections import defaultdict

import torch
from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler):
    """Yield batches of dataset indices that all share one sequence length.

    Args:
        lengths: sequence length of each dataset item, in dataset order.
        batch_size: maximum examples per batch.
        shuffle: shuffle within buckets and across batches (set False for eval).
        drop_last: drop a bucket's trailing partial batch.
        seed: base seed; the effective seed is ``seed + epoch``.
        max_batch_tokens: if > 0, a length-L bucket uses
            ``min(batch_size, max_batch_tokens // L)`` so the long bucket does not OOM at
            a batch size tuned for the short one.
    """

    def __init__(self, lengths, batch_size, shuffle=True, drop_last=False,
                 seed=0, max_batch_tokens=0):
        self.buckets = defaultdict(list)
        for i, length in enumerate(lengths):
            self.buckets[int(length)].append(i)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.max_batch_tokens = max_batch_tokens
        self.epoch = 0
        self._len = len(self._build(0))

    def _batch_size_for(self, length):
        if self.max_batch_tokens:
            return max(1, min(self.batch_size, self.max_batch_tokens // max(int(length), 1)))
        return self.batch_size

    def _build(self, epoch):
        generator = torch.Generator()
        generator.manual_seed(int(self.seed) + int(epoch))
        batches = []
        for length in sorted(self.buckets):
            indices = self.buckets[length]
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in perm]
            size = self._batch_size_for(length)
            for start in range(0, len(indices), size):
                batch = indices[start:start + size]
                if len(batch) == size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in perm]
        return batches

    def set_epoch(self, epoch):
        """Re-draw the ordering for this epoch. Without it every epoch replays one order."""
        self.epoch = epoch

    def __iter__(self):
        return iter(self._build(self.epoch))

    def __len__(self):
        return self._len

    def bucket_sizes(self):
        """{sequence_length: number of examples} — handy for logging."""
        return {length: len(idx) for length, idx in sorted(self.buckets.items())}
