"""dataset.py"""
__all__ = ["GRPOIteratorStore"]


class GRPOIteratorStore:
    """ iterator for storing data """
    def __init__(self, store):
        self._index = 0
        self.length = len(store)
        self.store = store

    def __next__(self):
        if self._index >= self.length:
            raise StopIteration
        item = (
            self.store[self._index].prompt_completion_ids,
            self.store[self._index].responses_mask,
            self.store[self._index].ref_per_token_logps,
            self.store[self._index].advantages,
            self.store[self._index].actual_sequence_length,
            self.store[self._index].sample_index,
            self.store[self._index].sample_valid_length,
            self.store[self._index].old_per_token_logps
        )
        self._index += 1
        return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return self.length
