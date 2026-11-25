"""Character-level tokenizer and random batches of training sequences.

Docs: docs/part1/data.md
"""

import numpy as np
import torch
from torch import Tensor


class CharTokenizer:
    """Map each distinct character of a text to an integer id.

    The vocabulary is the *sorted* list of distinct characters of the text
    given to the constructor: ``self.chars[i]`` is the character with id ``i``.
    """

    def __init__(self, text: str):
        self.chars = list(text)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Return the list of ids of the characters of ``text``."""
        
        set_training_test = set(self.chars)
        encoding = sorted(set_training_data)
        encoded_text = []
        N = len(text)
        for i in range(N) :
                c = text[i]
                
                try : 
                    encoded_text.append(encoding.index(c))
                
                except : 
                    raise KeyError('text contains {c} at index {i} which is not an element of the training data set. Sorry!'.format(c=c, i=i))
        return encoded_text

    def decode(self, ids: list[int]) -> str:
        """Inverse of :meth:`encode`."""
        
        set_training_test = set(self.chars)
        encoding = sorted(set_training_data)
        decoded_text = ""
        N = len(ids)

        for i in range(N):
            e = ids[i]
            try : 
                e_decoded = encoding[e]
                decoded_text = decoded_text + e_decoded

            except :
                raise KeyError("The list contains {e} at index {i} which is not encoded by the training data! Sorry!".format(e=e, i=i))

        return decoded_text
                

    # -- serialization (given) ----------------------------------------------
    def to_dict(self) -> dict:
        return {"type": "char", "chars": "".join(self.chars)}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["chars"])


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample ``batch_size`` random windows of the token sequence ``data``.

    Args:
        data: 1D integer array of token ids (possibly a ``np.memmap``).
        batch_size: number of sequences ``B``.
        context_length: length ``T`` of each sequence.
        device: device of the returned tensors.
        generator: optional ``torch.Generator`` used to draw the start
            positions (for reproducibility).

    Returns:
        ``(x, y)``, two ``torch.long`` tensors of shape ``(B, T)`` such that
        ``x[b] = data[s : s + T]`` and ``y[b] = data[s + 1 : s + T + 1]`` for a
        start position ``s`` drawn uniformly among all valid positions.
    """
    raise NotImplementedError("Exercise: get_batch (docs/part1/data.md)")
