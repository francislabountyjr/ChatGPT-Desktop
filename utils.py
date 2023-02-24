import re
import torch


def sluggify(string):
    """Converts a string to a slug."""
    string = re.sub(r'[^-\w\s./]', '',
                    string).strip().lower()
    string = re.sub(r'[.]+', '-', string)
    return re.sub(r'[-\s]+', '-', string)


def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
