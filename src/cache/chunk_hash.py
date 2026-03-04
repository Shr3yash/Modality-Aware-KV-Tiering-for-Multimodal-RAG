import hashlib
from typing import List


def compute_chunk_hash(token_ids: List[int], modality: str = "text") -> str:
    """Deterministic hash of chunk content based on modality and token ids."""
    payload = f"{modality}:{','.join(map(str, token_ids))}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]

