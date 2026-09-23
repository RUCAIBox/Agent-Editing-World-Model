"""Reference Search folding: whole observations, with the latest two retained."""

from copy import deepcopy
from functools import lru_cache

FOLD_TEXT = "Content folded due to space limitation"


@lru_cache(maxsize=8)
def load_tokenizer(path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("EditAct Search requires pip install 'awe-agent[editact]'") from exc
    return AutoTokenizer.from_pretrained(path)


class SearchObservationCondenser:
    def __init__(self, maximum=32000, target=5000, tokenizer_path="Qwen/Qwen3.5-35B-A3B"):
        self.maximum = maximum
        self.target = target
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.tokenizer_path)
        return self._tokenizer

    def count_tokens(self, messages):
        # Match the reference budget estimator: count message content, not a
        # newly introduced chat-template/serialized-tool token budget.
        return sum(len(self.tokenizer.encode(str(m.content or ""))) for m in messages)

    async def condense(self, messages):
        result = deepcopy(messages)
        lengths = {
            i: len(self.tokenizer.encode(str(m.content or "")))
            for i, m in enumerate(messages)
            if m.role == "tool"
        }
        total = sum(lengths.values())
        if total <= self.maximum:
            return result
        mask_length = len(self.tokenizer.encode(FOLD_TEXT))
        masked = 0
        for i, length in lengths.items():
            if len(lengths) - masked <= 2:
                break
            if result[i].content == FOLD_TEXT:
                masked += 1
                continue
            saved = length - mask_length
            if saved > 0:
                result[i].content = FOLD_TEXT
                total -= saved
                masked += 1
            if total <= self.target:
                break
        return result
