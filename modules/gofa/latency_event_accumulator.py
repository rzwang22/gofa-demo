DETAIL_GPU_EVENT_CATEGORIES = (
    "quant_kv_attention",
    "kv_prepare",
    "int_qk",
    "softmax_prob_quant",
    "int_pv",
    "gnn_score",
    "gnn_message",
    "gnn_update",
)


class LatencyEventAccumulator:
    """Collect event pairs without depending on a particular CUDA event type."""

    def __init__(self, categories=DETAIL_GPU_EVENT_CATEGORIES):
        self.categories = tuple(categories)
        self.enabled = False
        self._next_token = 0
        self._pending = {}
        self._pairs = {category: [] for category in self.categories}

    def begin(self, enabled=True):
        self.enabled = bool(enabled)
        self._next_token = 0
        self._pending = {}
        self._pairs = {category: [] for category in self.categories}

    def start(self, category, event):
        if not self.enabled:
            return None
        if category not in self._pairs:
            raise KeyError(f"Unknown latency event category: {category}")
        token = self._next_token
        self._next_token += 1
        self._pending[token] = (category, event)
        return token

    def end(self, token, event):
        if token is None:
            return
        if token not in self._pending:
            raise RuntimeError(f"Unknown or already completed latency event token: {token}")
        category, start_event = self._pending.pop(token)
        self._pairs[category].append((start_event, event))

    def finish(self):
        if self._pending:
            pending = sorted((token, category) for token, (category, _) in self._pending.items())
            raise RuntimeError(f"Unfinished per-query latency CUDA events: {pending}")
        result = {category: list(pairs) for category, pairs in self._pairs.items()}
        self.abort()
        return result

    def abort(self):
        self.enabled = False
        self._next_token = 0
        self._pending.clear()
        for pairs in self._pairs.values():
            pairs.clear()

    def pending_count(self):
        return len(self._pending)

    def pair_count(self):
        return sum(len(pairs) for pairs in self._pairs.values())


def elapsed_event_pairs_ms(pairs):
    return sum(float(start.elapsed_time(end)) for start, end in pairs)
