import queue
import threading
import torch
from src.model import forward, decode_step
from src.layers import rope_tables
from src.generate import sample


class Request:
    def __init__(self, ids, max_tokens, temperature, top_p):
        self.ids = ids
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.out = queue.Queue()
        self.count = 0
        self.cancelled = False


class BatchEngine:
    """Continuous batching: a background thread decodes every active request together in one CUDA graph step."""

    def __init__(self, weights, config, slots=8, max_len=2048):
        self.weights = weights
        self.config = config
        self.max_len = max_len
        self.slots = [None] * slots
        self.pending = queue.Queue()

        shape = (config.num_hidden_layers, slots, config.num_key_value_heads, max_len, config.head_dim)
        self.k_cache = torch.zeros(shape, device="cuda")
        self.v_cache = torch.zeros(shape, device="cuda")
        self.tok = torch.zeros(slots, 1, dtype=torch.long, device="cuda")
        self.pos = torch.zeros(slots, dtype=torch.long, device="cuda")
        # keep these alive, the graphs read them by address
        self.cos, self.sin = (t.to("cuda") for t in rope_tables(max_len, config.head_dim, config.rope_theta))
        self.positions = torch.arange(max_len, device="cuda")

        def step(b):
            logits = decode_step(self.tok[:b], self.pos[:b], weights, config, self.k_cache[:, :b],
                                 self.v_cache[:, :b], self.cos, self.sin, self.positions)
            # empty slots step too, keep them inside the cache
            self.pos[:b].add_(1).clamp_(max=max_len - 1)
            return logits[:, -1, :]

        # a graph per batch size so a single request doesn't pay for all 8 slots
        self.sizes = sorted({min(2 ** i, slots) for i in range(slots.bit_length() + 1)})
        self.graphs = {}
        with torch.no_grad():
            for b in self.sizes:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    step(b)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    logits = step(b)
                self.graphs[b] = (graph, logits)
        self.pos.zero_()

        threading.Thread(target=self.loop, daemon=True).start()

    def submit(self, ids, max_tokens, temperature, top_p):
        if ids.shape[1] + max_tokens > self.max_len:
            raise ValueError(f"prompt plus max_tokens must fit in {self.max_len} tokens")
        req = Request(ids, max_tokens, temperature, top_p)
        self.pending.put(req)
        return self.stream(req)

    def stream(self, req):
        try:
            while True:
                item = req.out.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            req.cancelled = True

    def pick(self, req, logits):
        return sample(logits, req.temperature, req.top_p).reshape(())

    def finish(self, slot, item=None):
        self.slots[slot].out.put(item)
        self.slots[slot] = None

    def advance(self, slot, t):
        req = self.slots[slot]
        if req.cancelled or t == self.config.eos_token_id or req.count >= req.max_tokens:
            self.finish(slot)
            return
        req.out.put(t)
        req.count += 1
        if req.count == req.max_tokens:
            self.finish(slot)
            return
        self.tok[slot, 0] = t

    def admit(self):
        while None in self.slots:
            idle = all(r is None for r in self.slots)
            try:
                req = self.pending.get(block=idle)
            except queue.Empty:
                return
            if req.cancelled:
                continue
            slot = self.slots.index(None)
            self.slots[slot] = req
            try:
                logits, cache = forward(req.ids, self.weights, self.config)
                n = req.ids.shape[1]
                for i, (k, v) in enumerate(cache):
                    self.k_cache[i, slot, :, :n] = k[0]
                    self.v_cache[i, slot, :, :n] = v[0]
                self.pos[slot] = n
                self.advance(slot, self.pick(req, logits[0, -1, :]).item())
            except Exception as e:
                self.finish(slot, e)

    def loop(self):
        with torch.no_grad():
            while True:
                self.admit()
                active = [s for s, r in enumerate(self.slots) if r is not None]
                if not active:
                    continue
                try:
                    graph, logits = self.graphs[next(b for b in self.sizes if b > active[-1])]
                    graph.replay()
                    picks = torch.stack([self.pick(self.slots[s], logits[s]) for s in active]).tolist()
                    for s, t in zip(active, picks):
                        self.advance(s, t)
                except Exception as e:
                    for s in active:
                        if self.slots[s] is not None:
                            self.finish(s, e)
