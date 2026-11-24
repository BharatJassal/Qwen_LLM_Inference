import torch
from collections import defaultdict
from torch.profiler import profile, ProfilerActivity
from torch.autograd import DeviceType
from src.loader import load_weights
from src.config import ModelConfig
from src.model import decode_step
from src.layers import rope_tables
from src.quantize import quantize_model, memory
from benchmark.analyze_trace import category

SPEC_GBS = 484  # GTX 1080 Ti spec sheet
MAX_LEN = 512
STEPS = 200
GB = 1e9

cfg = ModelConfig.from_json("model/config.json")
fp32 = load_weights("model/model.safetensors", "cuda")
int8 = quantize_model(fp32)


def time_ms(fn, reps):
    fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def copy_bandwidth():
    src = torch.empty(256 * 2**20, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    ms = time_ms(lambda: dst.copy_(src), 20)
    return 2 * src.numel() * 4 / (ms / 1000) / GB  # a copy reads and writes every byte


def decode_graph(weights):
    shape = (cfg.num_hidden_layers, 1, cfg.num_key_value_heads, MAX_LEN, cfg.head_dim)
    k_cache = torch.zeros(shape, device="cuda")
    v_cache = torch.zeros(shape, device="cuda")
    cos, sin = (t.cuda() for t in rope_tables(MAX_LEN, cfg.head_dim, cfg.rope_theta))
    positions = torch.arange(MAX_LEN, device="cuda")
    tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    pos = torch.zeros(1, dtype=torch.long, device="cuda")

    def step():
        decode_step(tok, pos, weights, cfg, k_cache, v_cache, cos, sin, positions)
        pos.add_(1)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    # decode_step attends over the whole preallocated cache, so all of it is read every step
    kv_bytes = 2 * k_cache.numel() * k_cache.element_size()
    return graph, pos, kv_bytes


def kernel_times(graph, pos, reps=20):
    pos.zero_()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            graph.replay()
        torch.cuda.synchronize()
    total, count = defaultdict(float), defaultdict(int)
    for e in prof.events():
        if e.device_type == DeviceType.CUDA:
            total[e.name] += e.device_time_total / reps
            count[e.name] += 1
    return total, {k: v / reps for k, v in count.items()}


with torch.no_grad():
    peak = copy_bandwidth()
    print(f"spec sheet bandwidth: {SPEC_GBS} GB/s, measured copy: {peak:.0f} GB/s ({100 * peak / SPEC_GBS:.0f}% of spec)\n")

    for name, weights in [("fp32", fp32), ("int8", int8)]:
        graph, pos, kv_bytes = decode_graph(weights)
        pos.zero_()
        ms = time_ms(graph.replay, STEPS)
        weight_bytes = memory(weights)[0]
        gbs = (weight_bytes + kv_bytes) / (ms / 1000) / GB
        print(f"{name}: {(weight_bytes + kv_bytes) / 1e6:.0f} MB read per step "
              f"({weight_bytes / 1e6:.0f} weights + {kv_bytes / 1e6:.0f} KV cache), {ms:.3f} ms per step, "
              f"{1000 / ms:.0f} steps/s")
        print(f"      {gbs:.0f} GB/s = {100 * gbs / SPEC_GBS:.0f}% of spec, {100 * gbs / peak:.0f}% of measured copy. "
              f"At copy speed this step would take {(weight_bytes + kv_bytes) / (peak * GB) * 1000:.3f} ms")

        total, count = kernel_times(graph, pos)
        busy = sum(total.values()) / 1000
        print(f"      {sum(count.values()):.0f} kernels per step, kernels add up to {busy:.3f} ms "
              f"(profiled, so a little slower than the {ms:.3f} ms above)")
        groups = defaultdict(lambda: [0.0, 0])
        for k in total:
            groups[category(k)][0] += total[k] / 1000
            groups[category(k)][1] += count[k]
        for g, (t, n) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
            line = f"      {g:32s} {t:6.3f} ms {100 * t / busy:4.0f}%  {n:5.0f} kernels"
            if g == "weight matmuls":
                line += f"  {weight_bytes / (t / 1000) / GB:.0f} GB/s ({100 * weight_bytes / (t / 1000) / GB / peak:.0f}% of copy)"
            print(line)
        print()
