import csv
import statistics
import sys
from src.config import ModelConfig

OUTPUT_LAYER_BLOCKS = "18992"  # 151936 vocab rows / 8 rows per block, last kernel of every decode step
COLORS = {"weight matmuls": "#4e79a7", "attention (QK, softmax, AV)": "#f28e2b",
          "small ops (norm, RoPE, add...)": "#59a14f"}


def category(kernel):
    if "int8_gemv" in kernel or "gemv2T" in kernel:
        return "weight matmuls"
    if "gemm" in kernel or "softmax" in kernel:
        return "attention (QK, softmax, AV)"
    return "small ops (norm, RoPE, add...)"


def read_trace(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    header = next(i for i, l in enumerate(lines) if l.startswith('"Start"'))
    rows = list(csv.DictReader(lines[header:]))[1:]  # second row is units
    return [(float(r["Start"]), float(r["Duration"]), r["Name"], r["Grid X"]) for r in rows if r["Duration"]]


def steps(kernels):
    ends = [i for i, k in enumerate(kernels) if "int8_gemv" in k[2] and k[3] == OUTPUT_LAYER_BLOCKS]
    return [kernels[a + 1:b + 1] for a, b in zip(ends, ends[1:])]


def step_stats(step):
    start, end = step[0][0], step[-1][0] + step[-1][1]
    busy = sum(k[1] for k in step)
    return end - start, busy, len(step)


def projection_table(all_steps):
    # per layer the kernel runs qkv, o, gate_up, down, then once for the output layer
    shapes = [("qkv", 1152, 896), ("o", 896, 896), ("gate_up", 9728, 896), ("down", 896, 4864)]
    times = {}
    for step in all_steps:
        gemvs = [k for k in step if "int8_gemv" in k[2]]
        assert len(gemvs) % len(shapes) == 1, f"expected {len(shapes)} per layer + 1, got {len(gemvs)}"
        for i, k in enumerate(gemvs[:-1]):
            times.setdefault(shapes[i % len(shapes)], []).append(k[1])
        times.setdefault(("output", 151936, 896), []).append(gemvs[-1][1])
    print(f"{'projection':>10} {'rows x cols':>13} {'blocks':>6} {'us':>7} {'GB/s':>6}")
    for (name, rows, cols), t in times.items():
        us = statistics.median(t)
        gbs = (rows * cols + 4 * rows + 4 * cols) / us / 1000
        print(f"{name:>10} {f'{rows}x{cols}':>13} {rows // 8:>6} {us:7.2f} {gbs:6.0f}")


def layer(step, i):
    # kernels from after layer i-1's down projection to layer i's down projection.
    # 7 matmuls per layer before q/k/v and gate/up were merged, 4 after
    idx = [j for j, k in enumerate(step) if "int8_gemv" in k[2]]
    per_layer = (len(idx) - 1) // ModelConfig.from_json("model/config.json").num_hidden_layers
    return step[idx[per_layer * i - 1] + 1 if i else 0:idx[per_layer * i + per_layer - 1] + 1]


def panel(parts, title, rows, y, x0=150, width=1000):
    span = max(r[3] for r in rows)
    scale = (width - x0 - 20) / span
    tick = next(t for t in [50, 100, 200, 500, 1000, 2000, 5000] if span / t <= 10)
    parts.append(f'<text x="10" y="{y}" font-size="14" font-weight="bold">{title}</text>')
    for i, (label, note, kernels, wall) in enumerate(rows):
        top = y + 12 + i * 50
        parts.append(f'<text x="10" y="{top + 14}">{label}</text>')
        parts.append(f'<text x="10" y="{top + 30}" fill="#555">{note}</text>')
        parts.append(f'<rect x="{x0}" y="{top}" width="{wall * scale:.2f}" height="34" fill="#e6e6e6"/>')
        t0 = kernels[0][0]
        for start, dur, name, _ in kernels:
            parts.append(f'<rect x="{x0 + (start - t0) * scale:.2f}" y="{top}" width="{max(dur * scale, 0.4):.2f}" '
                         f'height="34" fill="{COLORS[category(name)]}"/>')
    axis = y + 12 + len(rows) * 50
    for t in range(0, int(span) + 1, tick):
        x = x0 + t * scale
        label = f"{t / 1000:g} ms" if tick >= 1000 else f"{t} us"
        parts.append(f'<line x1="{x:.1f}" y1="{axis - 8}" x2="{x:.1f}" y2="{axis - 3}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{axis + 9}" text-anchor="middle" font-size="11">{label}</text>')
    return axis + 30


def chart(traces, out, table_for=None):
    steps_rows, layer_rows = [], []
    for path, label in traces:
        all_steps = steps(read_trace(path))[1:]  # first step still warms up
        stats = [step_stats(s) for s in all_steps]
        wall = statistics.median(s[0] for s in stats)
        busy = statistics.median(s[1] for s in stats)
        n = statistics.median(s[2] for s in stats)
        print(f"{label}: {len(all_steps)} steps, median {wall / 1000:.2f} ms per step, GPU busy {busy / 1000:.2f} ms "
              f"({100 * busy / wall:.0f}%), idle {100 * (1 - busy / wall):.0f}%, {n:.0f} kernels per step")
        if path == table_for:
            projection_table(all_steps)
        middle = sorted(all_steps, key=lambda s: step_stats(s)[0])[len(all_steps) // 2]
        wall, busy, n = step_stats(middle)
        steps_rows.append((label, f"{wall / 1000:.1f} ms, {n} kernels", middle, wall))
        one = layer(middle, 12)
        wall, busy, n = step_stats(one)
        layer_rows.append((label, f"{wall:.0f} us, {100 * (1 - busy / wall):.0f}% idle", one, wall))
        print(f"    layer 12 of the median step: {wall:.0f} us, {n} kernels, {100 * (1 - busy / wall):.0f}% idle")

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 380" font-family="sans-serif" font-size="13">',
             '<rect width="1000" height="380" fill="white"/>']
    y = panel(parts, "One INT8 decode step (nvprof trace, GTX 1080 Ti)", steps_rows, 22)
    y = panel(parts, "Zoomed in: one of the 24 transformer layers", layer_rows, y)
    x = 150
    for name, color in list(COLORS.items()) + [("idle", "#e6e6e6")]:
        parts.append(f'<rect x="{x}" y="{y - 8}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 17}" y="{y + 3}" font-size="12">{name}</text>')
        x += 30 + 7 * len(name)
    parts.append("</svg>")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    graph = "benchmark/trace_graph.csv"
    chart([("benchmark/trace_cached.csv", "without CUDA graph"), (graph, "with CUDA graph")],
          "benchmark/timeline.svg", table_for=graph)
    # optional: a graph trace of an older version to compare against
    if len(sys.argv) > 1:
        chart([(sys.argv[1], "before fusion"), (graph, "after fusion")], "benchmark/fusion.svg")


if __name__ == "__main__":
    main()
