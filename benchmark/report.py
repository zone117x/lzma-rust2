#!/usr/bin/env python3
"""Turns the benchmark's Criterion estimates into a table and two charts.

Run from this directory after `cargo bench --bench decoding`: it reads
`target/criterion`, prints a Markdown table of MiB/s per preset and decoder for
each group, and writes `assets/decompression_lzma2.svg` and
`assets/decompression_lzma.svg`.
"""

import json
import os
import sys

INPUT_BYTES = 41523712  # tests/data/executable.exe
GROUPS = [("decompression lzma2", "decompression_lzma2"), ("decompression lzma", "decompression_lzma")]
DECODERS = ["lzma-rust2 master", "lzma-rust2 asm", "liblzma", "7-Zip (C)", "7-Zip (asm)"]
COLOURS = {
    "lzma-rust2 master": "#9e9e9e",
    "lzma-rust2 asm": "#1f77b4",
    "liblzma": "#2ca02c",
    "7-Zip (C)": "#9467bd",
    "7-Zip (asm)": "#d62728",
}


def mib_per_s(path):
    with open(path) as f:
        mean_ns = json.load(f)["mean"]["point_estimate"]
    return INPUT_BYTES / (mean_ns / 1e9) / (1 << 20)


def collect(root, group):
    table = {}
    for decoder in DECODERS:
        for level in range(10):
            path = os.path.join(root, group, decoder, str(level), "new", "estimates.json")
            if os.path.exists(path):
                table.setdefault(decoder, {})[level] = mib_per_s(path)
    return table


def markdown(table):
    decoders = [d for d in DECODERS if d in table]
    lines = ["| preset | " + " | ".join(decoders) + " |", "|---|" + "---:|" * len(decoders)]
    for level in range(10):
        cells = [f"{table[d][level]:.0f}" if level in table[d] else "" for d in decoders]
        lines.append(f"| {level} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def svg(table, title):
    decoders = [d for d in DECODERS if d in table]
    width, height = 1100, 460
    left, right, top, bottom = 70, 20, 50, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    top_value = max(v for d in decoders for v in table[d].values())
    step = 50 if top_value <= 500 else 100
    y_max = (int(top_value / step) + 1) * step
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="13">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-size="16" font-weight="bold">{title}</text>',
    ]
    for value in range(0, y_max + 1, step):
        y = top + plot_h - value / y_max * plot_h
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#dddddd"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value}</text>')
    out.append(f'<text x="14" y="{top + plot_h / 2:.0f}" transform="rotate(-90 14 {top + plot_h / 2:.0f})" text-anchor="middle">MiB/s</text>')
    slot = plot_w / 10
    bar = slot * 0.8 / len(decoders)
    for level in range(10):
        x0 = left + level * slot + slot * 0.1
        for i, decoder in enumerate(decoders):
            value = table[decoder].get(level)
            if value is None:
                continue
            h = value / y_max * plot_h
            out.append(f'<rect x="{x0 + i * bar:.1f}" y="{top + plot_h - h:.1f}" width="{bar - 1:.1f}" height="{h:.1f}" fill="{COLOURS[decoder]}"/>')
        out.append(f'<text x="{left + level * slot + slot / 2:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{level}</text>')
    out.append(f'<text x="{left + plot_w / 2:.0f}" y="{height - 30}" text-anchor="middle">preset</text>')
    x = left
    for decoder in decoders:
        out.append(f'<rect x="{x}" y="{height - 18}" width="12" height="12" fill="{COLOURS[decoder]}"/>')
        out.append(f'<text x="{x + 16}" y="{height - 8}">{decoder}</text>')
        x += 16 + 8 * len(decoder) + 24
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target", "criterion")
    if not os.path.isdir(root):
        sys.exit("no target/criterion here: run `cargo bench --bench decoding` first")
    os.makedirs("assets", exist_ok=True)
    for group, stem in GROUPS:
        table = collect(root, group)
        if not table:
            continue
        print(f"### {group}\n")
        print(markdown(table))
        print()
        with open(os.path.join("assets", stem + ".svg"), "w") as f:
            f.write(svg(table, group + ", MiB/s of output"))


if __name__ == "__main__":
    main()
