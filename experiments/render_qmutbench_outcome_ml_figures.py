"""Render dependency-free SVG summaries from a QMutBench metrics JSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def _text(x, y, content, *, size=14, anchor="start", weight="normal", fill="#20242a"):
    return (f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{html.escape(str(content))}</text>')


def comparison_svg(metrics: dict) -> str:
    names = ["majority", "logistic_regression", "decision_tree", "random_forest"]
    labels = {"majority": "Majority baseline", "logistic_regression": "Logistic regression",
              "decision_tree": "Decision tree", "random_forest": "Random forest"}
    selected = metrics["selection"]["selected_model"]
    rows, width, height = [], 760, 330
    rows.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    rows.append('<rect width="100%" height="100%" fill="white"/>')
    rows.append(_text(30, 32, "Grouped cross-validation: macro F1", size=19, weight="bold"))
    x0, chart_w, y0, step = 235, 440, 78, 54
    for tick in range(0, 101, 20):
        x = x0 + chart_w * tick / 100
        rows.append(f'<line x1="{x:.1f}" y1="58" x2="{x:.1f}" y2="285" stroke="#e1e4e8"/>')
        rows.append(_text(x, 307, f"{tick}%", size=11, anchor="middle", fill="#59636e"))
    for i, name in enumerate(names):
        y = y0 + i * step
        score = metrics["selection"]["cv"][name]["mean_macro_f1"]
        color = "#176b87" if name == selected else "#89a9b8"
        rows.append(_text(x0 - 12, y + 17, labels[name], size=13, anchor="end"))
        rows.append(f'<rect x="{x0}" y="{y}" width="{chart_w * score:.2f}" height="28" rx="3" fill="{color}"/>')
        rows.append(_text(x0 + chart_w * score + 8, y + 19, f"{score:.3f}", size=12, weight="bold"))
    rows.append(_text(30, 325, "Three stratified folds grouped by Origin_program; final held-out groups excluded.", size=11, fill="#59636e"))
    rows.append("</svg>")
    svg = "\n".join(rows)
    ET.fromstring(svg)
    return svg


def confusion_svg(metrics: dict) -> str:
    matrix = metrics["final_test"]["confusion_matrix_labels_0_1"]
    row_labels = ["Actual survivor", "Actual detected"]
    col_labels = ["Predicted survivor", "Predicted detected"]
    max_value = max(max(row) for row in matrix)
    rows = ['<svg xmlns="http://www.w3.org/2000/svg" width="620" height="390" viewBox="0 0 620 390">',
            '<rect width="100%" height="100%" fill="white"/>',
            _text(30, 34, "Isolated held-out test confusion matrix", size=19, weight="bold")]
    left, top, cell_w, cell_h = 195, 100, 170, 95
    for j, label in enumerate(col_labels):
        rows.append(_text(left + j * cell_w + cell_w / 2, top - 16, label, size=12, anchor="middle"))
    for i, label in enumerate(row_labels):
        y = top + i * cell_h
        rows.append(_text(left - 16, y + cell_h / 2 + 5, label, size=13, anchor="end"))
        total = sum(matrix[i])
        for j, count in enumerate(matrix[i]):
            intensity = count / max_value
            shade = int(241 - intensity * 115)
            color = f"#{shade:02x}{min(255,shade+10):02x}{min(255,shade+18):02x}"
            x = left + j * cell_w
            rows.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="white" stroke-width="3"/>')
            rows.append(_text(x + cell_w / 2, y + 43, f"{count:,}", size=19, anchor="middle", weight="bold"))
            rows.append(_text(x + cell_w / 2, y + 68, f"{count / total:.1%} of actual class", size=11, anchor="middle"))
    rows.append(_text(30, 350, f"Accuracy {metrics['final_test']['accuracy']:.3f} | Macro F1 {metrics['final_test']['macro_f1']:.3f}", size=13, weight="bold"))
    rows.append(_text(30, 374, "Test partition contains 70 Origin_program groups; cells are row counts.", size=11, fill="#59636e"))
    rows.append("</svg>")
    svg = "\n".join(rows)
    ET.fromstring(svg)
    return svg


def render(metrics_path: Path, output_dir: Path) -> list[Path]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "grouped_cv_macro_f1.svg",
        output_dir / "heldout_confusion_matrix.svg",
    ]
    outputs[0].write_text(comparison_svg(metrics), encoding="utf-8")
    outputs[1].write_text(confusion_svg(metrics), encoding="utf-8")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("data/qmutbench_ml/outcome_model_v1/metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/qmutbench_outcome_ml"))
    args = parser.parse_args()
    for path in render(args.metrics, args.output_dir):
        print(path)
