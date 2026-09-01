from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


ORDER = ["A", "B", "C", "D", "E"]
COLORS = {
    "A": "#E76F51",
    "B": "#F4A261",
    "C": "#9CA3AF",
    "D": "#2A9D8F",
    "E": "#6C63A8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "ontology/output/failure_analysis/"
            "hybrid_steiner_on_ex_failures_0_99.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "ontology/output/failure_analysis/"
            "hybrid_steiner_failure_categories"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    counts = [
        int(report["category_counts"][code]["count"]) for code in ORDER
    ]
    names = [
        report["category_counts"][code]["name"] for code in ORDER
    ]
    total = sum(counts)
    if total != 28:
        raise SystemExit(f"Expected 28 failures, got {total}")

    font_path = _korean_font()
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#F8FAFC",
        }
    )

    fig, (bar_ax, pie_ax) = plt.subplots(
        1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    fig.suptitle(
        "Hybrid + Steiner ON: EX 실패 28건 원인 분포",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    labels = [f"{code}. {name}" for code, name in zip(ORDER, names)]
    bars = bar_ax.barh(
        labels,
        counts,
        color=[COLORS[code] for code in ORDER],
        height=0.62,
    )
    bar_ax.invert_yaxis()
    bar_ax.set_title("원인별 실패 건수", fontsize=15, pad=15)
    bar_ax.set_xlabel("문항 수")
    bar_ax.set_xlim(0, max(counts) + 4)
    bar_ax.grid(axis="x", linestyle="--", alpha=0.28)
    bar_ax.spines[["top", "right", "left"]].set_visible(False)
    bar_ax.tick_params(axis="y", length=0)
    for bar, count in zip(bars, counts):
        percentage = count / total * 100
        bar_ax.text(
            count + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{count}건 ({percentage:.1f}%)",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    nonzero = [
        (code, name, count)
        for code, name, count in zip(ORDER, names, counts)
        if count
    ]
    wedges, _ = pie_ax.pie(
        [item[2] for item in nonzero],
        startangle=90,
        counterclock=False,
        colors=[COLORS[item[0]] for item in nonzero],
        wedgeprops={"width": 0.42, "edgecolor": "#F8FAFC", "linewidth": 3},
    )
    pie_ax.text(
        0,
        0.07,
        "EX 실패",
        ha="center",
        va="center",
        fontsize=13,
        color="#475569",
    )
    pie_ax.text(
        0,
        -0.13,
        str(total),
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="#0F172A",
    )
    pie_ax.set_title("실패 비율", fontsize=15, pad=15)
    pie_ax.legend(
        wedges,
        [
            f"{code}: {count}건 ({count / total * 100:.1f}%)"
            for code, _name, count in nonzero
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    fig.text(
        0.5,
        0.025,
        "개발 세트 index 0–99 · EX 0.72 · C(Join Linker 최초 오류)는 0건",
        ha="center",
        fontsize=10.5,
        color="#64748B",
    )
    fig.tight_layout(rect=(0.03, 0.07, 0.97, 0.92))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        output = args.output_prefix.with_suffix(suffix)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(f"wrote {output}")
    plt.close(fig)


def _korean_font() -> str:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    matches = font_manager.findSystemFonts()
    for match in matches:
        if "NotoSansCJK" in match.replace("-", ""):
            return match
    raise SystemExit("Korean-capable Noto Sans CJK font not found")


if __name__ == "__main__":
    main()
