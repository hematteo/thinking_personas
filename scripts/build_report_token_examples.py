"""Export illustrative token-colored examples directly from frozen measurements.

Selection is qualitative and post hoc, not by activation extrema. These files
are static scientific figures; the interactive viewer is not used or captured.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from build_literature_figures import digest, text_groups

CASES = [
    ("01_poet_identity", "role2-identity-01", "poet", "Identity question · poet and poetic style"),
    ("02_mystic_identity", "role2-identity-01", "mystic", "Identity question · mystic and contemplative style"),
    ("03_poet_reasoning", "role2-reasoning-04", "poet", "Reasoning question · poet and poetic style"),
]


def wrap_groups(groups, width=76):
    chars = [(ch, float(np.mean(g["scores"]))) for g in groups for ch in g["text"]]
    raw = "".join(ch for ch, _ in chars)
    wrapper = textwrap.TextWrapper(width=width, replace_whitespace=False,
                                  drop_whitespace=False, expand_tabs=False,
                                  break_on_hyphens=False)
    lines, pointer = [], 0
    for paragraph in raw.split("\n"):
        for line in wrapper.wrap(paragraph) or [""]:
            part = chars[pointer:pointer + len(line)]
            assert "".join(ch for ch, _ in part) == line
            lines.append(part)
            pointer += len(line)
        if pointer < len(chars):
            assert chars[pointer][0] == "\n"
            pointer += 1
    assert pointer == len(chars)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/role-susceptibility-20260912"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/report-token-examples"))
    parser.add_argument("--tokens", type=int, default=112)
    args = parser.parse_args()
    assert args.tokens > 0
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path("data/literature-reference/tokenizer.json")
    scale_path = Path("artifacts/literature-figures/color_scale.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    color = json.loads(scale_path.read_text())
    norm = Normalize(color["low"], color["high"], clip=True)
    cmap = plt.get_cmap("cividis")
    questions = {r["prompt_id"]: r for r in map(json.loads, (args.run / "questions.jsonl").read_text().splitlines())}
    scalar = pd.read_csv(args.run / "segment_means.csv").set_index(["record_id", "segment"])
    needed = {(q, a) for _, q, role, _ in CASES for a in ["default", f"role_{role}", f"style_{role}"]}
    records, evidence, hashes = {}, [], {}
    for path in sorted((args.run / "transcripts").glob("*.json")):
        record = json.loads(path.read_text())
        key = record["prompt_id"], record["arm"]
        if record["phase"] != "main" or record["seed"] != 17 or key not in needed:
            continue
        assert key not in records and record["valid"] and not record["synthetic"]
        token_path = args.run / "token_shards" / f"{record['record_id']}.csv.gz"
        tokens = pd.read_csv(token_path)
        assert tokens.record_id.eq(record["record_id"]).all() and tokens.layer.eq(32).all()
        assert np.isfinite(tokens.dot_assistant).all()
        segments = {}
        for segment in ["think", "answer"]:
            rows = tokens[tokens.segment.eq(segment)].sort_values("token_idx")
            span = next(s for s in record["spans"] if s["segment"] == segment)
            assert rows.token_idx.tolist() == list(range(span["start"], span["end"]))
            assert rows.token_id.tolist() == [record["input_ids"][i] for i in rows.token_idx]
            groups = text_groups(rows.token_id, rows.token_idx, rows.dot_assistant, tokenizer)
            assert np.isclose(rows.dot_assistant.mean(), scalar.loc[(record["record_id"], segment), "dot_assistant"], atol=1e-5)
            assert "".join(g["text"] for g in groups).strip() == record[f"{segment}_text"].strip()
            selected, count = [], 0
            for group in groups:
                if count >= args.tokens:
                    break
                selected.append(group)
                count += len(group["ids"])
            segments[segment] = dict(groups=selected, total=len(rows), shown=count,
                                     mean=float(rows.dot_assistant.mean()), lines=wrap_groups(selected))
            evidence.append(dict(record_id=record["record_id"], prompt_id=key[0], arm=key[1],
                                 seed=17, segment=segment, shown=count, total=len(rows),
                                 full_segment_mean=float(rows.dot_assistant.mean()), groups=selected))
        records[key] = dict(record=record, segments=segments)
        for source in [path, token_path]:
            hashes[str(source)] = digest(source)
    assert set(records) == needed
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    figure_checks = []
    with PdfPages(out / "token_examples.pdf") as book:
        for name, qid, role, title in CASES:
            arms = ["default", f"role_{role}", f"style_{role}"]
            labels = ["Default assistant", f"{role.title()} role",
                      "Poetic style · instructed to remain AI" if role == "poet" else "Contemplative style · instructed to remain AI"]
            prompt_lines = textwrap.wrap("Prompt: " + questions[qid]["text"], 145)
            header = 0.85 + len(prompt_lines) * 0.19
            row_heights = [0.80 + max(len(records[qid, arm]["segments"][s]["lines"]) for s in ["think", "answer"]) * 0.19 for arm in arms]
            height, width = header + sum(row_heights) + 1.05, 14.5
            fig = plt.figure(figsize=(width, height))
            fig.text(0.035, 1 - 0.2 / height, title, fontsize=16, weight="bold", va="top")
            fig.text(0.035, 1 - 0.55 / height, "\n".join(prompt_lines), fontsize=10, va="top", linespacing=1.35)
            top = header
            text_artists = []
            for arm, label, row_height in zip(arms, labels, row_heights):
                for col, segment in enumerate(["think", "answer"]):
                    seg = records[qid, arm]["segments"][segment]
                    left = 0.035 if col == 0 else 0.525
                    fig.text(left, 1 - top / height, f"{label}  |  {'Thinking' if segment == 'think' else 'Answer'}", fontsize=10, weight="bold", va="top")
                    fig.text(left, 1 - (top + 0.23) / height,
                             f"First {seg['shown']} / {seg['total']} tokens · full-segment mean {seg['mean']:+.2f}",
                             fontsize=9, color="#555555", va="top")
                    nlines = len(seg["lines"])
                    ax = fig.add_axes([left, 1 - (top + 0.49 + nlines * 0.19) / height, 0.44, nlines * 0.19 / height])
                    ax.set(xlim=(0, 76), ylim=(-nlines + 0.05, 0.05))
                    ax.axis("off")
                    for line_idx, line in enumerate(seg["lines"]):
                        for char_idx, (char, score) in enumerate(line):
                            bg = cmap(norm(score))
                            # Choose text color by sRGB relative luminance contrast.
                            rgb = np.asarray(bg[:3]); linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                            lum = linear @ np.array([.2126, .7152, .0722])
                            fg = "black" if lum > .179 else "white"
                            ax.add_patch(Rectangle((char_idx, -line_idx - .85), 1.015, .85, facecolor=bg, edgecolor="none"))
                            text_artists.append((ax, ax.text(char_idx + .5, -line_idx - .41, char, ha="center", va="center", fontfamily="DejaVu Sans Mono", fontsize=9, color=fg)))
                    if seg["shown"] < seg["total"]:
                        ax.text(75.5, -nlines + .05, "…", color="#555555", fontsize=10, ha="right", va="top")
                top += row_height
            cax = fig.add_axes([.26, .57 / height, .48, .13 / height])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both")
            cb.ax.tick_params(labelsize=8, length=2)
            cb.set_label("Raw Assistant-axis projection · blue = lower, yellow = higher", fontsize=9, labelpad=3)
            fig.text(.035, .93 / height,
                     f"{qid} · seed 17 · Qwen3-32B, block 32 · illustrative selection; opening excerpts, not full responses",
                     fontsize=9)
            fig.text(.035, .06 / height,
                     "Same global color scale in every panel (1st–99th token percentiles); colors saturate at endpoints. Scores are not persona probabilities.", fontsize=8)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax, artist in text_artists:
                bbox = artist.get_window_extent(renderer)
                bounds = ax.get_window_extent(renderer)
                assert bbox.x0 >= bounds.x0 - 2 and bbox.x1 <= bounds.x1 + 2
                assert bbox.y0 >= bounds.y0 - 2 and bbox.y1 <= bounds.y1 + 2
            for ext in ["png", "pdf"]:
                fig.savefig(out / f"{name}.{ext}", dpi=180)
            book.savefig(fig)
            figure_checks.append(dict(figure=name, text_characters_checked=len(text_artists), text_within_axes=True))
            plt.close(fig)
            print(f"Exported {name}", flush=True)
    (out / "excerpt_data.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    full = ["# Complete source transcripts for the illustrated examples", "", "Post hoc qualitative examples; all use seed 17. No generated text is corrected or rewritten.", ""]
    for (qid, arm), item in sorted(records.items()):
        r = item["record"]
        full.extend([f"## {qid} / {arm}", "", f"Record: `{r['record_id']}`", "", "### System instruction", "", r["messages"][0]["content"], "", "### User question", "", questions[qid]["text"], "", "### Thinking", "", r["think_text"], "", "### Answer", "", r["answer_text"], ""])
    (out / "full_transcripts.md").write_text("\n".join(full))
    for p in [Path(__file__), tokenizer_path, scale_path, args.run / "questions.jsonl", args.run / "arms.json", args.run / "segment_means.csv"]:
        hashes[str(p)] = digest(p)
    (out / "verification.json").write_text(json.dumps(dict(records=len(records), segments=len(evidence),
        exact_ids_spans_text_scores_checked=True, figures=figure_checks, source_hashes=hashes,
        color_scale=color, selection="Post hoc qualitative selection from previously reviewed questions; seed 17 fixed; no activation-extrema search.",
        excerpt_rule=f"First {args.tokens} content tokens per segment, extending only to complete split UTF-8 characters."), indent=2) + "\n")


if __name__ == "__main__":
    main()
