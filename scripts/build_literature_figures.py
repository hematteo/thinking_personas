"""CPU-only adaptations of Assistant Axis and Representation Engineering figures.

The reference PCA is fitted solely to the 275 released Qwen role vectors.
No experiment observations select its basis. Original run artifacts stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

BLUE, GOLD, GRAY = "#2962a3", "#b58a28", "#787878"
SEGMENTS = ["think", "answer"]
EXAMPLE_QUESTIONS = ["role2-identity-01", "role2-social-04", "role2-reasoning-04"]
EXAMPLE_ARMS = ["default", "role_poet", "style_poet"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pca(x):
    """Unscaled mean-centered PCA; variance uses a population (1/n) denominator."""
    x = np.asarray(x, dtype=np.float64)
    center = x.mean(axis=0)
    _, singular, basis = np.linalg.svd(x - center, full_matrices=False)
    keep = singular > singular[0] * 1e-10
    basis, singular = basis[keep], singular[keep]
    for row in basis:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    variance = singular**2 / len(x)
    assert np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-8)
    assert np.isclose(variance.sum(), np.mean(np.sum((x-center)**2, axis=1)))
    return center, basis, variance


def normalize(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    assert np.all(norms > 1e-12)
    return x / norms


def byte_decoder():
    # Inverse GPT/Qwen byte-level alphabet; tokenizers' ByteLevel decoder is
    # checked independently against every reconstructed segment below.
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = values.copy()
    for b in range(256):
        if b not in values:
            values.append(b)
            chars.append(256 + len(chars) - 188)
    return {chr(c): b for c, b in zip(chars, values)}


def text_groups(ids, indices, scores, tokenizer):
    """Preserve UTF-8 even when a character crosses token boundaries.

    A merged group gets the mean display color; every constituent token's raw
    score and index is retained. No decoded replacement characters are inserted.
    """
    decoder = byte_decoder()
    groups, pending, token_indices, token_ids, values = [], b"", [], [], []
    for token_id, index, score in zip(ids, indices, scores):
        piece = tokenizer.id_to_token(int(token_id))
        pending += bytes(decoder[c] for c in piece)
        token_indices.append(int(index)); token_ids.append(int(token_id)); values.append(float(score))
        try:
            text = pending.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data":
                raise
            continue
        groups.append(dict(text=text, indices=token_indices, ids=token_ids, scores=values))
        pending, token_indices, token_ids, values = b"", [], [], []
    assert not pending, "Incomplete Unicode at segment end"
    actual = "".join(g["text"] for g in groups)
    expected = tokenizer.decode([int(t) for t in ids], skip_special_tokens=False)
    assert actual == expected, "Token-to-text reconstruction disagrees with pinned tokenizer"
    return groups


def bin_tokens(values, bins=100):
    """Mean in normalized-position bins; never interpolate or fill empty bins."""
    values = np.asarray(values)
    positions = np.minimum(np.arange(len(values)) * bins // len(values), bins-1)
    sums = np.bincount(positions, weights=values, minlength=bins)
    counts = np.bincount(positions, minlength=bins)
    means = np.divide(sums, counts, out=np.full(bins, np.nan), where=counts > 0)
    assert np.isclose(np.nansum(means*counts), values.sum())
    return means, counts


def load_run(root, tokenizer):
    manifest = json.loads((root / "manifest.json").read_text())
    assert not manifest["synthetic"]
    questions = [json.loads(s) for s in (root / "questions.jsonl").read_text().splitlines()]
    qids = [q["prompt_id"] for q in questions]
    arm_specs = json.loads((root / "arms.json").read_text())
    arms = [a["arm"] for a in arm_specs]
    seeds = manifest["spec"]["seeds"]
    data = np.full((len(qids), len(seeds), len(arms), 2, 5120), np.nan)
    records, hashes, counts, all_scores = [], {}, np.zeros(data.shape[:3], int), []
    scalar = pd.read_csv(root / "segment_means.csv").set_index(["record_id", "segment"])
    direction_file = np.load(root / "directions.npz")
    names = json.loads(direction_file["metadata"].item())["direction_names"]
    assistant = direction_file["directions"][names.index("assistant")].astype(float)
    max_error, n_tokens, merged_groups = 0., 0, 0
    for path in sorted((root / "transcripts").glob("*.json")):
        record = json.loads(path.read_text())
        if record["phase"] != "main":
            continue
        assert record["valid"] and not record["synthetic"]
        rid = record["record_id"]
        vector_path = root / "segment_vectors" / f"{rid}.npz"
        token_path = root / "token_shards" / f"{rid}.csv.gz"
        for source in [path, vector_path, token_path]:
            hashes[str(source.relative_to(root))] = digest(source)
        vectors = np.load(vector_path)
        idx = qids.index(record["prompt_id"]), seeds.index(record["seed"]), arms.index(record["arm"])
        data[idx] = np.stack([vectors[s] for s in SEGMENTS])
        counts[idx] += 1
        tokens = pd.read_csv(token_path)
        assert tokens.record_id.eq(rid).all() and tokens.layer.eq(32).all()
        assert not tokens.token_idx.duplicated().any()
        assert np.isfinite(tokens.dot_assistant).all()
        assert tokens.token_id.tolist() == [record["input_ids"][int(i)] for i in tokens.token_idx]
        n_tokens += len(tokens)
        all_scores.extend(tokens.dot_assistant.tolist())
        entry = {k: record[k] for k in ["record_id", "prompt_id", "domain", "arm", "seed"]}
        entry["system_prompt"] = record["messages"][0]["content"]
        entry["segments"] = {}
        for segment in SEGMENTS:
            rows = tokens[tokens.segment.eq(segment)].sort_values("token_idx")
            span = next(s for s in record["spans"] if s["segment"] == segment)
            assert rows.token_idx.tolist() == list(range(span["start"], span["end"]))
            score = float(scalar.loc[(rid, segment), "dot_assistant"])
            max_error = max(max_error, abs(vectors[segment] @ assistant - score))
            assert np.isclose(rows.dot_assistant.mean(), score, atol=1e-5)
            groups = text_groups(rows.token_id, rows.token_idx, rows.dot_assistant, tokenizer)
            assert "".join(g["text"] for g in groups).strip() == record[f"{segment}_text"].strip()
            merged_groups += sum(len(g["ids"]) > 1 for g in groups)
            entry["segments"][segment] = dict(groups=groups, count=len(rows), mean=score)
        records.append(entry)
    assert np.all(counts == 1) and np.isfinite(data).all()
    assert max_error < 1e-4
    return dict(manifest=manifest, questions=questions, arms=arms, arm_specs=arm_specs,
                seeds=seeds, data=data, records=records, hashes=hashes, assistant=assistant,
                all_scores=np.array(all_scores), verification=dict(
                    records=len(records), vectors=int(counts.sum()*2), tokens=n_tokens,
                    all_token_ids_and_spans_match=True, all_segment_text_roundtrips_match=True,
                    merged_unicode_groups=merged_groups, max_vector_projection_error=max_error))


def reference_pca(reference, assistant):
    manifest = json.loads((reference / "vector_download_manifest.json").read_text())
    cache_manifest = json.loads((reference / "reference_layer32_manifest.json").read_text())
    assert cache_manifest["original_hashes"] == manifest["hashes"]
    assert digest(reference / "reference_layer32.npz") == cache_manifest["sha256"]
    cache = np.load(reference / "reference_layer32.npz")
    vectors = cache["role_vectors"].astype(float)
    assert vectors.shape == (275, 5120) and manifest["role_count"] == 275
    default = cache["default"].astype(float)
    published_axis = cache["assistant"].astype(float)
    axis_cos = float(normalize(published_axis) @ normalize(assistant))
    assert axis_cos > .999999, "Published axis differs from the axis used in our experiment"
    center, basis, variance = pca(vectors)
    # Sign is arbitrary; put Assistant-like at positive PC1. Remaining PCs use
    # the documented largest-loading convention, independent of our results.
    if basis[0] @ published_axis < 0:
        basis[0] *= -1
    return dict(vectors=vectors, labels=cache["labels"].tolist(), default=default,
                center=center, basis=basis, variance=variance, axis_cos=axis_cos,
                manifest=manifest)


def plot_scree(run, ref, out, save):
    means = run["data"].mean(axis=(0, 1))
    roles = [i for i, a in enumerate(run["arms"]) if a.startswith("role_")]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), layout="constrained")
    rows, summaries = [], {}
    for s, label, color in [(0, "Thinking", BLUE), (1, "Answer", GOLD)]:
        _, basis, var = pca(means[roles, s]); frac = var/var.sum()
        xs = np.arange(1, len(var)+1)
        axes[0].plot(xs, var, "o-", color=color, label=label)
        axes[1].plot(xs, np.cumsum(frac)*100, "o-", color=color, label=label)
        for seed_idx in range(len(run["seeds"])):
            _, _, seed_var = pca(run["data"][:, seed_idx, roles, s].mean(axis=0))
            axes[0].plot(xs, seed_var, color=color, alpha=.22, lw=.9)
            axes[1].plot(xs, np.cumsum(seed_var/seed_var.sum())*100, color=color, alpha=.22, lw=.9)
        summaries[SEGMENTS[s]] = dict(total_variance=float(var.sum()), rms_spread=float(np.sqrt(var.sum())),
                                     dimensions_70=int(np.searchsorted(np.cumsum(frac), .7)+1),
                                     first_three_fractions=frac[:3].tolist())
        rows.extend(dict(dataset=SEGMENTS[s], pc=i+1, variance=v, fraction=f,
                         cumulative=np.sum(frac[:i+1])) for i,(v,f) in enumerate(zip(var,frac)))
    axes[0].set(title="Absolute between-role variance", xlabel="Principal component", ylabel="Mean squared activation distance", ylim=(0, None))
    axes[1].set(title="Fraction of each segment's variance", xlabel="Principal component", ylabel="Cumulative variance (%)", ylim=(0, 105))
    axes[1].axhline(70, color=GRAY, ls="--", lw=.8)
    for ax in axes[:2]:
        ax.set_xticks(range(1, 9)); ax.legend(); ax.grid(alpha=.15)
    f = ref["variance"]/ref["variance"].sum(); xs = np.arange(1, len(f)+1)
    axes[2].bar(xs[:40], f[:40]*100, color=BLUE, alpha=.4, label="Individual PC")
    axes[2].plot(xs[:40], np.cumsum(f)[:40]*100, color=BLUE, label="Cumulative")
    d70 = int(np.searchsorted(np.cumsum(f), .7)+1)
    axes[2].axhline(70, color=GRAY, ls="--", lw=.8)
    axes[2].text(39, 72, f"70%: {d70} PCs", ha="right", fontsize=9)
    axes[2].set(title="Released reference: 275 roles", xlabel="Principal component (first 40)", ylabel="Variance explained (%)", ylim=(0, 105), xlim=(0, 41))
    axes[2].legend(loc="lower right")
    fig.suptitle("PCA variance · Qwen3-32B, decoder block 32", fontsize=14)
    fig.supxlabel("Left and middle: nine fixed roles; bold = pooled seeds, faint = individual seeds. Default/style excluded from fitting.", fontsize=9)
    rows.extend(dict(dataset="released_reference", pc=i+1, variance=v, fraction=f[i], cumulative=np.sum(f[:i+1])) for i,v in enumerate(ref["variance"]))
    pd.DataFrame(rows).to_csv(out / "scree_values.csv", index=False)
    summaries["released_reference"] = dict(dimensions_70=d70, rank=len(ref["variance"]), first_three_fractions=f[:3].tolist())
    save(fig, "01_scree")
    return summaries


def plot_reference(run, ref, out, save):
    means = run["data"].mean(axis=(0, 1)); arms = run["arms"]
    center, basis = ref["center"], ref["basis"]
    coords = (means-center) @ basis[:3].T
    background = (ref["vectors"]-center) @ basis[:3].T
    default = (ref["default"]-center) @ basis[:3].T
    fractions = ref["variance"]/ref["variance"].sum()
    np.savez_compressed(out / "reference_pca.npz", center=center, components=basis,
                        variance=ref["variance"], role_vectors=ref["vectors"], labels=np.array(ref["labels"]),
                        default_vector=ref["default"], experiment_centroids=means, arms=np.array(arms))
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.8), layout="constrained")
    for ax, component in zip(axes, [1, 2]):
        ax.scatter(background[:, 0], background[:, component], c="#bababa", s=13, alpha=.45, edgecolors="none")
        ax.scatter(default[0], default[component], marker="*", c="black", s=160, zorder=5)
        for i, arm in enumerate(arms):
            marker = "*" if arm == "default" else "s" if arm.startswith("style_") else "o"
            size = 130 if arm == "default" else 44
            a, b = coords[i][:, [0, component]]
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=GRAY, lw=.8, alpha=.6))
            ax.scatter(*a, marker=marker, facecolors="white", edgecolors=BLUE, s=size, linewidths=1.5, zorder=4)
            ax.scatter(*b, marker=marker, c=GOLD, s=size, edgecolors="white", linewidths=.4, zorder=4)
            offsets = {0:(0,16), 1:(15,0), 2:(15,-13), 3:(15,13), 4:(-15,12), 6:(-13,-12), 9:(8,0)}
            offset = offsets.get(i, (4,4))
            ax.annotate(str(i), b, xytext=offset, textcoords="offset points", fontsize=8,
                        arrowprops=dict(arrowstyle="-",color=GRAY,lw=.45) if i in offsets else None)
        ax.set(xlabel=f"Reference PC1 ({100*fractions[0]:.1f}%)", ylabel=f"Reference PC{component+1} ({100*fractions[component]:.1f}%)")
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.15)
    labels = [f"{i}: {a.replace('role_', '').replace('style_', 'style ')}" for i,a in enumerate(arms)]
    handles = [Line2D([],[],ls="", marker="o", markerfacecolor="white", color=BLUE, label="Thinking"),
               Line2D([],[],ls="", marker="o", color=GOLD, label="Answer"),
               Line2D([],[],ls="", marker=".", color="#aaa", label="275 reference roles"),
               Line2D([],[],ls="", marker="*", color="black", label="Released default")]
    axes[0].legend(handles=handles, loc="lower left", fontsize=8, framealpha=.95)
    fig.suptitle("Thinking and answers in the released Qwen persona space", fontsize=14)
    fig.supxlabel("\n".join("    ".join(labels[j:j+5]) for j in range(0,len(labels),5)) + "\nArrows: thinking → answer. Circles: roles; squares: styles; stars: defaults. Percentages refer to the 275-role reference.", fontsize=8.3)
    save(fig, "02_reference_pca")
    rows = [dict(source="experiment", arm=arm, segment=segment, **{f"pc{k+1}":coords[i,s,k] for k in range(3)})
            for i,arm in enumerate(arms) for s,segment in enumerate(SEGMENTS)]
    rows += [dict(source="released_reference", arm=role, segment="reference", **{f"pc{k+1}":background[i,k] for k in range(3)}) for i,role in enumerate(ref["labels"])]
    pd.DataFrame(rows).to_csv(out / "reference_coordinates.csv", index=False)

    # Fig.17's readout is cosine AFTER reference mean subtraction.
    cosines = normalize(means-center) @ basis[:3].T
    ref_cos = normalize(ref["vectors"]-center) @ basis[:3].T
    default_cos = normalize(ref["default"]-center) @ basis[:3].T
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 7), sharey=True, layout="constrained")
    rows=[]
    for k, ax in enumerate(axes):
        for i, arm in enumerate(arms):
            ax.plot(cosines[i,:,k], [i,i], color="#aaa", lw=1)
            ax.scatter(cosines[i,0,k], i, facecolors="white", edgecolors=BLUE, s=40, linewidths=1.4)
            ax.scatter(cosines[i,1,k], i, color=GOLD, s=35)
            published = default_cos[k] if arm == "default" else ref_cos[ref["labels"].index(arm[5:]),k] if arm.startswith("role_") else None
            if published is not None:
                ax.scatter(published, i, marker="x", c=GRAY, s=30)
            for s, segment in enumerate(SEGMENTS):
                rows.append(dict(arm=arm, segment=segment, pc=k+1, centered_cosine=cosines[i,s,k], published_role_cosine=published))
        ax.axvline(0, c=GRAY, lw=.7); ax.set(xlim=(-1,1), title=f"Reference PC{k+1}", xlabel="Cosine after reference centering")
        ax.grid(axis="x", alpha=.15); ax.axhline(9.5, c="#ccc", lw=.8)
    axes[0].set_yticks(range(len(arms)), [a.replace("role_", "").replace("style_", "style: ") for a in arms]); axes[0].invert_yaxis()
    fig.suptitle("Role alignment with the leading reference PCs", fontsize=14)
    fig.legend(handles=handles[:2] + [Line2D([],[],ls="",marker="x",color=GRAY,label="Same role in release")],loc="outside lower center", ncol=3)
    pd.DataFrame(rows).to_csv(out / "pc_cosines.csv",index=False)
    save(fig, "03_pc_alignment")
    captured={}
    roles=[i for i,a in enumerate(arms) if a.startswith("role_")]
    for s,seg in enumerate(SEGMENTS):
        centered = means[roles,s]-means[roles,s].mean(axis=0)
        captured[seg] = float(np.sum((centered @ basis[:3].T)**2)/np.sum(centered**2))
    return dict(reference_axis_cosine=ref["axis_cos"], assistant_pc1_cosine=float(basis[0]@run["assistant"]),
                experiment_role_variance_captured_by_reference_top3=captured)


def plot_tokens(run, out, save):
    scores = run["all_scores"]
    low, high = np.quantile(scores, [.01, .99])
    norm = Normalize(low, high, clip=True); cmap = plt.get_cmap("cividis").copy(); cmap.set_bad("#ddd")
    color = dict(low=float(low), high=float(high), minimum=float(scores.min()), maximum=float(scores.max()),
                 clipped_fraction=float(np.mean((scores<low)|(scores>high))), colormap="cividis",
                 definition="Global 1st–99th percentiles of all measured tokens; display clipping only, raw values unchanged.")
    lookup={(r["prompt_id"],r["seed"],r["arm"]):r for r in run["records"]}
    rows=[]
    fig,axes=plt.subplots(3,2,figsize=(13.2,6.8),layout="constrained")
    for j,q in enumerate(EXAMPLE_QUESTIONS):
        for s,segment in enumerate(SEGMENTS):
            heat=[]; labels=[]
            for arm in EXAMPLE_ARMS:
                r=lookup[q,17,arm]; seg=r["segments"][segment]
                values=[v for g in seg["groups"] for v in g["scores"]]
                binned,counts=bin_tokens(values); heat.append(binned)
                labels.append(f"{arm.replace('role_','').replace('style_','style ')}  (n={len(values)})")
                rows.extend(dict(prompt_id=q,seed=17,arm=arm,segment=segment,bin=i,count=int(counts[i]),mean=v)
                            for i,v in enumerate(binned))
            im=axes[j,s].imshow(np.stack(heat),cmap=cmap,norm=norm,aspect="auto",extent=(0,100,2.5,-.5),interpolation="nearest")
            axes[j,s].set_yticks(range(3),labels);axes[j,s].set(title=f"{q.split('-')[1].title()} · {segment}",xlabel="Position within segment (%)")
    fig.colorbar(im,ax=axes,label="Raw Assistant projection · one scale for every panel",shrink=.85,extend="both")
    fig.suptitle("Matched token trajectories · default, poet, poetic style",fontsize=14)
    fig.supxlabel("Seed 17; fixed examples from the existing balanced review set. Each cell averages a 1%-position bin; empty bins are gray.",fontsize=8.5)
    pd.DataFrame(rows).to_csv(out/"token_heatmap_bins.csv",index=False)
    save(fig,"04_matched_token_heatmaps")

    # A static companion to the full exact-text browser. Excerpt lengths are
    # fixed by token position, not selected around the most extreme scores.
    fig,axes=plt.subplots(3,2,figsize=(14,8.5),layout="constrained")
    excerpt_rows=[]
    for i,arm in enumerate(EXAMPLE_ARMS):
        record=lookup[EXAMPLE_QUESTIONS[0],17,arm]
        for s,segment in enumerate(SEGMENTS):
            seg=record["segments"][segment]; limit=64 if s==0 else 80
            selected=[]; nt=0
            for group in seg["groups"]:
                if nt>=limit: break
                selected.append(group);nt+=len(group["ids"])
            ax=axes[i,s]; col=line=0; width=76
            characters=[]
            for group in selected:
                value=float(np.mean(group["scores"])); bg=cmap(norm(value)); fg="white" if np.mean(bg[:3])<.48 else "black"
                characters.extend((ch,bg,fg) for ch in group["text"])
                excerpt_rows.append(dict(record_id=record["record_id"],segment=segment,indices=group["indices"],text=group["text"],scores=group["scores"]))
            raw_text="".join(ch for ch,_,_ in characters); pointer=0
            wrapper=textwrap.TextWrapper(width=width,replace_whitespace=False,drop_whitespace=False,expand_tabs=False,break_on_hyphens=False)
            for paragraph in raw_text.split("\n"):
                wrapped=wrapper.wrap(paragraph) or [""]
                assert "".join(wrapped)==paragraph
                for line_text in wrapped:
                    for col,ch in enumerate(line_text):
                        stored,bg,fg=characters[pointer];assert ch==stored;pointer+=1
                        ax.text(col,-line,ch,fontfamily="DejaVu Sans Mono",fontsize=7.6,color=fg,
                                bbox=dict(facecolor=bg,edgecolor="none",pad=.9),va="top")
                    line+=1
                if pointer<len(characters):
                    assert characters[pointer][0]=="\n";pointer+=1
            assert pointer==len(characters)
            ax.set_xlim(-1,width+1);ax.set_ylim(-max(line+1,8),1);ax.axis("off")
            ax.set_title(f"{arm.replace('role_','').replace('style_','style ')} · {segment}\nFirst {nt} of {seg['count']} content tokens",loc="left",fontsize=10)
    fig.suptitle("Exact generated text colored by Assistant-axis projection",fontsize=14)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,label="Raw projection · same global scale as the heatmaps",shrink=.7,extend="both")
    fig.supxlabel("Identity question 01, seed 17. Fixed opening excerpts; full transcripts and individual-token scores are in token_viewer.html.",fontsize=8.5)
    (out/"token_text_excerpts.json").write_text(json.dumps(excerpt_rows,ensure_ascii=False,indent=2)+"\n")
    save(fig,"05_colored_token_text")

    payload=dict(questions=run["questions"], arms=run["arms"], seeds=run["seeds"], records=run["records"],color=color,
                 palette=[to_hex(cmap(i/255)) for i in range(256)])
    # Escape HTML script delimiters in untrusted model-generated text.
    raw=json.dumps(payload,ensure_ascii=False,separators=(",",":")).replace("<","\\u003c").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    template=Path(__file__).with_name("token_viewer_template.html").read_text()
    (out/"token_viewer.html").write_text(template.replace("__PAYLOAD__",raw))
    (out/"color_scale.json").write_text(json.dumps(color,indent=2)+"\n")
    return color


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    parser.add_argument("--reference",type=Path,default=Path("data/literature-reference"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    tokenizer=Tokenizer.from_file(str(args.reference/"tokenizer.json"))
    run=load_run(args.run_dir,tokenizer)
    print(f"Verified {len(run['records'])} transcripts and all token/text mappings",flush=True)
    ref=reference_pca(args.reference,run["assistant"])
    print("Reconstructed reference PCA",flush=True)
    figures=[]
    with PdfPages(out/"literature_figures.pdf") as book:
        def save(fig,name):
            for ext in ["png","pdf"]: fig.savefig(out/f"{name}.{ext}",dpi=180,bbox_inches="tight")
            book.savefig(fig,bbox_inches="tight");plt.close(fig);figures.append(name)
        scree=plot_scree(run,ref,out,save)
        geometry=plot_reference(run,ref,out,save)
        colors=plot_tokens(run,out,save)
    summary=dict(run_id=run["manifest"]["run_id"],scree=scree,reference=geometry,color=colors,figures=figures)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    provenance=dict(run_directory=str(args.run_dir.resolve()),run_id=run["manifest"]["run_id"],
                    source_hashes=run["hashes"],reference_release=ref["manifest"],
                    local_inputs={str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name("token_viewer_template.html"),args.reference/"tokenizer.json",args.reference/"reference_layer32.npz",args.reference/"reference_layer32_manifest.json",args.reference/"notebooks__pca.ipynb",args.reference/"assistant_axis__pca.py",args.run_dir/"manifest.json",args.run_dir/"directions.npz",args.run_dir/"questions.jsonl",args.run_dir/"arms.json",args.run_dir/"segment_means.csv"]},
                    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,tokenizers=__import__('tokenizers').__version__),
                    example_selection=dict(questions=EXAMPLE_QUESTIONS,seed=17,arms=EXAMPLE_ARMS,
                        rationale="Reuse existing balanced-review question IDs; show the poet role with its matched style control. Viewer includes all 312 records."),
                    interpretation="Exploratory adaptations. Released 275-role PCA is not an exact reproduction of the paper's reported larger fitting set. No new model inference.")
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    (out/"verification.json").write_text(json.dumps(run["verification"],indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
