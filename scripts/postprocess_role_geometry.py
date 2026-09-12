"""Exploratory E4 geometry from saved segment means; no model inference.

All observations are matched by question, arm and seed. PCA is descriptive;
full-dimensional norms and seed reproducibility accompany the projection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def fit_pca(x):
    center = x.mean(axis=0)
    _, singular, basis = np.linalg.svd(x - center, full_matrices=False)
    for row in basis:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    assert np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-9)
    return center, basis, singular**2 / np.sum(singular**2)


def spread(x):
    """RMS distance of fixed role centroids from their own segment centroid."""
    return np.sqrt(np.mean(np.sum((x - x.mean(axis=0))**2, axis=-1)))


def cosine_matrix(x):
    norms = np.linalg.norm(x, axis=1)
    assert np.all(norms > 1e-10)
    unit = x / norms[:, None]
    similarity = np.clip(unit @ unit.T, -1, 1)
    assert np.allclose(np.diag(similarity), 1)
    return similarity, unit


def save_figure(fig, out, name):
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"{name}.{ext}", dpi=165, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run_dir, args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    assert not manifest["synthetic"]
    questions = [json.loads(s) for s in (root / "questions.jsonl").read_text().splitlines()]
    qids = [q["prompt_id"] for q in questions]
    domains = np.array([q["domain"] for q in questions])
    arms = [a["arm"] for a in json.loads((root / "arms.json").read_text())]
    seeds = manifest["spec"]["seeds"]
    segments = ["think", "answer"]
    role_indices = [i for i, a in enumerate(arms) if a.startswith("role_")]
    default = arms.index("default")
    nondefault = [i for i in range(len(arms)) if i != default]
    names = [arms[i].replace("role_", "").replace("style_", "style: ") for i in nondefault]
    directions = np.load(root / "directions.npz")
    direction_names = json.loads(directions["metadata"].item())["direction_names"]
    assistant = directions["directions"][direction_names.index("assistant")]
    dim = len(assistant)
    data = np.full((len(qids), len(seeds), len(arms), 2, dim), np.nan)
    counts = np.zeros(data.shape[:-2], dtype=int)
    source_hashes = {}
    for path in sorted((root / "transcripts").glob("*.json")):
        record = json.loads(path.read_text())
        if record["phase"] != "main":
            continue
        idx = (qids.index(record["prompt_id"]), seeds.index(record["seed"]), arms.index(record["arm"]))
        counts[idx] += 1
        vector_path = root / "segment_vectors" / (record["record_id"] + ".npz")
        source_hashes[vector_path.name] = hashlib.sha256(vector_path.read_bytes()).hexdigest()
        vectors = np.load(vector_path)
        data[idx] = np.stack([vectors[s] for s in segments])
    assert np.all(counts == 1) and np.isfinite(data).all()
    # Directly verify the new analysis against the independently audited scalars.
    scalar = pd.read_csv(root / "segment_means.csv")
    for row in scalar[scalar.segment.isin(segments)].itertuples():
        v = data[qids.index(row.prompt_id), seeds.index(row.seed), arms.index(row.arm), segments.index(row.segment)]
        assert abs(v @ assistant - row.dot_assistant) < 1e-4

    centroids = data.mean(axis=(0, 1))
    deltas = data - data[:, :, default:default+1, :, :]
    changes = deltas.mean(axis=(0, 1))
    np.savez_compressed(out / "centroids.npz", centroids=centroids, changes=changes,
                        arms=np.array(arms), segments=np.array(segments))
    center, basis, fractions = fit_pca(centroids[role_indices, 1])
    _, joint_basis, joint_fractions = fit_pca(changes[role_indices].reshape(-1, dim))
    np.savez_compressed(out / "pca_frames.npz", answer_center=center, answer_components=basis,
                        answer_variance_fractions=fractions, joint_shift_components=joint_basis,
                        joint_shift_variance_fractions=joint_fractions)
    colors = [plt.get_cmap("tab20")(i) for i in range(len(nondefault))]
    coords = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), layout="constrained")
    panels = [("Absolute segment centroids", (centroids - center) @ basis[:2].T),
              ("Matched shifts from segment default", changes @ basis[:2].T),
              ("Matched shifts · joint-PCA sensitivity", changes @ joint_basis[:2].T)]
    for panel, (title, xy) in enumerate(panels):
        ax = axes[panel]
        for i, color, label in zip(nondefault, colors, names):
            marker = "s" if arms[i].startswith("style_") else "o"
            ax.plot(xy[i, :, 0], xy[i, :, 1], color=color, alpha=.55, lw=1)
            ax.scatter(*xy[i, 0], facecolors="none", edgecolors=[color], marker=marker, s=65, linewidths=1.7)
            ax.scatter(*xy[i, 1], c=[color], marker=marker, s=40)
            # A fixed index makes close roles legible without outcome-tuned labels.
            offset = {"role_editor":(8,-25), "role_translator":(10,12), "role_doctor":(8,-12),
                      "role_composer":(-15,-20), "role_optimist":(-12,24), "role_alien":(-30,10)}.get(arms[i],(4,3))
            ax.annotate(str(nondefault.index(i)+1), xy[i, 1], xytext=offset, textcoords="offset points", fontsize=8,
                        arrowprops=dict(arrowstyle="-",color=color,lw=.5) if offset!=(4,3) else None)
        for s in range(2):
            ax.scatter(*xy[default, s], marker="*", s=150, facecolors="black" if s else "white", edgecolors="black", zorder=4)
        fs = joint_fractions if panel == 2 else fractions
        ax.set(title=title, xlabel=f"PC1 ({100*fs[0]:.1f}% of fitting variance)",
               ylabel=f"PC2 ({100*fs[1]:.1f}% of fitting variance)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.15)
        for a, arm in enumerate(arms):
            for s, segment in enumerate(segments):
                coords.append(dict(panel=panel, arm=arm, segment=segment, x=xy[a,s,0], y=xy[a,s,1]))
    handles = [plt.Line2D([], [], color=c, marker="s" if arms[i].startswith("style_") else "o", ls="", label=f"{k+1}. {label}")
               for k, (i, c, label) in enumerate(zip(nondefault, colors, names))]
    handles += [plt.Line2D([],[],marker="*",color="black",markerfacecolor="white",ls="",label="Default thinking",markersize=10),
                plt.Line2D([],[],marker="*",color="black",ls="",label="Default answer",markersize=10)]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=8)
    fig.suptitle("Role geometry · hollow: thinking, filled: answer; stars: defaults", fontsize=13)
    save_figure(fig, out, "01_pca_geometry")
    pd.DataFrame(coords).to_csv(out / "pca_coordinates.csv", index=False)

    rows = []
    for i in nondefault:
        t, a = changes[i]
        per_seed = deltas[:, :, i].mean(axis=0)
        rows.append(dict(arm=arms[i], think_norm=np.linalg.norm(t), answer_norm=np.linalg.norm(a),
                         norm_ratio=np.linalg.norm(t)/np.linalg.norm(a),
                         same_role_segment_cosine=t@a/np.linalg.norm(t)/np.linalg.norm(a),
                         think_seed_cosine=np.dot(per_seed[0,0],per_seed[1,0])/np.linalg.norm(per_seed[0,0])/np.linalg.norm(per_seed[1,0]),
                         answer_seed_cosine=np.dot(per_seed[0,1],per_seed[1,1])/np.linalg.norm(per_seed[0,1])/np.linalg.norm(per_seed[1,1])))
    norms = pd.DataFrame(rows)
    norms.to_csv(out / "full_space_role_effects.csv", index=False)
    seed_rows = []
    fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
    for s, label, color, offset in [(0,"Thinking","#2962a3",-.13),(1,"Answer","#b58a28",.13)]:
        y = np.arange(len(nondefault)) + offset
        values = np.linalg.norm(changes[nondefault, s], axis=-1)
        ax.scatter(values, y, color=color, s=42, label=label)
        for j, seed in enumerate(seeds):
            per_seed = np.linalg.norm(deltas[:, j, nondefault, s].mean(axis=0), axis=-1)
            ax.scatter(per_seed, y, color=color, marker="x", s=20, alpha=.45)
            seed_rows.extend([dict(arm=arms[i],segment=segments[s],seed=seed,norm=v) for i,v in zip(nondefault,per_seed)])
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.15); ax.legend(loc="lower right")
    ax.set(xlabel="Norm of mean matched role − default activation vector",
           title="Role effects in all 5,120 dimensions\nDots: seeds pooled; faint crosses: individual seeds")
    save_figure(fig, out, "02_full_space_effects")
    pd.DataFrame(seed_rows).to_csv(out / "seed_effect_norms.csv", index=False)

    # Cluster full-dimensional, unit-normalized matched effect vectors. No 2-D
    # embedding is used, and no number of clusters is selected.
    vectors = changes[nondefault].reshape(-1, dim)
    labels = [f"{name} · {segment}" for name in names for segment in ["T", "A"]]
    similarity, unit = cosine_matrix(vectors)
    hierarchy = linkage(pdist(unit, metric="euclidean"), method="average")
    order = leaves_list(hierarchy)
    pd.DataFrame(similarity, index=labels, columns=labels).to_csv(out / "effect_cosine_matrix.csv")
    pd.DataFrame(hierarchy, columns=["left", "right", "distance", "leaves"]).to_csv(out / "linkage.csv", index=False)
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[24, 1], height_ratios=[1, 7])
    top = fig.add_subplot(gs[0,0]); ax = fig.add_subplot(gs[1,0]); bar = fig.add_subplot(gs[1,1])
    dendrogram(hierarchy, ax=top, no_labels=True, color_threshold=0, above_threshold_color="#666")
    top.set_xticks([]); top.set_ylabel("Distance",fontsize=8)
    top.set_title("Similarity of matched role effects · descriptive hierarchy, no selected cluster count")
    im = ax.imshow(similarity[np.ix_(order,order)], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)),[labels[i] for i in order],rotation=90,fontsize=7)
    ax.set_yticks(range(len(labels)),[labels[i] for i in order],fontsize=7)
    fig.colorbar(im,cax=bar,label="Cosine between full-dimensional effect vectors")
    fig.canvas.draw()
    assert np.allclose([top.get_position().x0,top.get_position().x1],
                       [ax.get_position().x0,ax.get_position().x1],atol=1e-5), "Dendrogram columns must align with matrix columns"
    save_figure(fig,out,"03_similarity_hierarchy")

    matrices = []
    for j, seed in enumerate(seeds):
        mat,_ = cosine_matrix(deltas[:, j, nondefault].mean(axis=0).reshape(-1, dim))
        matrices.append(mat)
        pd.DataFrame(mat,index=labels,columns=labels).to_csv(out/f"effect_cosine_seed_{seed}.csv")
    triangle = np.triu_indices(len(labels),1)
    seed_similarity = float(spearmanr(matrices[0][triangle],matrices[1][triangle]).statistic)
    # Ordinary edge-wise p-values would incorrectly count dependent matrix cells.
    distances = [pdist(centroids[role_indices,s]) for s in range(2)]
    role_distance_agreement = float(spearmanr(*distances).statistic)
    segment_distance_seed_agreement = {segments[s]:float(spearmanr(
        pdist(data[:,0,role_indices,s].mean(axis=0)),pdist(data[:,1,role_indices,s].mean(axis=0))).statistic) for s in range(2)}
    without_axis = centroids[role_indices] - (centroids[role_indices] @ assistant)[...,None] * assistant
    orthogonal_spreads = [float(spread(without_axis[:,s])) for s in range(2)]
    leave_one_out = []
    for q in range(len(qids)):
        c = data[np.arange(len(qids))!=q].mean(axis=(0,1))
        leave_one_out.append(dict(excluded_prompt=qids[q],cloud_rms_ratio=float(spread(c[role_indices,0])/spread(c[role_indices,1])),
                                  role_distance_spearman=float(spearmanr(pdist(c[role_indices,0]),pdist(c[role_indices,1])).statistic)))
    pd.DataFrame(leave_one_out).to_csv(out/"leave_one_question_out.csv",index=False)

    by_domain = []
    pairs = pd.read_csv(root / "paired_question_effects.csv")
    interactions = pairs[pairs.metric.eq("dot_assistant") & pairs.policy.eq("all_content")
                         & pairs.reference_arm.eq("default") & pairs.contrast.eq("segment_interaction")]
    domain_order = ["identity","social","reasoning"]
    scalar_domains = interactions.groupby(["arm","domain"]).value.mean().unstack().loc[[arms[i] for i in nondefault],domain_order]
    ratios = np.empty((len(nondefault),len(domain_order)))
    for d,domain in enumerate(domain_order):
        local = deltas[domains==domain].mean(axis=(0,1))
        for j,i in enumerate(nondefault):
            nt,na = np.linalg.norm(local[i],axis=1)
            ratios[j,d] = nt/na
            by_domain.append(dict(arm=arms[i],domain=domain,n_questions=int((domains==domain).sum()),think_norm=nt,answer_norm=na,norm_ratio=nt/na,
                                  assistant_interaction=float(scalar_domains.loc[arms[i],domain])))
    pd.DataFrame(by_domain).to_csv(out/"domain_effects.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,7),layout="constrained")
    for ax,values,title,limit,cmap in [(axes[0],scalar_domains.to_numpy(),"Assistant-axis interaction",None,"RdBu_r"),
                                      (axes[1],ratios,"Full-space effect norm · thinking / answer",(0,2),"viridis")]:
        low,high=(-np.max(abs(values)),np.max(abs(values))) if limit is None else limit
        im=ax.imshow(values,vmin=low,vmax=high,cmap=cmap,aspect="auto")
        ax.set_yticks(range(len(names)),names);ax.set_xticks(range(3),domain_order)
        ax.set_title(title,fontsize=11)
        for i,row in enumerate(values):
            for j,value in enumerate(row):
                fraction=(value-low)/(high-low)
                color="white" if (cmap=="RdBu_r" and (fraction<.2 or fraction>.8)) or (cmap=="viridis" and fraction<.4) else "black"
                ax.text(j,i,f"{value:.2f}",ha="center",va="center",color=color,fontsize=9)
        fig.colorbar(im,ax=ax,shrink=.65)
    fig.suptitle("Task dependence · four fixed questions per domain; descriptive means")
    save_figure(fig,out,"04_domain_comparison")

    spreads = np.array([spread(centroids[role_indices,s]) for s in range(2)])
    seed_spreads = {str(seed):[float(spread(data[:,j,role_indices,s].mean(axis=0))) for s in range(2)] for j,seed in enumerate(seeds)}
    summary = dict(run_id=manifest["run_id"],post_hoc=True,n_questions=len(qids),seeds=seeds,
                   n_roles=len(role_indices),n_style_arms=len(nondefault)-len(role_indices),n_vectors=int(np.prod(data.shape[:-1])),
                   answer_pca_variance_fractions=fractions.tolist(),joint_shift_pca_variance_fractions=joint_fractions.tolist(),
                   role_cloud_rms_thinking=float(spreads[0]),role_cloud_rms_answer=float(spreads[1]),
                   role_cloud_rms_ratio=float(spreads[0]/spreads[1]),seed_role_cloud_rms=seed_spreads,
                   role_pair_distance_spearman=role_distance_agreement,effect_similarity_seed_spearman=seed_similarity,
                   role_distance_seed_agreement=segment_distance_seed_agreement,
                   role_cloud_rms_without_assistant=orthogonal_spreads,
                   role_cloud_rms_ratio_without_assistant=orthogonal_spreads[0]/orthogonal_spreads[1],
                   interpretation="Descriptive fixed-role geometry. No independent matrix-cell p-values, chosen cluster count, or persona identity classification.",
                   source_vector_sha256=source_hashes)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!='source_vector_sha256'},indent=2))


if __name__ == "__main__":
    main()
