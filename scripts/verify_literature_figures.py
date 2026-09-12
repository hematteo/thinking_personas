"""Independent numerical verification of the exported literature figure bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args();out=args.output
    provenance=json.loads((out/"provenance.json").read_text())
    root=Path(provenance["run_directory"])
    data=np.load(out/"reference_pca.npz")
    rows=pd.read_csv(out/"scree_values.csv")
    summary=json.loads((out/"summary.json").read_text())
    arms=data["arms"].tolist();roles=[i for i,a in enumerate(arms) if a.startswith("role_")]
    checks={}
    for label,points in [("think",data["experiment_centroids"][roles,0]),
                         ("answer",data["experiment_centroids"][roles,1]),
                         ("released_reference",data["role_vectors"])]:
        centered=points-points.mean(0)
        # Eigenspectrum of the smaller Gram matrix, independent of the SVD
        # used by the plotting code. Its trace is the total population variance.
        gram=centered@centered.T/len(points)
        eigen=np.linalg.eigvalsh(gram)[::-1]
        actual=rows[rows.dataset.eq(label)].sort_values("pc").variance.to_numpy()
        assert np.allclose(eigen[:len(actual)],actual,atol=1e-8,rtol=1e-8)
        assert np.isclose(np.trace(gram),actual.sum(),atol=1e-8)
        checks[f"{label}_spectrum_verified_with_gram_eigenvalues"]=True
    coordinates=pd.read_csv(out/"reference_coordinates.csv")
    max_error=0.
    for row in coordinates[coordinates.source.eq("experiment")].itertuples():
        vector=data["experiment_centroids"][arms.index(row.arm),["think","answer"].index(row.segment)]
        for k in range(3):
            score=sum((vector-data["center"])*data["components"][k])
            max_error=max(max_error,abs(score-getattr(row,f"pc{k+1}")))
    assert max_error<1e-8
    checks["reference_coordinate_max_error"]=max_error
    cosine_rows=pd.read_csv(out/"pc_cosines.csv")
    for row in cosine_rows.itertuples():
        vector=data["experiment_centroids"][arms.index(row.arm),["think","answer"].index(row.segment)]-data["center"]
        expected=np.dot(vector,data["components"][row.pc-1])/np.sqrt(np.dot(vector,vector))
        assert abs(expected-row.centered_cosine)<1e-10
    checks["all_centered_pc_cosines_verified"]=True
    html=(out/"token_viewer.html").read_text()
    payload=re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S).group(1)
    assert "<" not in payload, "Model text must not contain literal HTML delimiters in script data"
    exported=json.loads(payload)
    assert len(exported["records"])==312
    assert len({(r["prompt_id"],r["seed"],r["arm"]) for r in exported["records"]})==312
    means=pd.read_csv(root/"segment_means.csv").set_index(["record_id","segment"])
    groups=0
    for record in exported["records"]:
        source=json.loads((root/"transcripts"/f"{record['record_id']}.json").read_text())
        tokens=pd.read_csv(root/"token_shards"/f"{record['record_id']}.csv.gz")
        for segment,value in record["segments"].items():
            positions=[i for g in value["groups"] for i in g["indices"]]
            ids=[i for g in value["groups"] for i in g["ids"]]
            scores=[i for g in value["groups"] for i in g["scores"]]
            actual=tokens[tokens.segment.eq(segment)].sort_values("token_idx")
            assert positions==actual.token_idx.tolist()
            assert ids==actual.token_id.tolist()
            assert np.array_equal(scores,actual.dot_assistant.to_numpy())
            assert len(ids)==value["count"]
            assert "".join(g["text"] for g in value["groups"]).strip()==source[f"{segment}_text"].strip()
            assert np.isclose(np.mean(scores),means.loc[(record["record_id"],segment),"dot_assistant"],atol=1e-5)
            groups+=len(value["groups"])
    checks.update(all_624_exported_text_segments_and_scores_verified=True,display_groups=groups,
                  script_data_html_delimiters_escaped=True)
    bins=pd.read_csv(out/"token_heatmap_bins.csv")
    for keys,group in bins.groupby(["prompt_id","seed","arm","segment"]):
        q,seed,arm,segment=keys
        record=next(r for r in exported["records"] if (r["prompt_id"],r["seed"],r["arm"])==(q,seed,arm))
        segment_data=record["segments"][segment]
        assert group["count"].sum()==segment_data["count"]
        assert np.isclose((group["mean"]*group["count"]).sum()/group["count"].sum(),segment_data["mean"],atol=1e-5)
    checks["all_18_heatmap_rows_reconcile_to_full_segment_means"]=True
    script=re.findall(r'<script>(.*?)</script>',html,re.S)[0]
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/"viewer.js";path.write_text(script)
        subprocess.run(["node","--check",str(path)],check=True,capture_output=True)
    checks["viewer_javascript_syntax_check"]=True
    checks["browser_visual_verification"]="Unavailable: browser URL policy blocks local file URLs; no browser workaround attempted."
    for filename,expected in provenance["local_inputs"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected
    checks["local_input_hashes_verified"]=True
    checks["figures"]=summary["figures"]
    (out/"independent_verification.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    main()
