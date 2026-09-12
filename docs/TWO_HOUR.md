# Two-hour A40 run: secure E1 before adding E2

Use `configs/a40_two_hour.yaml` after preparing inputs. This is a reduced, single-seed study, not the original complete proposal. It uses the verified published Qwen3-32B axis at layer 32, advice and math only, and retains both orthogonalized role controls, the held-out null, and random control. E3/E4 and their PCs are deferred. The numerical E0 pass rule is unchanged, although three held-out E0 questions rather than five give a less precise sanity check.

## Workload and timing

The profile generates **200 E1 responses**: 50 candidates × two domains × think/no-think, with a target of 40 valid paired survivors per domain. It also generates **49 calibration/E0 responses**: 16 calibration + three questions × (default + ten roles). Fewer than ten available named E0 roles reduces that count. These are 249 requests, not the 160 E1 responses alone. Only 160 retained E1 responses get the main extraction pass. Failed/short/truncated candidates remain auditable; fewer than 40 paired survivors stops the primary run rather than quietly weakening the claim.

E2, if reached, adds eight already-implemented exact-token transplant cells per retained advice prompt: **320 teacher-forced continuations and no new model generation**. It keeps the matched 2×2 content/context cells, scratch and atomic-tag controls. The generation-heavy step-by-step baseline is explicitly deferred. Reducing the number of mechanisms controlled should be the last cut; if E1 is late, stop with E1 rather than oversell three uncontrolled bars. E2 can still take material time, so do not start it without a measured extraction-rate estimate and room to finish.

Two hours is plausible only if CUDA/vLLM, the checkpoint cache and the data are ready, and E0 passes quickly. There is no verified A40 runtime yet. An [A40 has 48 GB memory and NVLink connects pairs](https://www.nvidia.com/en-au/data-center/a40/); eight A40s are not automatically one fast eight-GPU interconnect. Check `nvidia-smi topo -m`. Tensor parallelism communicates every decoder step; topology affects throughput. [vLLM's parallelism documentation](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) distinguishes tensor and pipeline parallelism.

The existing program is **staged, not overlapped**. Generation exits before HF extraction loads. `device_map="auto"` places layers across visible devices; it is not HF tensor parallelism and does not give fourfold speedup. Reuse a known-working four-GPU group for both stages if that is simpler. Assigning generation GPUs 0–3 and extraction GPUs 4–7 below does not itself create concurrency. Do not write a new streaming scheduler under this deadline.

## Before starting the two-hour clock, if possible

Activate the tested environment; prepare `data/prepared/prompts.jsonl` and `roles.jsonl` using the README/data guide; download the pinned model and axis artifacts. Verify actual advice provenance, prompt counts, available memory, and shared checkpoint cache. Do not spend the experiment allocation downloading 60+ GB of weights if it can be done beforehand. No GPU rental or remote launch is performed by this profile.

The existing `persona fetch-assets` imports the published roles and vectors. The axis importer was already tested against the actual released `[64, 5120]` tensors. A load error is a path/format/model compatibility problem to debug; it is not a reason to manufacture a new axis after seeing results.

## Staged commands on one host with eight visible A40s

Run from the repository root. These commands preserve one frozen config and run identity. `time` measures each stage; retain the terminal log. Adapt GPU IDs to the actual allocated topology **before running**.

```bash
source .venv/bin/activate
nvidia-smi
nvidia-smi topo -m

# Minutes 0–30: independent calibration and axis sanity check.
time CUDA_VISIBLE_DEVICES=0,1,2,3 persona worker --config configs/a40_two_hour.yaml --stage generate-gate
time CUDA_VISIBLE_DEVICES=4,5,6,7 persona worker --config configs/a40_two_hour.yaml --stage gate

# Only after E0 passes. Target complete E1 figures by minute 70.
time CUDA_VISIBLE_DEVICES=0,1,2,3 persona worker --config configs/a40_two_hour.yaml --stage generate-experiments
time CUDA_VISIBLE_DEVICES=4,5,6,7 persona worker --config configs/a40_two_hour.yaml --stage extract-e1
time persona worker --config configs/a40_two_hour.yaml --stage analyze
```

At this point the result exists in `runs/a40-two-hour/results.md`, with plots 1, 5 and 10 plus the inexpensive boundary/norm diagnostics. Missing E2 and judge outputs are stated explicitly. The report does not claim a completed mechanism test or full-proposal human review.

If E1 is saved by minute 70 and the measured hook rate permits E2 to finish by minute 100, first preserve the E1 deliverable, then add E2 **without changing the config or regenerating E1**:

```bash
mkdir -p runs/a40-two-hour/e1_snapshot
cp runs/a40-two-hour/results.md runs/a40-two-hour/e1_snapshot/
cp runs/a40-two-hour/summary.json runs/a40-two-hour/e1_snapshot/
cp -R runs/a40-two-hour/figures runs/a40-two-hour/e1_snapshot/
time CUDA_VISIBLE_DEVICES=4,5,6,7 persona worker --config configs/a40_two_hour.yaml --stage extract
time persona worker --config configs/a40_two_hour.yaml --stage analyze
```

The second extraction reuses the E1 projection shards, although it reloads HF weights. An interrupted E2 does not replace the already-written E1 token table before all shards finish. The snapshot keeps the initial report/figures separately. E2 inference remains conditional on one seed and the selected long-CoT advice population. Without step-by-step outputs, do not claim all content/format alternatives are ruled out.

`persona run --config configs/a40_two_hour.yaml` also works, but automatically proceeds to E2; use the staged commands to enforce the clock. There is **no automatic wall-clock cancellation**. The researcher checks the cutoffs using the stage timings.

## Stop rules and interpretation

- By minute 35, E0 must pass. Never bypass the gate. A vector fitted on the same role responses used to validate it is circular. A new axis would require disjoint fitting/validation data and a changed methodological claim; that is not a 20-line fallback under this deadline.
- If E0 fails, report failed calibration. A black-box pilot is an alternative study, not a substitute validating the geometry. A reliable API judge still needs at least 30 real manual calibration labels; the present client runs sequentially and is not a safe last-minute fallback. A small manually coded sample can be explicitly exploratory.
- If E1 is not plotted by minute 70, defer E2. Do not weaken the axis controls to force a mechanism result.
- Reserve minutes 100–120 for reading actual retained CoTs and drafting findings. Read 15 total if that is all time allows, but report the exact count: it does **not** satisfy the proposal's 20-per-domain review requirement. Blank review forms remain incomplete.
- Keep the H1/H2/H3 prediction table in the write-up and mark E3/E4 untested. E1 alone tests a geometric segment difference, not persona identity; unrelated controls determine whether a persona-specific interpretation is even plausible.
- If two hours includes the submission itself, reserve writing time inside those two hours. An extra hour afterward requires an actual extra hour of access/deadline, not a scheduling assumption.

Start the adjacent `WRITEUP_TWO_HOUR.md` now. Fill results only from real saved tables; none of the synthetic smoke values belong in the research write-up.

## Local verification of this reduced path

The updated suite passes **60 tests**. The new staged test finishes and analyzes E1, adds E2 without regenerating transcripts or recomputing E1 shards, and rejects overwriting completed E2 with E1-only output. A separate small synthetic run of this profile produced 10,140 scalar-token rows, five directions and six figures; the artifact verifier passed. This checks the reduced workflow only. No A40 job or pretrained generation was launched from this local workspace.
