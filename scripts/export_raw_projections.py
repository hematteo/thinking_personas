"""Export unaveraged linear projections without inference or inclusion filtering.

Raw here means h_t dot a_unit: no activation centering or norm division. These
are scalar readouts at the saved layer, not the complete residual vectors.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from persona_dynamics.io import file_hash


def export(run_dir, output_dir):
    root, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    if out == root or root in out.parents:
        raise ValueError('Export outside the frozen run directory')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory; existing exports are immutable')
    source = root/'tokens.csv.gz'
    tokens = pd.read_csv(source, low_memory=False, float_precision='round_trip')
    required = {'record_id', 'prompt_id', 'token_idx', 'token_id', 'segment', 'layer', 'resid_norm', 'dot_assistant'}
    if missing := required.difference(tokens):
        raise ValueError(f'Missing raw measurement columns: {sorted(missing)}')
    if tokens.duplicated(['record_id', 'token_idx']).any():
        raise ValueError('Duplicate measured token identities')
    dots = [c for c in tokens if c.startswith('dot_')]
    if not np.isfinite(tokens[dots+['resid_norm']].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite raw measurements')
    # Preserve identities, positions and inclusion flags, but never apply the
    # flags. All saved measured boundary/first-five/content rows survive.
    columns = [c for c in tokens if not c.startswith('cos_') and c != 'centered_norm']
    raw = tokens[columns]
    out.mkdir(parents=True, exist_ok=True)
    path = out/'raw_token_projections.csv.gz'
    raw.to_csv(path, index=False, compression={'method':'gzip', 'mtime':0})
    recovered = pd.read_csv(path, low_memory=False, float_precision='round_trip')
    pd.testing.assert_frame_equal(raw, recovered, check_exact=True)
    config_path = root/'config.json'
    manifest = {
        'source_run':str(root), 'source_tokens_sha256':file_hash(source),
        'export_sha256':file_hash(path), 'script_sha256':file_hash(__file__),
        'rows':len(raw), 'records':int(raw.record_id.nunique()), 'projection_columns':dots,
        'formula':'dot_direction(t) = h_t @ unit_direction',
        'activation_centering':False, 'activation_norm_division':False,
        'token_averaging':False, 'additional_row_filtering':False,
        'roundtrip_exact':True, 'config':json.loads(config_path.read_text()) if config_path.exists() else None,
        'limitations':[
            'Contains every previously measured row, not every generated or prompt token.',
            'Original cohort selection and tokenizer span choices remain in force.',
            'primary_include and first5 are metadata, not filters applied to this export.',
            'Full hidden vectors are not saved; projecting new directions requires another extraction pass.',
            'A signed activation coordinate is not a probability or calibrated measure of persona.'
        ]
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=export(args.run_dir,args.output)
    print(json.dumps({k:result[k] for k in ('rows','records','projection_columns','roundtrip_exact')},indent=2))


if __name__=='__main__':
    main()
