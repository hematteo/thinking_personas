import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('raw_export',Path(__file__).parents[1]/'scripts/export_raw_projections.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_raw_export_preserves_boundary_first_tokens_and_exact_scores(tmp_path):
    source=tmp_path/'run';source.mkdir()
    frame=pd.DataFrame(dict(record_id=['r']*3,prompt_id=['p']*3,token_idx=[0,2,8],token_id=[1,2,3],
        segment=['boundary','answer','answer'],layer=[32]*3,primary_include=[False,False,True],
        resid_norm=[2.,3.,4.],centered_norm=[1.,2.,3.],cos_assistant=[.1,.2,.3],
        dot_assistant=[np.nextafter(1.,2.),-123.45678901234567,0.],dot_null=[3.,4.,5.]))
    frame.to_csv(source/'tokens.csv.gz',index=False)
    before=(source/'tokens.csv.gz').read_bytes()
    manifest=module.export(source,tmp_path/'raw')
    result=pd.read_csv(tmp_path/'raw/raw_token_projections.csv.gz',float_precision='round_trip')
    assert manifest['rows']==3 and manifest['roundtrip_exact']
    assert not manifest['token_averaging'] and not manifest['additional_row_filtering']
    np.testing.assert_array_equal(result.dot_assistant,frame.dot_assistant)
    assert result.primary_include.tolist()==[False,False,True]
    assert 'cos_assistant' not in result and 'centered_norm' not in result
    assert (source/'tokens.csv.gz').read_bytes()==before
    with pytest.raises(ValueError,match='immutable'):
        module.export(source,tmp_path/'raw')
    with pytest.raises(ValueError,match='outside'):
        module.export(source,source/'export')
