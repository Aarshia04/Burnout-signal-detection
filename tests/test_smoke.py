import pandas as pd
from src.modeling import get_feature_cols

def test_get_feature_cols():
    df = pd.DataFrame({
        "student_key":["a"],
        "id_student":[1],
        "code_module":["AAA"],
        "code_presentation":["2013J"],
        "final_result":["Pass"],
        "burnout_label":[0],
        "total_clicks":[10],
        "active_days":[3],
    })
    cols = get_feature_cols(df)
    assert "total_clicks" in cols
    assert "burnout_label" not in cols
