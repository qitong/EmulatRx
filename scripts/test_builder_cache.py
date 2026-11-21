import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for k in ["DB_HOST","DB_PORT","DB_USER","DB_PASSWORD","DB_NAME"]:
    os.environ.pop(k, None)
from dataframe_utils import DataframeBuilder

def test_cache_returns_without_db():
    b = DataframeBuilder(trialID="NCT00000099", trial_info={}, dataset="synthetic",
                         user=None, password=None, host=None, port=None, database=None,
                         root_dir=os.path.join(ROOT, "trials"))
    df, cd = b.get_dataframe()
    assert cd["patient_id"] == "stay_id"
    assert cd["treatment_cols"] == "treatment"
    assert len(df) > 0
    assert "treatment" in df.columns and "event" in df.columns and "duration_day" in df.columns
