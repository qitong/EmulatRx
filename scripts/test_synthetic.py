import json, subprocess, os, sys, math, pandas as pd
PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIAL = "NCT00000099"
CACHE = os.path.join(ROOT, "trials", TRIAL, "dataframes")

def _gen():
    subprocess.run([PY, os.path.join(ROOT,"scripts","generate_synthetic_oncology.py"),
                    "--trial-id", TRIAL, "--no-postgres"], check=True, cwd=ROOT)

def test_generator_emits_contract():
    _gen()
    df = pd.read_csv(os.path.join(CACHE,"dataframe.csv"))
    cd = json.load(open(os.path.join(CACHE,"col_dict.json")))
    assert list(df.columns)[:6] == ["stay_id","subject_id","treatment","treatment_time_delta","event","duration_day"]
    assert cd["patient_id"]=="stay_id" and cd["treatment_cols"]=="treatment"
    assert cd["outcome_cols"]==["event"] and cd["duration_cols"]==["duration_day"]
    assert set(df["treatment"].unique()) <= {0,1} and df["event"].isin([0,1]).all()
    assert (df["duration_day"]>0).all()
    assert list(df.columns)[6:] == cd["covariate_cols"]

def test_hazard_ratio_recovered():
    from lifelines import CoxPHFitter
    df = pd.read_csv(os.path.join(CACHE,"dataframe.csv"))
    cph = CoxPHFitter().fit(df[["duration_day","event","treatment"]], "duration_day","event")
    ci = cph.confidence_intervals_.loc["treatment"]
    lo, hi = math.exp(ci.iloc[0]), math.exp(ci.iloc[1])
    assert lo <= 0.65 <= hi, f"HR CI [{lo:.2f},{hi:.2f}] must cover target 0.65"

def test_postgres_optional():
    # --postgres with NO DB env available must still exit 0 (non-fatal skip)
    # and always leave the CSV cache in place.
    env = os.environ.copy()
    for k in ["DB_USER","DB_PASSWORD","DB_HOST","DB_PORT","DB_NAME"]:
        env.pop(k, None)
    proc = subprocess.run(
        [PY, os.path.join(ROOT,"scripts","generate_synthetic_oncology.py"),
         "--trial-id", TRIAL, "--postgres"],
        cwd=ROOT, env=env)
    assert proc.returncode == 0
    assert os.path.exists(os.path.join(CACHE,"dataframe.csv"))
