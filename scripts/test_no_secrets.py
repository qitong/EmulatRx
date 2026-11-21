import subprocess, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Pattern assembled from parts so this test file does not match itself.
PATTERN = "sk-" + "proj-|" + "sk-" + "[A-Za-z0-9]{20,}"
def test_no_hardcoded_keys():
    r = subprocess.run(["grep","-rnE",PATTERN,"--include=*.py","."],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == "", f"hardcoded key(s) found:\n{r.stdout}"
