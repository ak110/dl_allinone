
def test_run(data_dir):
    import subprocess
    r = subprocess.run(['jupyter', 'nbconvert', '--execute', '--stdout', data_dir / 'jupyter.ipynb'], stdout=subprocess.PIPE)
    assert r.returncode == 0
