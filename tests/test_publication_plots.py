from pathlib import Path
import matplotlib.pyplot as plt

from scripts.analyze import save_figure


def test_unchanged_observations_preserve_all_publication_bytes_and_modified_data_updates(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    def draw(last):
        plt.figure();plt.plot([1,2],[.7,last]);plt.xlabel('Epoch');plt.ylabel('Observed validation AUC');save_figure('observations')
    draw(.8)
    paths=[Path('results/figures/observations.'+ext) for ext in ['png','svg','pdf']]
    before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}
    draw(.8)
    assert before=={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}
    draw(.9)
    assert all(p.read_bytes()!=before[str(p)][0] for p in paths)
