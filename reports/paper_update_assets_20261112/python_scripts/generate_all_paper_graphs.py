"""Regenerate every numerical graph from the portable genuine input extracts."""
import subprocess
import sys
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    for script in ['generate_training_curves.py','generate_model_comparison.py',
                   'generate_ablation_graphs.py','generate_parameter_graph.py']:
        subprocess.run([sys.executable,'-B',str(root/script)],check=True)


if __name__=='__main__':main()
