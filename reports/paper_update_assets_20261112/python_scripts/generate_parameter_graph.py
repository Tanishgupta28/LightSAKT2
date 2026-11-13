"""Figure7 horizontal style using actual instantiated, strictly restored checkpoint counts."""
import os
os.environ['PAPER_GENERATOR']='generate_parameter_graph.py'
import pandas as pd
import matplotlib.pyplot as plt
from common import ASSETS, DATASETS, MODELS, SHORT, DISPLAY, COLORS, csv, save_figure


def main():
    source='source_data/verified_metric_references/individual_seed_metrics.csv'
    frame=pd.read_csv(ASSETS/source);rows=[]
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.0))
    for ax,d in zip(axes,DATASETS):
        for i,m in enumerate(MODELS):
            group=frame[(frame.dataset==d)&(frame.model==m)]
            if len(group):
                assert group.parameters.nunique()==1
                value=int(group.parameters.iloc[0]);rows.append({'dataset':d,'model':m,'parameters':value,'checkpoint_count':len(group)})
                ax.barh(i,value/1000,color=COLORS[m],height=.7)
                ax.text(value/1000+.02,i,f'{value:,}'+(' (1 checkpoint)' if len(group)==1 else ''),va='center',fontsize=6)
            else:ax.text(.01,i,'PENDING',va='center',fontsize=6)
        ax.set_yticks(range(9),[SHORT.get(m,m) for m in MODELS]);ax.set_ylim(8.6,-.6)
        ax.set(xlabel='Trainable parameters (thousands)',title=DISPLAY.get(d,d));ax.grid(axis='x');ax.margins(x=.23)
    fig.tight_layout()
    csv(ASSETS/'tables/training_configurations/parameter_counts.csv',rows)
    save_figure(fig,'graphs/parameters/model_parameters',[source],
      'Actual parameter counts from instantiated, strictly restored configurations. Vocabulary/dimensions differ by dataset and model. '
      'GKT count1 is a single selected checkpoint count, not a five-seed performance aggregate. No arbitrary num_q=100 standardization or inferred counts for absent checkpoints. Difficulty adds one trainable scalar; external fitted statistics are additional uncounted estimator capacity. '
      'Parameter totals alone do not establish matched memory/runtime efficiency.')


if __name__=='__main__':main()
