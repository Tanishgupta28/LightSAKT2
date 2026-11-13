"""Figures 4--6 layout: genuine seed42 histories; never padded or interpolated."""
import os
os.environ['PAPER_GENERATOR']='generate_training_curves.py'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import ASSETS, DATASETS, MODELS, COLORS, SHORT, DISPLAY, FOLDERS, save_figure


def main():
    source='source_data/epoch_records/completed_final_epochs.csv'
    data=pd.read_csv(ASSETS/source)
    for dataset in DATASETS:
        fig,axes=plt.subplots(1,2,figsize=(9.2,3.0))
        for model in MODELS:
            group=data[(data.dataset==dataset)&(data.model==model)&(data.seed==42)].sort_values('epoch')
            if group.empty:continue
            assert group.epoch.tolist()==list(range(1,len(group)+1))
            label=SHORT.get(model,model)
            if model.startswith('GKT'):label+=' [1 selected seed]'
            for ax,metric in zip(axes,['training_objective','validation_auc']):
                ax.plot(group.epoch,group[metric],color=COLORS[model],label=label)
        title=DISPLAY.get(dataset,dataset)
        for ax,subtitle,ylabel in zip(axes,['Training objective','Validation AUC'],['Objective loss','AUC']):
            ax.set(title=f'{title} — {subtitle}',xlabel='Epoch',ylabel=ylabel);ax.grid(True)
        handles,labels=axes[1].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.16),ncol=4,frameon=True)
        fig.tight_layout()
        save_figure(fig,f'graphs/{FOLDERS[dataset]}/training_performance_seed42',[source],
          f'{title}, actual seed42 training only. Every curve ends at its logged final epoch; no smoothing, extension, interpolation or five-seed averaging. '
          'DKT+ objective includes reconstruction and waviness penalties, so loss magnitudes are not directly comparable to pure BCE. '
          'Two ASSIST2009 GKT curves represent only one validation-selected seed each; five-seed groups remain incomplete. Missing model curves are omitted.')


if __name__=='__main__':main()
