"""Conceptual figures only; no toy values are reported as experimental results."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scripts.analyze import save_figure


def box(ax,x,y,text,width=2.8,height=.75,color='#e3edf8'):
    ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,boxstyle='round,pad=0.05',facecolor=color,edgecolor='#345678'))
    ax.text(x,y,text,ha='center',va='center',fontsize=9)


def arrow(ax,a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#345678','lw':1.5})


def main():
    fig,ax=plt.subplots(figsize=(9,3))
    illustrative=np.array([[1,0,0,0,1,1,0,1,1],[0,0,0,0,1,1,0,0,1],[1,1,0,1,1,1,0,1,1]])
    ax.imshow(illustrative,cmap='RdYlGn',vmin=0,vmax=1)
    ax.set_yticks(range(3),['Student 1','Student 2','Student 3']);ax.set_xticks(range(9),[f'q{i+1}' for i in range(9)])
    for i in range(3):
        for j in range(9):ax.text(j,i,'correct' if illustrative[i,j] else 'incorrect',ha='center',va='center',fontsize=7)
    ax.set_title('Illustrative learning sequence — conceptual, not measured mastery');save_figure('Figure_1_conceptual')
    fig,ax=plt.subplots(figsize=(10,4));ax.set_xlim(0,12);ax.set_ylim(0,5);ax.axis('off')
    nodes=[(2,3.5,'Assessment response\nobserved'),(6,3.5,'Update chronological\ninteraction history'),(10,3.5,'Predict correctness\nfor next question'),(10,1,'Teacher reviews\nprobability and evidence'),(6,1,'Adapt practice\nand feedback'),(2,1,'Choose next question\nidentity available')]
    for x,y,label in nodes:box(ax,x,y,label)
    for i in range(len(nodes)):arrow(ax,nodes[i][:2],nodes[(i+1)%len(nodes)][:2])
    ax.set_title('Conceptual formative assessment cycle; effectiveness requires separate validation');save_figure('Figure_2_conceptual')
    fig,ax=plt.subplots(figsize=(11,4));ax.set_xlim(0,13);ax.set_ylim(0,5);ax.axis('off')
    for x,y,label in [(2,3.5,'BKT\nprobabilistic states'),(6,3.5,'DKT / DKT+\nrecurrent tracing'),(6,1,'SAKT\nsoftmax attention'),(10,1,'LightSAKT\nELU+1 causal kernel')]:box(ax,x,y,label)
    arrow(ax,(3.5,3.5),(4.5,3.5));arrow(ax,(3.5,3.1),(4.5,1.4));arrow(ax,(7.5,1),(8.5,1))
    ax.set_title('Conceptual model progression; LightSAKT preserves attention-based tracing');save_figure('Figure_3_conceptual')
    fig,ax=plt.subplots(figsize=(11,10));ax.set_xlim(0,12);ax.set_ylim(0,12);ax.axis('off')
    box(ax,3,11,'Observed history: q_t, r_t');box(ax,9,11,'Known next question: q_(t+1)')
    box(ax,3,9.7,'x_t = q_t + Q*r_t\nM(x_t) + P_t');box(ax,9,9.7,'Question embedding only\nE(q_(t+1))',width=3.7)
    arrow(ax,(3,10.6),(3,10.1));arrow(ax,(9,10.6),(9,10.1))
    box(ax,3,8.3,'K = W_K m ; V = W_V m\nphi(K) = ELU(K)+1');box(ax,9,8.3,'Q = W_Q e\nphi(Q) = ELU(Q)+1')
    arrow(ax,(3,9.3),(3,8.7));arrow(ax,(9,9.3),(9,8.7))
    box(ax,6,6.9,'Causal prefix kernel attention per head\n(phi(Q_t) · sum_valid,j<=t phi(K_j) outer V_j)\n/ (phi(Q_t) · sum_valid,j<=t phi(K_j) + epsilon)',width=9,height=1.1)
    arrow(ax,(3,7.9),(4.5,7.5));arrow(ax,(9,7.9),(7.5,7.5))
    box(ax,6,5.3,'Concatenate heads + output projection\nLayerNorm(dropout(attention) + m + e)',width=8)
    arrow(ax,(6,6.3),(6,5.8))
    box(ax,6,3.9,'Compact FFN: d -> floor(d/2) -> d\nReLU, dropout; residual + LayerNorm',width=8)
    arrow(ax,(6,4.8),(6,4.3))
    box(ax,6,2.5,'Linear d -> 1; sigmoid\npredict correctness of next response',width=8)
    arrow(ax,(6,3.4),(6,2.9))
    box(ax,6,1.1,'Masked next-response BCE\nTarget r_(t+1) is used only for the loss',width=8,color='#f5e7d5')
    arrow(ax,(6,2),(6,1.5));ax.set_title('Corrected LightSAKT — original parameter structure preserved');save_figure('Figure_4_architecture')
    Path('paper/corrected_architecture.mmd').write_text('flowchart TD\n history[Observed q_t and r_t] --> interaction[M of q_t+Q*r_t plus P_t]\n query[Known q_t+1] --> exercise[E of q_t+1]\n interaction --> kv[Learned K and V]\n exercise --> q[Learned Q]\n kv --> prefix[Valid causal prefix sums]\n prefix --> attn[Normalized ELU+1 multihead attention]\n q --> attn\n attn --> residual[Output projection, dropout, residual m+e, layer norm]\n residual --> ffn[Compact FFN d to d/2 to d, residual and norm]\n ffn --> probability[Sigmoid next-response probability]\n probability --> loss[Masked BCE against r_t+1]\n')

if __name__=='__main__':main()
