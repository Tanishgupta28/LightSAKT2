import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from models.light_sakt import Light_SAKT
from data_loaders.algebra2005 import Algebra2005
from models.utils import collate_fn

def generate_and_plot_heatmap(model, data_loader, device, question_map=None):
    model.eval()
    with torch.no_grad():
        data = next(iter(data_loader))
        q, r, qshft, rshft, m = data
        
        q = q[0:1].to(device)
        r = r[0:1].to(device)
        qshft = qshft[0:1].to(device)

        _, attn_weights = model(q.long(), r.long(), qshft.long(), get_attn_weights=True)
        
        attn_weights = attn_weights[0].cpu().numpy()

        past_interactions = q.squeeze(0).cpu().numpy()
        current_questions = qshft.squeeze(0).cpu().numpy()

        y_labels = [f"Q: {question_map.get(qid, qid)}" for qid in current_questions] if question_map else current_questions
        x_labels = [f"Hist {i}: (Q:{qid}, R:{resp})" for i, (qid, resp) in enumerate(zip(past_interactions, r.squeeze(0).cpu().numpy()))]

        plt.figure(figsize=(12, 10))
        sns.heatmap(attn_weights, xticklabels=x_labels, yticklabels=y_labels, cmap="viridis", cbar=True)
        
        plt.title("Attention Head 1: Question Relationships")
        plt.ylabel("Current Question (Query)")
        plt.xlabel("Past Interactions (Key/Value)")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    NUM_QUESTIONS = 177
    EMBED_DIM = 64
    NUM_HEADS = 4
    DROPOUT = 0.2
    MAX_SEQ_LEN = 100

    MODEL_CKPT_PATH = "ckpts/light_sakt/Algebra2005/epochs50_tr0.8_lr0.001_embed64_heads4_drop0.2/model.ckpt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Light_SAKT(
        num_q=NUM_QUESTIONS, 
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN
    )
    
    if not os.path.exists(MODEL_CKPT_PATH):
        raise FileNotFoundError(f"Could not find model at: {MODEL_CKPT_PATH}")

    model.load_state_dict(torch.load(MODEL_CKPT_PATH, map_location=device))
    model.to(device)
    print("Model loaded successfully.")

    dataset = Algebra2005(seq_len=MAX_SEQ_LEN)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    print("Test data loaded.")
    
    question_id_to_name = {
        10: "Fraction Addition", 
        15: "Algebra Basics", 
        25: "Geometry Shapes"
    }

    print("Generating heatmap...")
    generate_and_plot_heatmap(model, test_loader, device, question_map=question_id_to_name)