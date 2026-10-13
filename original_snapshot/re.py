import pickle

with open("train/ckpts/sakt/Algebra2005/metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

print(metrics.keys())   # see what’s inside
# Best values
best_auc = max(metrics["valid_auc"])
best_acc = max(metrics["accuracy"])
best_f1 = max(metrics["f1"])
best_mse = min(metrics["mse"])
min_loss = min(metrics["train_loss"])

print("Best AUC:", best_auc)
print("Best Accuracy:", best_acc)
print("Best F1:", best_f1)
print("Best MSE:", best_mse)
print("Lowest Loss:", min_loss)
