import pickle

with open("ckpts/dkt+/Algebra2005/metrics.pkl", "rb") as f:
    dkt_data = pickle.load(f)

with open("ckpts/light_sakt/Algebra2005/metrics.pkl", "rb") as f:
    light_sakt_data = pickle.load(f)

print("DKT+ keys:", dkt_data.keys())
print("LightSAKT keys:", light_sakt_data.keys())
