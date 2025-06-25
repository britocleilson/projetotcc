import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, RecurrentNetwork
import pandas as pd

from a3tgcn import KubernetesA3TGCN, create_dataset
from recurrent import get_dataset


a3tgcn = KubernetesA3TGCN(12, 116, 1)
a3tgcn.load_state_dict(torch.load("./modelos_treinados/a3tgcn.pt"))

a3tgcn.to("cpu")
print(sum(p.numel() for p in a3tgcn.parameters()), "parameters")

node_features = np.load("./data/hyper/test/node_features.npz")
edge_features = np.load("./data/hyper/test/edge_features.npz")

X, y = node_features["X"], node_features["y"]
A = edge_features["A"]

k8s_dataset = create_dataset(X, A, y, 1)

y_a3tgcn = []
p_a3tgcn = []
for (x, edge_index, edge_attr, y) in k8s_dataset:
    y_a3tgcn.append(y[1].numpy())
    p_a3tgcn.append(
        a3tgcn(x[1].unsqueeze(2), edge_index[1], edge_attr[1]).detach().numpy().ravel()
    )
y_a3tgcn = np.vstack(y_a3tgcn)
p_a3tgcn = np.vstack(p_a3tgcn)

print("A3TGCN mse:", np.mean((y_a3tgcn - p_a3tgcn) ** 2))
print("A3TGCN mape:", np.mean(np.abs((y_a3tgcn - p_a3tgcn) / y_a3tgcn)))

print("MSE by pod:", np.mean((y_a3tgcn - p_a3tgcn) ** 2, axis=0))
print("MAPE by pod:", np.mean(np.abs((y_a3tgcn - p_a3tgcn) / y_a3tgcn), axis=0))

## Temporal-Fusion Transformer
tft = TemporalFusionTransformer.load_from_checkpoint("./modelos_treinados/tft.ckpt")
tft.eval()
print(sum(p.numel() for p in tft.parameters()), "parameters")

recurrent_dataset, _ = get_dataset("./data/hyper/test/", 1, 12)
dataloader = recurrent_dataset.to_dataloader(train=False, batch_size=1)

tft_predictions = tft.predict(dataloader, return_y=True)

p_tft = torch.cat(tft_predictions.output, axis=1).swapaxes(0, 1).cpu().detach().numpy()
y_tft = torch.cat(tft_predictions.y[0]).cpu().detach().numpy()

print("TFT mse:", np.mean((y_tft - p_tft) ** 2))
print("TFT mape:", np.mean(np.abs((y_tft - p_tft) / y_tft)))

print("MSE by pod:", np.mean((y_tft - p_tft) ** 2, axis=1))
print("MAPE by pod:", np.mean(np.abs((y_tft - p_tft) / y_tft), axis=1))

## RNN
rnn = RecurrentNetwork.load_from_checkpoint("./modelos_treinados/recurrent_lstm.ckpt")
rnn.eval()
print(sum(p.numel() for p in rnn.parameters()), "parameters")

rnn_predictions = rnn.predict(dataloader, return_y=True)

p_rnn = torch.cat(rnn_predictions.output, axis=1).swapaxes(0, 1).cpu().detach().numpy()
y_rnn = torch.cat(rnn_predictions.y[0]).cpu().detach().numpy()
print("RNN mse:", np.mean((y_rnn - p_rnn) ** 2))
print("RNN mape:", np.mean(np.abs((y_rnn - p_rnn) / y_rnn)))

print("MSE by pod:", np.mean((y_rnn - p_rnn) ** 2, axis=1))
print("MAPE by pod:", np.mean(np.abs((y_rnn - p_rnn) / y_rnn), axis=1))

np.mean((y_rnn - p_rnn) ** 2, axis=1).argmax()


###Plots

fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
color_cycle = plt.rcParams['axes.prop_cycle']()

labels = ["A3TGCN", "TFT", "RNN"]
features_names = ["Redis", "Currency Service"]
model_colors = [next(color_cycle), next(color_cycle), next(color_cycle)]
feature_colors = [next(color_cycle), next(color_cycle)]

for i, feature in enumerate([9, 3]):
    for j, p in enumerate([p_a3tgcn.swapaxes(0, 1), p_tft, p_rnn]):
        axs[i, j].plot(y_a3tgcn[:, feature], label=features_names[i], alpha=0.7, **feature_colors[i])
        axs[i, j].plot(p[feature], label=labels[j], ls="dotted", **model_colors[j])

legend_lines = []
legend_labels = []
for i in range(2):
    legend_lines.append(axs[i, 0].get_legend_handles_labels()[0][0])
    legend_labels.append(axs[i, 0].get_legend_handles_labels()[1][0])
for j in range(3):
    legend_lines.append(axs[0, j].get_legend_handles_labels()[0][1])
    legend_labels.append(axs[0, j].get_legend_handles_labels()[1][1])

fig.supxlabel("Time step (5 minutes)")
fig.supylabel("Megabytes (normalized)", x=0.08)
fig.legend(legend_lines, legend_labels)
fig.savefig("./img/RAM predictions.png", dpi=300, transparent=True, bbox_inches='tight');


