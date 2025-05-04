import os
import numpy as np
import matplotlib.pyplot as plt
from exp.exp_informer import Exp_Informer
import exp


# set saved model path
setting = 'point9_change_in_lr_stop_at_6_ep_early_234mse_11mae_starting_lr_at_point005'
# setting = 'informer_custom_ftM_sl96_ll48_pl24_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
# path = os.path.join(args.checkpoints,setting,'checkpoint.pth')
sample_index = 5
# When we finished exp.train(setting) and exp.test(setting), we will get a trained model and the results of test experiment
# The results of test experiment will be saved in ./results/{setting}/pred.npy (prediction of test dataset) and ./results/{setting}/true.npy (groundtruth of test dataset)

preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')

# [samples, pred_len, dimensions]
preds.shape, trues.shape

# Get first prediction and corresponding ground truth
pred_sample = preds[sample_index, :, :6]  # First 3 features: x, y, z
true_sample = trues[sample_index, :, :6]
print(pred_sample)
print(true_sample)

# Save values (optional, for analysis)
np.save("first_predicted_path_xyz.npy", pred_sample)
np.save("first_true_path_xyz.npy", true_sample)

# n_outputs = preds.shape[1]
# fig, axs = plt.subplots(n_outputs, 1, figsize=(10, 2.5 * n_outputs), sharex=True)

# for i in range(n_outputs):
#     axs[i].plot(preds[:, i], label='Prediction', color='blue')
#     axs[i].plot(trues[:, i], label='True', color='orange')
#     axs[i].set_title(f'Output Dimension {i+1}')
#     axs[i].legend()

# plt.xlabel('Sample Index')
# plt.tight_layout()
# plt.show()

import plotly.graph_objects as go

fig = go.Figure()
for i in range(preds.shape[1]):
    fig.add_trace(go.Scatter(y=preds[:, i], name=f'Pred {i+1}', mode='lines'))
    fig.add_trace(go.Scatter(y=trues[:, i], name=f'True {i+1}', mode='lines'))

fig.update_layout(title="Predicted vs. True Values (All Outputs)",
                  xaxis_title="Sample Index",
                  yaxis_title="Value",
                  height=600)
fig.show()

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i in range(4):  # Plot only first 4 columns
    axs[i].plot(range(len(pred_sample)), pred_sample[:, i], label='Prediction', marker='o')
    axs[i].plot(range(len(true_sample)), true_sample[:, i], label='True Value', marker='x')
    axs[i].set_title(f'Column {i}')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()


# print(preds[:, 0])
# print(trues[:, 0])

'''
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Add true box (green, constant)
# true_lats = [true_sample['min_lat'], true_sample['max_lat'], true_sample['max_lat'], true_sample['min_lat'], true_sample['min_lat']]
# true_lons = [true_sample['min_lon'], true_sample['min_lon'], true_sample['max_lon'], true_sample['max_lon'], true_sample['min_lon']]

# fig.add_trace(go.Scattergeo(
#     lat=true_lats,
#     lon=true_lons,
#     mode='lines',
#     line=dict(color='green', width=2),
#     name='True Box'
# ))

# Create slider steps for each predicted box
steps = []
for i, pred in enumerate(pred_sample):
    pred_lats = [pred[0], pred[1], pred[1], pred[0], pred[0]]
    pred_lons = [pred[2], pred[2], pred[3], pred[3], pred[2]]

    fig.add_trace(go.Scattergeo(
        lat=pred_lats,
        lon=pred_lons,
        mode='lines',
        line=dict(color='red', width=2),
        name=f'Prediction {i+1}',
        visible=(i == 0)  # Only show the first prediction initially
    ))


    # true_sample_sample = true_sample[i]
    # true_lats = [true_sample_sample[0], true_sample_sample[1], true_sample_sample[1], true_sample_sample[0], true_sample_sample[0]]
    # true_lons = [true_sample_sample[2], true_sample_sample[2], true_sample_sample[3], true_sample_sample[3], true_sample_sample[2]]

    # fig.add_trace(go.Scattergeo(
    #     lat=true_lats,
    #     lon=true_lons,
    #     mode='lines',
    #     line=dict(color='green', width=2),
    #     name=f'Prediction {i+1}',
    #     visible=(i == 0)
    # ))

    step = dict(
        method='update',
        args=[{'visible': [True] + [j == i for j in range(len(pred_sample))]},
              {'title': f'Prediction {i+1}'}],
    )
    steps.append(step)

# Add slider to the layout
fig.update_layout(
    title='Predicted vs True Bounding Boxes',
    geo=dict(
        scope='usa',
        showland=True,
    ),
    sliders=[dict(
        active=0,
        steps=steps,
        x=0.1,
        xanchor="left",
        y=0,
        yanchor="top"
    )]
)

fig.show()
'''