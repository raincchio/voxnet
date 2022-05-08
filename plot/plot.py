import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib


runs_1 = [pd.read_csv('loss2/' + x) for x in os.listdir('loss2')]


def create_smooth(dataframe, window=10):
    variable = [dataframe.Value[0]]
    for i in range(2, window):
        variable.append(dataframe.Value.rolling(i).mean()[i - 1])

    new_variable = np.r_[variable, list(dataframe.Value.rolling(window).mean()[window - 1:])]

    dataframe['smooth'] = new_variable


def create_bands(dataframe):
    steps = np.tile(dataframe.Step.values, 3)
    real_values = dataframe.Value.values
    smooth_values = dataframe.smooth.values
    mirrored_values = np.abs(2 * smooth_values - real_values)
    new_df = pd.DataFrame({'steps': steps,
                           'values': np.r_[real_values, smooth_values, mirrored_values]})
    return new_df

for run in runs_1:
    create_smooth(run)

sns.set_context('paper')
sns.lineplot(x='Step', y='smooth', data=runs_1[1], label='SparseVoxNet-D')
sns.lineplot(x='Step', y='smooth', data=runs_1[2], label='SparseVoxNet-DS')
sns.lineplot(x='Step', y='smooth', data=runs_1[3], label='SparseVoxNet-DC')
sns.lineplot(x='Step', y='smooth', data=runs_1[0], label='SparseVoxNet-DSC')
# sns.lineplot(x='Step', y='smooth', data=runs_1[4], label='sparsevoxnet-dsc')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper right')
# ax.set_title('densevoxnet vs sparsevoxnet')

# ax.text(runs_1[0].Step.values[-1]+0.1, runs_1[0].smooth.values[-1]-0.003,
#         '{:.2f}'.format(runs_1[0].smooth.values[-1]))
# # ax.text(runs_1[3].Step.values[-1]+0.1, runs_1[3].smooth.values[-1]-0.005,
# #         '{:.2f}%'.format(100*runs_1[3].smooth.values[-1]))
# ax.text(runs_1[1].Step.values[-1]+0.1, runs_1[1].smooth.values[-1],
#         '{:.2f}'.format(runs_1[1].smooth.values[-1]))

baseline = runs_1[0].smooth.values[-1]

ax.set_xlabel('epochs')
ax.set_ylabel('entropy loss')
# plt.show()
plt.savefig('figures/baseline.png', bbox_inches='tight')