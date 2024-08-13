import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style('ticks')

MAIN = True

def read_file(path):
    with open(path) as f:
        out = [
        float(line.strip().split()[-1])
        for line in f.readlines()[1::2]
        ]
    if 'topk' in path:
        out = out[1:-1]
    return out

methods = [
    ['w/o diversity', 'topk', '#E1C999', 'o', (1,1)],
    ['w/ Kuncheva’s diversity', 'OE', '#CE88B2', '^', ''],
    ['w/ society entropy', 'SE', '#4CE0D2', 'P', ''],
    ['Ensemble validation', 'forward', 'gray', 'p', (1,2)],
] if MAIN else [
    ['w/ disagreement', 'shasha', 'gray', '*', (1,1)],
    ['w/ KW variance', 'KW', '#E1C999', 'o', (3,2)],
    ['w/ Fleiss’ Kappa', 'DK', '#226BAB', 'v', ''],
    ['w/ PCDM', 'tally', '#CE88B2', 'p', ''],
    ['w/ society entropy', 'SE', '#4CE0D2', 'P', ''],
]
names, methods, palette, markers, dashes = zip(*methods)

experiment = 'model_selection_logs/{}.5x5.28Jul.log'.format

data = pd.DataFrame({name: read_file(experiment(method))[1:] for name, method in zip(names, methods)})
data.iloc[:, :] *= 100
data.index = list(range(3, len(data)+3))

plt.figure(figsize=(16, 8))
ax = sns.lineplot(
    data,
    alpha=.6,
    dashes=dashes,
    linewidth=4.5,
    palette=palette,
    legend=False,
)
sns.lineplot(
    data,
    markers=['o']*len(markers),
    markersize=20,
    alpha=1,
    linestyle='',
    dashes=False,
    linewidth=3,
    palette=['white']*(len(palette)),
    legend=False,
)
ax = sns.lineplot(
    data,
    markers=markers,
    markersize=14,
    alpha=1,
    linestyle='',
    dashes=False,
    linewidth=3,
    palette=palette,
)

plt.ylim(63.5 + 0.1*(1-MAIN), 68.8 + 0.1*(1-MAIN))
plt.xlim(2.7, len(data)+2.3)
fontsize=24
plt.legend(fontsize=fontsize*(1.4 if MAIN else .95))
plt.xticks(range(3, len(data)+3), fontsize=fontsize)
plt.xlabel('# ensemble components', fontsize=fontsize*1.4)
plt.yticks(fontsize=fontsize)
plt.ylabel('Ensemble UAS', fontsize=fontsize*1.4)
plt.savefig(open(f'plots/model_selection_{"main" if MAIN else "appndx"}.pdf', 'wb'), bbox_inches='tight')