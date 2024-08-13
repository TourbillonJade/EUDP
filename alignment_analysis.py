from library.evaluation import *
from library.utils import read_attachment_file
import numpy as np

FOLD = 'te'
USE_DEV_WEIGHTS = True

teachers_paths = [
    [
    "outputs/NDMV/2/{}.out",
    "outputs/ENDMV/6/{}.out",
    "outputs/JointNDMV/1/{}.out",
    "outputs/LNDMV/1/{}.txt",
    "outputs/CRFAE/1/{}.txt",
    "outputs/SibNDMV/1/{}.txt",
    ],[
    "outputs/NDMV/3/{}.out",
    "outputs/ENDMV/8/{}.out",
    "outputs/JointNDMV/2/{}.out",
    "outputs/LNDMV/2/{}.txt",
    "outputs/CRFAE/3/{}.txt",
    "outputs/SibNDMV/2/{}.txt",
    ],[
    "outputs/NDMV/7/{}.out",
    "outputs/ENDMV/11/{}.out",
    "outputs/JointNDMV/3/{}.out",
    "outputs/LNDMV/3/{}.txt",
    "outputs/CRFAE/6/{}.txt",
    "outputs/SibNDMV/3/{}.txt",
    ],[
    "outputs/NDMV/16/{}.out",
    "outputs/ENDMV/13/{}.out",
    "outputs/JointNDMV/4/{}.out",
    "outputs/LNDMV/4/{}.txt",
    "outputs/CRFAE/9/{}.txt",
    "outputs/SibNDMV/4/{}.txt",
    ],[
    "outputs/NDMV/20/{}.out",
    "outputs/ENDMV/15/{}.out",
    "outputs/JointNDMV/5/{}.out",
    "outputs/LNDMV/5/{}.txt",
    "outputs/CRFAE/13/{}.txt",
    "outputs/SibNDMV/5/{}.txt",
    ]
]
name_extractor = lambda x: x.split('/')[1]
gold_path = "outputs/Gold/{}.txt"

preds = [[read_attachment_file(t.format(FOLD)) for t in tps] for tps in teachers_paths]
gold = read_attachment_file(gold_path.format(FOLD))

results = []
for i in range(len(preds)):
    results.append({})
    for j1, pred1 in enumerate(preds[i][:3]):
        for j2, pred2 in enumerate(preds[i][3:]):
            results[-1][f'{name_extractor(teachers_paths[i][j1])}_{name_extractor(teachers_paths[i][3+j2])}'] = alignment(pred1, pred2, gold)

means, stds = {}, {}
for key in results[0]:
    means[key] = {}
    stds[key] = {}
    for cat in results[0][key]:
        keycat_results = [results[i][key][cat] for i in range(len(results))]
        means[key][cat] = np.mean(keycat_results).round(3)*100
        stds[key][cat] = np.std(keycat_results).round(3)*100

for key in means:
    print(key)
    items = {cat: str(means[key][cat])[:4]+' ± '+str(stds[key][cat])[:4] for cat in means[key]}
    tt, tf, ft, ff = items['TT'], items['TF'], items['FT'], items['FF']
    print(f'{tt}\t{ft}')
    print(f'{tf}\t{ff}')
    print()