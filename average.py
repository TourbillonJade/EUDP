from library.evaluation import *
from library.ensemble import ensemble
from library.utils import attachment2span, read_attachment_file
from references import INDIVIDUALS, GOLD
import numpy as np

FOLD = 'te'
USE_DEV_WEIGHTS = False
AGG = 'acc'
WRITE = False

teachers_paths = [[tps[5], tps[2], tps[3], tps[4], tps[1], tps[0], tps[6]][:-1] for tps in INDIVIDUALS]
# teachers_paths = [['outputs/JointNDMV/4/{}.out', 'outputs/ENDMV/15/{}.out', 'outputs/LNDMV/1/{}.txt', 'outputs/CRFAE/13/{}.txt', 'outputs/SibNDMV/1/{}.txt']]
teachers_paths = [['outputs/Flow/'+str(i)+'/{}.out'] for i in range(1, 21)]
gold_path = "outputs/Gold/{}.txt"
output_path = "outputs/ensemble/"+FOLD+"_{}.txt"


def f1_eval(preds, refs):
    preds_spans = [[attachment2span(a) for a in pds] for pds in preds]
    refs_spans = [attachment2span(a) for a in refs]
    f1s = [corpus_f1(pds, refs_spans) for pds in preds_spans]
    return f"CorpusF1: {(np.mean(f1s)*100).round(1)}±{(np.std(f1s)*100).round(1)}"


def acc_eval(preds, refs):
    accs = [corpus_acc(pds, refs) for pds in preds]
    return f"CorpusUAS: {(np.mean(accs)*100).round(1)}±{(np.std(accs)*100).round(1)}"


def eval(preds, refs):
    return f1_eval(preds, refs) + "\t" + acc_eval(preds, refs)


preds = [[read_attachment_file(t.format(FOLD)) for t in tps] for tps in teachers_paths]
gold = read_attachment_file(GOLD.format(FOLD))

# if USE_DEV_WEIGHTS:
#     preds_d = [[read_attachment_file(t.format('d')) for t in tps] for tps in teachers_paths]
#     gold_d = read_attachment_file(GOLD.format('d'))
#     avg = [ensemble(pr, agg=AGG, parallel=True, weights=[corpus_acc(pds, gold_d) for pds in pr_d]) for pr, pr_d in zip(preds, preds_d)]
# else:
#     avg = [ensemble(pr, agg=AGG, parallel=True) for pr in preds]


print("Teachers' performance:")
for i,t in enumerate(zip(*preds)):
    name = teachers_paths[0][i].split('/')[-3]
    print(name, ' '*(15-len(name)), ':\t', eval(t, gold))
print("-" * 30)
name = "Average"
print(name, ' '*(15-len(name)), ':\t', eval(avg, gold))

if WRITE:
    for i, ag in enumerate(avg):
        with open(output_path.format(i+1), 'w') as f:
            out = '\n'.join([' '.join(map(str, [j]+a)) for j,a in enumerate(ag)])
            f.write(out)