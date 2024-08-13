from library.evaluation import corpus_acc
from library.utils import read_attachment_file, flatten
import numpy as np
from library.ensemble import ensemble
import torch
from library.model_selection_library import *
from references import INDIVIDUALS, GOLD
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

ALPHA_RANGE = np.arange(0.0, .81, .1)
USE_DEV_WEIGHTS_FOR_SELECTION = False
USE_DEV_WEIGHTS_FOR_ENSEMBLE = False
ONE_PER_EACH_GROUP = False
POOL_SIZE = 'all'

TEACGER_PATHS = [[tps[5], tps[2], tps[3], tps[4], tps[1], tps[0], tps[6]][:-1] for tps in INDIVIDUALS]
FINE_TUNING_K = 5

def main(
    METHOD = 'forward society entropy',
    ALPHA = .1,
    ALPHA2 = .1,
    SEED = 2,
    MAX_K = None,
    FINE_TUNE_ALPHA = False,
    TIME_ONLY = False,
    PRINT_TIME = True,
):
    teachers_paths = TEACGER_PATHS.copy()
    gold_path = GOLD

    assert METHOD in ALL_METHODS
    assert not ((METHOD not in TUNABLE_METHODS) and (FINE_TUNE_ALPHA))
    assert not ((METHOD not in NONFORWARD_METHODS) and (ONE_PER_EACH_GROUP))
    assert not (POOL_SIZE!='all' and ONE_PER_EACH_GROUP)
    assert not (TIME_ONLY and FINE_TUNE_ALPHA)
    assert not (TIME_ONLY and (MAX_K is None))

    groups_size = len(teachers_paths[0]) if ONE_PER_EACH_GROUP else -1
    teachers_paths = flatten(teachers_paths)
    if POOL_SIZE!='all':
        np.random.seed(SEED)
        teachers_paths = np.random.choice(teachers_paths, size=POOL_SIZE, replace=False).tolist()


    gold_te = read_attachment_file(gold_path.format('te'))
    gold_d = read_attachment_file(gold_path.format('d'))
    preds_te = [read_attachment_file(pred.format('te')) for pred in teachers_paths]
    preds_d = [read_attachment_file(pred.format('d')) for pred in teachers_paths]
    individuals_UAS_d = np.array([corpus_acc(p, gold_d) for p in preds_d])
    cpu_weights = individuals_UAS_d if USE_DEV_WEIGHTS_FOR_SELECTION else [1]*len(preds_d)
    torch_weights = torch.tensor(cpu_weights).to(device)
    ensemble_weights = individuals_UAS_d if USE_DEV_WEIGHTS_FOR_ENSEMBLE else [1]*len(preds_d)
    individuals_UAS_d = torch.from_numpy(individuals_UAS_d).to(device)
    oh_flatten_preds_d = torch.Tensor([flatten(pred) for pred in preds_d]).to(device) # KxN
    oh_flatten_preds_d = torch.nn.functional.one_hot(oh_flatten_preds_d.to(torch.int64), int((oh_flatten_preds_d.max()+1).item())) # KxNxdim


    if FINE_TUNE_ALPHA:
        best_alpha, best_alpha2, best_score = -1, -1, -1
        for alpha in ALPHA_RANGE:
          for alpha2 in ALPHA_RANGE:
            selected_indexes = find_best_combination(
                preds_d,
                gold_d,
                oh_flatten_preds_d,
                individuals_UAS_d,
                FINE_TUNING_K,
                alpha=alpha,
                torch_weights=torch_weights,
                cpu_weights=cpu_weights,
                groups_size=groups_size,
                method=METHOD,
                alpha2=alpha2
            )
            ensemble_output = ensemble(
                [p for j, p in enumerate(preds_d) if j in selected_indexes],
                weights=[cpu_weights[j] for j in selected_indexes]
            )
            ensemble_UAS = corpus_acc(ensemble_output, gold_d)
            if METHOD in ['society entropy and Dietterichs Kappa', 'society entropy and oracle entropy']:
                print('Alpha2 =', alpha2, end=', ')
            print('Alpha =', alpha, '|', 'UAS:\t', ensemble_UAS)
            if ensemble_UAS > best_score:
                best_alpha, best_alpha2, best_score = alpha, alpha2, ensemble_UAS
            if METHOD not in ['society entropy and Dietterichs Kappa', 'society entropy and oracle entropy']:
                break
        print('-'*25)
        print('Best alpha:', best_alpha)
        if METHOD in ['society entropy and Dietterichs Kappa', 'society entropy and oracle entropy']:
            print('Best alpha2:', best_alpha2)

    else:
        _, selected_indexes = torch.topk(individuals_UAS_d, len(individuals_UAS_d)) if METHOD=='backward selection' else torch.topk(individuals_UAS_d, 1)
        selected_indexes = selected_indexes.tolist()
        k_range = reversed(range(2 if MAX_K is None else MAX_K, len(preds_d))) if  METHOD=='backward selection' else\
            range(groups_size, groups_size+1) if ONE_PER_EACH_GROUP else\
            range(1, len(preds_d)+1 if MAX_K is None else MAX_K+1) if METHOD=='topk' else\
            range(2, len(preds_d) if MAX_K is None else MAX_K+1)
        total_time = 0
        for k in k_range:
            start_time = time.time()
            selected_indexes = find_best_combination(
                preds_d,
                gold_d,
                oh_flatten_preds_d,
                individuals_UAS_d,
                k,
                alpha=ALPHA,
                alpha2=ALPHA2,
                torch_weights=torch_weights,
                cpu_weights=cpu_weights,
                groups_size=groups_size,
                method=METHOD,
                k_1_selection=selected_indexes,
            )
            end_time = time.time()
            total_time += end_time-start_time
            if not TIME_ONLY:
                ensemble_output = ensemble(
                    [p for j, p in enumerate(preds_te) if j in selected_indexes],
                    weights=[ensemble_weights[j] for j in selected_indexes]
                )
                print('k =', k, '|', 'UAS:\t', corpus_acc(ensemble_output, gold_te))
                print([teachers_paths[j] for j in selected_indexes])
                print('-'*25)
        if PRINT_TIME:
            print(total_time)

# main('society entropy', ALPHA=None, FINE_TUNE_ALPHA=True)

# main('topk', ALPHA=None)
# main('forward selection', ALPHA=None, MAX_K=5)
# main('forward society entropy', ALPHA=.3)
# main('forward Dietterichs Kappa', ALPHA=.1)
# main('forward oracle entropy', ALPHA=.1)
# main('forward tally', ALPHA=.3)
# main('forward KW variance', ALPHA=.3)
# main('forward shasha diversity', ALPHA=.1)

for seed in range(1):
    print("seed:", seed)
    # main('forward oracle entropy', ALPHA=.1, SEED=seed, MAX_K=5)
    main('forward society entropy', ALPHA=.4, SEED=seed, MAX_K=5)
    # main('forward selection', ALPHA=None, SEED=seed, MAX_K=5)
    # main('topk', ALPHA=None, SEED=seed, MAX_K=5)