import numpy as np

n = 30
experiment = '{}.15of5x5.28Jul.log'.format
time_experiment = '{}.15of5x5.28Jul.time.log'.format
methods = [
    'OE',
    'SE',
    'forward',
    'topk'
]

def read_file(path, time_path):
    with open(path) as f:
        data = [
            float(line.strip().split()[-1])*100
            for line in f.readlines()
            if line.startswith('k = ')
        ]
    with open(time_path) as tf:
        time_data = [
            float(line.strip())*1000
            for line in tf.readlines()[2::2]
        ]
    group_size = len(data)//n
    data = [data[i*group_size:(i+1)*group_size] for i in range(n)]
    if 'topk' in path:
        data = [d[1:] for d in data]
    return [time_data]+list(zip(*data))

def flatten(data):
    flatten_data = []
    for d in data:
        flatten_data += d
    return flatten_data

data = {method: read_file(experiment(method), time_experiment(method)) for method in (methods)}
flatten_data = {method: flatten(d[2:]) for method, d in data.items()}

stats ={
    method: (
        [np.mean(di).round(1) for di in d]+[np.mean(flatten_data[method]).round(1)],
        [np.std(di).round(1) for di in d]+[np.std(flatten_data[method]).round(1)]
    ) for method, d in data.items()
}

for method, stat in stats.items():
    print(method)
    print('mean:', stat[0])
    print('std: ', stat[1])
    print()