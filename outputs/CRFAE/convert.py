for i in range(16, 21):
    with open(f'{i}/te.txt', 'r') as f:
        lines = f.readlines()
    lines = lines[3::5]
    lines = list(map(lambda x: ' '.join(x.strip().split()), lines))
    olines = []
    for j, l in enumerate(lines):
        olines.append(f'{j} {l}')
    output = '\n'.join(olines)
    with open(f'{i}/te.txt', 'w') as f:
        f.write(output)