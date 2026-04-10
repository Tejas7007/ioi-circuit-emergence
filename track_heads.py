import json

with open('results/part1_component_emergence.json') as f:
    data = json.load(f)

print('HEAD TRACKING ACROSS TRAINING (160M)')
print('='*60)

interesting = ['L0H5', 'L0H6', 'L0H10', 'L1H3', 'L1H4', 'L1H7', 'L1H8', 'L1H9', 'L8H9']

header = '{:>8}  '.format('Step')
for h in interesting:
    header += '{:>6}'.format(h)
print(header)
print('-' * len(header))

for r in data['results']:
    step = r['step']
    nm_names = set()
    for h in r.get('name_mover_heads', []):
        name = h[0] if isinstance(h, (list, tuple)) else h
        nm_names.add(name)
    
    line = '{:>8}  '.format(step)
    for h in interesting:
        if h in nm_names:
            line += '   NM '
        elif h == r.get('top_ioi_head'):
            line += '  TOP '
        else:
            line += '    . '
    print(line)
