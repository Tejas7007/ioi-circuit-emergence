#!/usr/bin/env python3
"""
parse_pile_ioi.py - Extract IOI examples from Pythia's training data (The Pile).

Scans EleutherAI/the_pile_deduplicated for sentences matching the IOI pattern:
- A transfer verb (gave, told, handed, etc.)
- Followed by "to [Name]" within 8 words
- The IO name must appear BEFORE the verb in context
- A different name (subject) must appear at least twice
- Both names present in the final prompt

Scanned 13.9M texts to find 500 validated examples.

Usage:
    pip install datasets
    python3 parse_pile_ioi.py --n-examples 500
"""

import json
import re
import argparse
from collections import Counter
from datasets import load_dataset

NAMES = set([
    'Aaron','Adam','Alan','Alex','Alice','Amanda','Amy','Andrew','Angela','Anna',
    'Anne','Arthur','Ben','Beth','Bill','Bob','Brad','Brian','Carl','Carol',
    'Charlie','Chris','Claire','Colin','Dan','Daniel','Dave','David','Dean','Diana',
    'Don','Donna','Ed','Edward','Elena','Ellen','Emily','Emma','Eric','Eva',
    'Frank','Fred','Gary','George','Greg','Hannah','Harry','Helen','Henry','Holly',
    'Ian','Iris','Jack','Jake','James','Jane','Jason','Jean','Jeff','Jennifer',
    'Jerry','Jim','Joan','Joe','John','Jon','Julia','Julie','Justin','Karen',
    'Kate','Keith','Kelly','Ken','Kevin','Kim','Larry','Laura','Lee','Leon',
    'Linda','Lisa','Louis','Lucy','Luke','Lynn','Marc','Maria','Marie','Mark',
    'Martin','Mary','Matt','Max','Meg','Michael','Mike','Nancy','Neil','Nick',
    'Noah','Oliver','Owen','Pat','Paul','Peter','Phil','Rachel','Ray','Richard',
    'Rob','Robert','Robin','Roger','Ron','Rose','Roy','Ruth','Ryan','Sam',
    'Sarah','Scott','Sean','Sharon','Simon','Sophie','Steve','Susan','Tim','Tom',
    'Tony','Victor','Will','Zoe'
])

VERBS = [
    'gave', 'told', 'handed', 'sent', 'showed', 'passed',
    'offered', 'brought', 'lent', 'paid', 'threw', 'taught'
]

BLACKLIST = [
    'alice and bob', 'bob and alice', 'bitcoin', 'blockchain', 'crypto',
    'sold to', 'ownership history', 'owner of record', 'deed', 'parcel',
    'fort lee', 'fort johnson', 'henry iv', 'henry v', 'richard ii',
    'henry ii', 'henry iii', 'louis xiv', 'charles ii', 'edward i', 'james i'
]


def parse_pile(n_examples=500):
    print(f"Loading The Pile (deduplicated)...")
    ds = load_dataset('EleutherAI/the_pile_deduplicated', split='train', streaming=True)

    examples = []
    n = 0

    for item in ds:
        if len(examples) >= n_examples:
            break
        n += 1
        if n % 2000 == 0:
            print(f'Scanned {n}, found {len(examples)}')

        text = item['text']
        if any(b in text.lower() for b in BLACKLIST):
            continue

        words = text.split()
        for i, w in enumerate(words):
            wl = w.lower().rstrip('.,;:!?"()')
            if wl not in VERBS:
                continue

            for j in range(i + 1, min(i + 8, len(words))):
                if words[j].lower().rstrip('.,;:!?') != 'to':
                    continue
                if j + 1 >= len(words):
                    continue
                io_cand = words[j + 1].rstrip('.,;:!?"()[]')
                if io_cand not in NAMES:
                    continue

                # Look in big window for both names
                window_start = max(0, i - 60)
                window = words[window_start:j + 2]

                # IO must appear before verb
                pre_verb = words[window_start:i]
                io_found = any(
                    ww.rstrip('.,;:!?"()[]').lstrip('"(') == io_cand
                    for ww in pre_verb
                )
                if not io_found:
                    continue

                # S must appear 2+ times, different from IO
                name_counts = Counter()
                for ww in window:
                    clean = ww.rstrip('.,;:!?"()[]').lstrip('"(')
                    if clean in NAMES and clean != io_cand:
                        name_counts[clean] += 1

                s_cand = None
                for name, count in name_counts.items():
                    if count >= 2:
                        s_cand = name
                        break
                if not s_cand:
                    continue

                # Find where IO first appears to start prompt there
                io_pos = None
                for k in range(window_start, i):
                    if words[k].rstrip('.,;:!?"()[]').lstrip('"(') == io_cand:
                        io_pos = k
                        break

                # Build prompt from 5 words before IO to 'to'
                prompt_start = max(0, io_pos - 5) if io_pos else window_start
                prompt = ' '.join(words[prompt_start:j + 1])

                # Final verify
                if io_cand not in prompt or s_cand not in prompt:
                    continue
                if len(prompt) < 30:
                    continue

                examples.append({
                    'prompt': prompt.strip(),
                    'io_name': io_cand,
                    's_name': s_cand,
                    'full_text': ' '.join(words[max(0, i - 10):min(len(words), j + 5)])[:200],
                })
                break

    print(f'\nDone. Scanned {n}, found {len(examples)}')

    # Verify
    for e in examples:
        assert e['io_name'] in e['prompt'], f"BUG: {e['io_name']} not in prompt"
        assert e['s_name'] in e['prompt'], f"BUG: {e['s_name']} not in prompt"

    print(f'All {len(examples)} examples verified: both names in every prompt')

    for e in examples[:10]:
        print(f"  IO={e['io_name']}, S={e['s_name']}")
        print(f"    {e['prompt'][:150]}")
        print()

    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-examples', type=int, default=500)
    parser.add_argument('--out', default='data/pile_real_ioi_v5.json')
    args = parser.parse_args()

    examples = parse_pile(args.n_examples)

    with open(args.out, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f'Saved {len(examples)} to {args.out}')
