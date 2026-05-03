import json
from collections import defaultdict

def train_markov(corpus_file, order=2):
    """
    Train a Markov model of given order (default 2 = trigram).
    Saves the model as 'markov_model.json'.
    """
    counts = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    first_counts = defaultdict(int)
    total_first = 0

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            pwd = line.strip()
            if len(pwd) <= order:
                continue

            # First characters
            first = pwd[:order]
            first_counts[first] += 1
            total_first += 1

            # Transitions
            for i in range(len(pwd) - order):
                prefix = pwd[i:i+order]
                next_char = pwd[i+order]
                counts[prefix][next_char] += 1
                total[prefix] += 1

    # Convert to probabilities
    first_probs = {pre: cnt/total_first for pre, cnt in first_counts.items()}
    trans_probs = {}
    for prefix, nxt_counts in counts.items():
        trans_probs[prefix] = {nxt: cnt/total[prefix] for nxt, cnt in nxt_counts.items()}

    model = {
        'order': order,
        'first': first_probs,
        'transitions': trans_probs
    }

    with open('markov_model.json', 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False)

    print(f"Markov model saved with {len(first_probs)} starting sequences and {len(trans_probs)} transitions.")

if __name__ == '__main__':
    train_markov('markov_sample.txt')