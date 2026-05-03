import random

def preprocess(input_file, output_markov, output_rules, markov_sample_size=6373958, rule_sample_size=100000):
    """
    Reads a large wordlist, normalizes to lowercase, filters length,
    and performs reservoir sampling to create two samples:
    - one for Markov training (default 5 million)
    - one for rule set generation (default 100,000)
    """
    markov_lines = []
    rule_lines = []
    total_lines = 0

    # Read input with UTF-8 encoding, ignoring errors to handle any malformed bytes
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word = line.strip().lower()
            if len(word) < 4 or len(word) > 64:
                continue
            total_lines += 1

            # Reservoir sampling for Markov
            if len(markov_lines) < markov_sample_size:
                markov_lines.append(word)
            else:
                j = random.randint(0, total_lines - 1)
                if j < markov_sample_size:
                    markov_lines[j] = word

            # Reservoir sampling for rules
            if len(rule_lines) < rule_sample_size:
                rule_lines.append(word)
            else:
                j = random.randint(0, total_lines - 1)
                if j < rule_sample_size:
                    rule_lines[j] = word

    # Write outputs with UTF-8 encoding
    with open(output_markov, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markov_lines))

    with open(output_rules, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rule_lines))

    print(f"Processed {total_lines} valid lines.")
    print(f"Markov sample: {len(markov_lines)} words")
    print(f"Rules sample: {len(rule_lines)} words")

if __name__ == '__main__':
    # Adjust input filename as needed
    preprocess('data/realhuman_phill.txt', 'markov_sample.txt', 'rules_sample.txt')