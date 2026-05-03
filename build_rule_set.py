import pickle
import sys
# Import the mangling function from your main analyzer
# Ensure password_analyzer.py is in the same directory or PYTHONPATH
sys.path.append('.')
from password_analyzer import apply_mangling_rules

def build_rule_set(input_file, output_file):
    """
    Reads a wordlist, applies mangling rules to each word,
    and saves the set of unique mangled forms as a pickle.
    """
    rule_set = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            variations = apply_mangling_rules(word)
            rule_set.update(variations)
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx+1} words, rule set size: {len(rule_set)}")

    with open(output_file, 'wb') as f:
        pickle.dump(rule_set, f)

    print(f"Rule set saved with {len(rule_set)} unique entries to {output_file}")

if __name__ == '__main__':
    build_rule_set('rules_sample.txt', 'rule_set.pkl')