import re
import math
import ahocorasick
import requests
import hashlib
import json
import pickle
import random
import string
from collections import Counter

# ----------------------------------------------------------------------
# Global variables (loaded at startup)
# ----------------------------------------------------------------------
PASSWORDS = set()
ENGLISH_WORDS = set()
FEMALE_NAMES = set()
MALE_NAMES = set()
SURNAMES = set()
TV_FILM = set()
ALL_WORDS = set()
AUTOMATON = None

MARKOV_MODEL = None
RULE_WORDS = set()

HASH_SPEEDS = {
    'md5': 200e9,
    'sha1': 80e9,
    'sha256': 30e9,
    'ntlm': 400e9,
    'bcrypt': 10e3,
    'argon2': 1e3,
    'pbkdf2': 100e3,
    'default': 1e9
}

# ----------------------------------------------------------------------
# Loading functions
# ----------------------------------------------------------------------
def load_wordlists(data_dir='data'):
    """Load all zxcvbn wordlists and build Aho-Corasick automaton."""
    global PASSWORDS, ENGLISH_WORDS, FEMALE_NAMES, MALE_NAMES, SURNAMES, TV_FILM, ALL_WORDS, AUTOMATON
    import os
    def load(fname):
        path = os.path.join(data_dir, fname)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return {line.strip().lower() for line in f if line.strip()}
    PASSWORDS      = load('passwords.txt')
    ENGLISH_WORDS  = load('english_wikipedia.txt')
    FEMALE_NAMES   = load('female_names.txt')
    MALE_NAMES     = load('male_names.txt')
    SURNAMES       = load('surnames.txt')
    TV_FILM        = load('us_tv_and_film.txt')
    ALL_WORDS      = PASSWORDS | ENGLISH_WORDS | FEMALE_NAMES | MALE_NAMES | SURNAMES | TV_FILM
    AUTOMATON      = build_automaton(ALL_WORDS)
    print(f"✅ Loaded {len(ALL_WORDS)} unique words from zxcvbn lists.")

def build_automaton(word_set):
    """Build Aho-Corasick automaton from a set of words."""
    A = ahocorasick.Automaton()
    for word in word_set:
        A.add_word(word.lower(), word)
    A.make_automaton()
    return A

def load_markov_model(model_file='markov_model.json'):
    """Load pre-trained Markov model from JSON."""
    global MARKOV_MODEL
    try:
        with open(model_file, 'r', encoding='utf-8') as f:
            MARKOV_MODEL = json.load(f)
        print("✅ Markov model loaded.")
    except Exception as e:
        print("⚠️  Markov model not found. Markov estimation disabled.")

def load_rule_set(rule_file='rule_set.pkl'):
    """Load pre-computed rule-based variations set."""
    global RULE_WORDS
    try:
        with open(rule_file, 'rb') as f:
            RULE_WORDS = pickle.load(f)
        print(f"✅ Rule set loaded with {len(RULE_WORDS)} entries.")
    except Exception as e:
        print("⚠️  Rule set not found. Rule-based detection disabled.")

# ----------------------------------------------------------------------
# Dictionary substring search (Aho-Corasick)
# ----------------------------------------------------------------------
def find_dictionary_substrings(password):
    """Return list of dictionary substrings found in password."""
    if AUTOMATON is None:
        return []
    lower = password.lower()
    found = set()
    for end_index, word in AUTOMATON.iter(lower):
        found.add(word)
    return list(found)

# ----------------------------------------------------------------------
# Rule-based attack simulation
# ----------------------------------------------------------------------
def apply_mangling_rules(word):
    """Generate common mangling variations of a word."""
    variations = set()
    wl = word.lower()
    variations.add(wl)
    variations.add(wl.capitalize())
    variations.add(wl.upper())
    variations.add(wl[::-1])
    leet_map = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$'}
    leet = ''.join(leet_map.get(c, c) for c in wl)
    variations.add(leet)
    for digits in ['123', '1234', '1', '12', '69', '00', '2020', '2021', '2022', '2023', '2024']:
        variations.add(wl + digits)
        variations.add(leet + digits)
        variations.add(wl.capitalize() + digits)
    for sym in ['!', '@', '#', '$', '%']:
        variations.add(sym + wl)
        variations.add(sym + leet)
        variations.add(wl + sym)
        variations.add(leet + sym)
    return variations

def rule_based_match(password):
    """Return True if password matches a mangled dictionary word."""
    if not RULE_WORDS:
        return False
    return password.lower() in RULE_WORDS

# ----------------------------------------------------------------------
# Pattern analysis
# ----------------------------------------------------------------------
KEYBOARD_ROWS = [
    "qwertyuiop", "asdfghjkl", "zxcvbnm",
    "QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"
]

def keyboard_sequences(password, min_len=4):
    lower = password.lower()
    for row in KEYBOARD_ROWS:
        row_lower = row.lower()
        for i in range(len(row_lower) - min_len + 1):
            seq = row_lower[i:i+min_len]
            if seq in lower:
                return True
    return False

def repeated_chars(password, threshold=3):
    return bool(re.search(r'(.)\1{%d,}' % (threshold-1), password))

def sequential_digits(password):
    digits = re.findall(r'\d+', password)
    for d in digits:
        if len(d) >= 3:
            for i in range(len(d)-2):
                if int(d[i+1]) == int(d[i])+1 and int(d[i+2]) == int(d[i])+2:
                    return True
    return False

def common_substitutions(password):
    subs = {'@': 'a', '0': 'o', '1': 'i', '3': 'e', '$': 's', '5': 's', '7': 't'}
    lower = password.lower()
    for ch, orig in subs.items():
        if ch in password and orig in lower:
            return True
    return False

def contains_name_or_popculture(password):
    lower = password.lower()
    for name in FEMALE_NAMES | MALE_NAMES | SURNAMES:
        if len(name) > 2 and name in lower:
            return True, "name"
    for term in TV_FILM:
        if len(term) > 2 and term in lower:
            return True, "pop culture"
    return False, None

# ----------------------------------------------------------------------
# ✅ NEW: User Info Analysis
# ----------------------------------------------------------------------
def get_user_info_tokens(user_info: dict) -> list:
    """
    Extract all meaningful tokens from user info dict.

    Expected keys (all optional):
        first_name  : str  – e.g. "John"
        last_name   : str  – e.g. "Smith"
        username    : str  – e.g. "john_smith"
        email       : str  – e.g. "john@gmail.com"
        birthdate   : str  – "YYYY-MM-DD" or "DD/MM/YYYY"
        pet_name    : str  – e.g. "Buddy"
        city        : str  – e.g. "London"
        company     : str  – e.g. "Google"
        phone       : str  – e.g. "9876543210"

    Returns a flat list of lowercase string tokens to check against password.
    """
    tokens = []

    # ── Simple string fields ──────────────────────────────────────────
    simple_fields = ['first_name', 'last_name', 'username', 'pet_name', 'city', 'company']
    for field in simple_fields:
        val = user_info.get(field, '')
        if val and isinstance(val, str):
            tokens.append(val.lower().strip())

    # ── Email: extract local part before @ ───────────────────────────
    email = user_info.get('email', '')
    if email and '@' in email:
        local = email.split('@')[0].lower().strip()
        tokens.append(local)
        # Also split on dots/underscores inside local part
        for part in re.split(r'[._\-]', local):
            if part:
                tokens.append(part)

    # ── Birthdate: individual parts + common combos ──────────────────
    birthdate = user_info.get('birthdate', '')
    if birthdate and isinstance(birthdate, str):
        parts = re.split(r'[-/.]', birthdate.strip())
        tokens.extend(parts)   # raw parts: year, month, day
        if len(parts) == 3:
            # Try to detect format YYYY-MM-DD vs DD/MM/YYYY
            if len(parts[0]) == 4:          # YYYY-MM-DD
                year, month, day = parts
            else:                           # DD/MM/YYYY
                day, month, year = parts
            short_year = year[-2:] if len(year) >= 2 else year
            tokens += [
                month + day,                # 0312
                day + month,                # 1203
                short_year,                 # 95
                year,                       # 1995
                day + month + short_year,   # 120395
                month + day + short_year,   # 031295
                day + month + year,         # 12031995
            ]

    # ── Phone: full digits + common slices ──────────────────────────
    phone = user_info.get('phone', '')
    if phone and isinstance(phone, str):
        digits = re.sub(r'\D', '', phone)
        if digits:
            tokens.append(digits)
            if len(digits) >= 4:
                tokens.append(digits[-4:])   # last 4 digits
            if len(digits) >= 6:
                tokens.append(digits[-6:])   # last 6 digits
            if len(digits) >= 10:
                tokens.append(digits[-10:])  # full 10-digit number

    # ── Filter: remove empty / too-short tokens (avoid false positives)
    tokens = list({t for t in tokens if t and len(t) >= 3})
    return tokens


def check_user_info_in_password(password: str, user_info: dict) -> list:
    """
    Check if the password contains any user-info tokens.

    Returns a list of matched token strings (for reporting in issues/suggestions).
    Returns an empty list if user_info is None / empty.
    """
    if not user_info:
        return []

    lower_pwd = password.lower()
    tokens    = get_user_info_tokens(user_info)
    matched   = []

    for token in tokens:
        if token in lower_pwd and token not in matched:
            matched.append(token)

    return matched

# ----------------------------------------------------------------------
# Entropy calculation
# ----------------------------------------------------------------------
def calculate_entropy(password):
    charsets = 0
    if re.search(r'[a-z]', password): charsets += 26
    if re.search(r'[A-Z]', password): charsets += 26
    if re.search(r'[0-9]', password): charsets += 10
    if re.search(r'[^a-zA-Z0-9]', password): charsets += 32
    if charsets == 0:
        return 0
    entropy = len(password) * math.log2(charsets)
    return round(entropy, 2)

# ----------------------------------------------------------------------
# Markov model functions
# ----------------------------------------------------------------------
def markov_log_prob(password):
    """Return log2 probability of password under Markov model, or None if model missing."""
    if MARKOV_MODEL is None:
        return None
    order = MARKOV_MODEL['order']
    pwd = password.lower()
    if len(pwd) <= order:
        return -float('inf')
    logprob = 0.0
    first = pwd[:order]
    first_probs = MARKOV_MODEL['first']
    if first in first_probs:
        logprob += math.log2(first_probs[first])
    else:
        logprob += math.log2(1e-10)
    trans = MARKOV_MODEL['transitions']
    for i in range(len(pwd)-order):
        prefix = pwd[i:i+order]
        nxt = pwd[i+order]
        if prefix in trans and nxt in trans[prefix]:
            logprob += math.log2(trans[prefix][nxt])
        else:
            logprob += math.log2(1e-10)
    return logprob

def markov_entropy(password):
    logp = markov_log_prob(password)
    if logp is None:
        return None
    return -logp

# ----------------------------------------------------------------------
# HIBP breach check
# ----------------------------------------------------------------------
def check_hibp(password):
    """Return True if password appears in known breaches (HIBP)."""
    sha1 = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
    prefix, suffix = sha1[:5], sha1[5:]
    try:
        resp = requests.get(f'https://api.pwnedpasswords.com/range/{prefix}', timeout=2)
        if resp.status_code == 200:
            hashes = (line.split(':')[0] for line in resp.text.splitlines())
            return suffix in hashes
    except Exception:
        pass
    return False

# ----------------------------------------------------------------------
# Attack time estimation
# ----------------------------------------------------------------------
def estimate_crack_time_simple(password, entropy, dict_hit):
    guesses_per_sec = 1e9
    if dict_hit:
        seconds = 10_000_000 / guesses_per_sec
    else:
        possibilities = 2 ** entropy
        seconds = possibilities / guesses_per_sec
    return format_time(seconds)

def estimate_guesses_hybrid(password):
    """Estimate number of guesses needed in an optimal attack order."""
    dict_size = len(ALL_WORDS)
    rule_size = len(RULE_WORDS)
    lower = password.lower()

    if lower in ALL_WORDS:
        return dict_size / 2

    if rule_based_match(password):
        return dict_size + rule_size / 2

    me = markov_entropy(password)
    if me is not None:
        markov_guesses = 2 ** me
        charset_size = 0
        if any(c.islower() for c in password): charset_size += 26
        if any(c.isupper() for c in password): charset_size += 26
        if any(c.isdigit() for c in password): charset_size += 10
        if any(not c.isalnum() for c in password): charset_size += 32
        if charset_size == 0: charset_size = 26
        brute_guesses = charset_size ** len(password)
        effective = min(markov_guesses, brute_guesses)
        return dict_size + rule_size + effective / 2

    charset_size = 0
    if any(c.islower() for c in password): charset_size += 26
    if any(c.isupper() for c in password): charset_size += 26
    if any(c.isdigit() for c in password): charset_size += 10
    if any(not c.isalnum() for c in password): charset_size += 32
    if charset_size == 0: charset_size = 26
    brute_guesses = charset_size ** len(password)
    return dict_size + rule_size + (2 ** 40) + brute_guesses / 2

def estimate_crack_time_advanced(password, hash_type='default', gpu_count=1):
    guesses = estimate_guesses_hybrid(password)
    speed = HASH_SPEEDS.get(hash_type, HASH_SPEEDS['default']) * gpu_count
    seconds = guesses / speed
    return format_time(seconds)

def format_time(seconds):
    if seconds < 1:
        return "instantly (< 1 second)"
    elif seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    elif seconds < 31536000:
        return f"{seconds/86400:.1f} days"
    else:
        return f"{seconds/31536000:.1f} years"

# ----------------------------------------------------------------------
# ✅ UPDATED: Feature extraction (now 15 features — added user_info_hits)
# ----------------------------------------------------------------------
def extract_features(password, issues, entropy, rule_hit=False, breached=False, user_info_hits=0):
    """
    Extract numerical features for the ML model.

    Features (15 total):
        0  length
        1  entropy
        2  num_char_classes
        3  has_upper
        4  has_lower
        5  has_digit
        6  has_special
        7  digit_ratio
        8  special_ratio
        9  has_pattern
        10 dict_hit
        11 name_hit
        12 rule_hit
        13 breached
        14 user_info_hits   ← NEW
    """
    length = len(password)
    has_upper   = int(any(c.isupper() for c in password))
    has_lower   = int(any(c.islower() for c in password))
    has_digit   = int(any(c.isdigit() for c in password))
    has_special = int(any(not c.isalnum() for c in password))
    num_char_classes = has_upper + has_lower + has_digit + has_special
    digit_ratio   = sum(c.isdigit() for c in password) / length if length else 0
    special_ratio = sum(not c.isalnum() for c in password) / length if length else 0

    has_pattern = int(any([
        keyboard_sequences(password),
        repeated_chars(password),
        sequential_digits(password),
        common_substitutions(password)
    ]))
    name_hit, _ = contains_name_or_popculture(password)
    dict_hit = int(any("common word" in i or "common password" in i for i in issues))

    features = [
        length,
        entropy,
        num_char_classes,
        has_upper,
        has_lower,
        has_digit,
        has_special,
        digit_ratio,
        special_ratio,
        has_pattern,
        dict_hit,
        int(name_hit),
        int(rule_hit),
        int(breached),
        min(user_info_hits, 5),   # ← NEW: cap at 5 to avoid outliers
    ]
    return features

# ----------------------------------------------------------------------
# ✅ UPDATED: Scoring (now accepts user_info_matches)
# ----------------------------------------------------------------------
def compute_score(password, entropy, dict_hit, issues, ml_risk,
                  rule_hit=False, breached=False, user_info_matches=None):
    """
    Compute overall password strength score (0–100).

    Deduction table:
        dict_hit            → -30
        rule_hit            → -20
        breached            → -40
        per issue           → -3 each
        ml_risk == weak     → -20
        ml_risk == medium   → -10
        per user_info match → -15 each    ← NEW
        >2 user_info matches→ -20 extra   ← NEW
    """
    base       = min(entropy, 100)
    deductions = 0

    if dict_hit:
        deductions += 30
    if rule_hit:
        deductions += 20
    if breached:
        deductions += 40
    if issues:
        deductions += len(issues) * 3
    if ml_risk == "weak":
        deductions += 20
    elif ml_risk == "medium":
        deductions += 10

    # ── NEW: penalise personal info usage ────────────────────────────
    if user_info_matches:
        deductions += len(user_info_matches) * 15
        if len(user_info_matches) > 2:
            deductions += 20   # extra penalty for heavily personal passwords

    score = max(0, min(100, base - deductions))
    return int(score)

# ----------------------------------------------------------------------
# ✅ UPDATED: Suggestions (now accepts user_info_matches)
# ----------------------------------------------------------------------
def generate_suggestions(issues, entropy, length, user_info_matches=None):
    """Generate human-readable improvement tips."""
    tips = []
    if length < 8:
        tips.append("Use at least 8 characters.")
    if entropy < 30:
        tips.append("Increase entropy by mixing uppercase, digits, and symbols.")
    if any("keyboard" in i for i in issues):
        tips.append("Avoid keyboard sequences like 'qwerty'.")
    if any("repeated" in i for i in issues):
        tips.append("Avoid repeated characters.")
    if any("sequential digits" in i for i in issues):
        tips.append("Avoid sequential numbers.")
    if any("common word" in i for i in issues):
        tips.append("Avoid common words or names.")
    if any("common password" in i for i in issues):
        tips.append("This password appears in data breaches. Choose a unique one.")
    if any("rule" in i for i in issues):
        tips.append("Avoid simple transformations of common words (e.g., appending digits).")
    if any("exposed" in i for i in issues):
        tips.append("This password has been leaked in a breach. Never reuse it.")

    # ── NEW: user info warning ────────────────────────────────────────
    if user_info_matches:
        shown = ', '.join(user_info_matches[:3])
        tips.append(
            f"Your password contains personal information ({shown}). "
            "Attackers who know you can guess this easily — avoid your name, "
            "birthday, pet name, or phone number."
        )

    if not tips:
        tips.append("Your password looks strong. Keep it safe!")
    return tips

# ----------------------------------------------------------------------
# Password generation helpers
# ----------------------------------------------------------------------
def generate_password_from_markov(length=16):
    """Generate a password using the Markov model (or fallback to random)."""
    if MARKOV_MODEL is None:
        return generate_random_password(length)
    try:
        order       = MARKOV_MODEL['order']
        first_probs = MARKOV_MODEL['first']
        transitions = MARKOV_MODEL['transitions']
        if not first_probs:
            return generate_random_password(length)
        start_bigrams = list(first_probs.keys())
        start_probs   = list(first_probs.values())
        start = random.choices(start_bigrams, weights=start_probs)[0]
        pwd   = list(start)
        while len(pwd) < length:
            context = ''.join(pwd[-order:])
            if context in transitions and transitions[context]:
                next_chars = list(transitions[context].keys())
                next_probs = list(transitions[context].values())
                pwd.append(random.choices(next_chars, weights=next_probs)[0])
            else:
                break
        all_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*'
        while len(pwd) < length:
            pwd.append(random.choice(all_chars))
        pwd_str = ''.join(pwd)
        if not (any(c.islower() for c in pwd_str) and
                any(c.isupper() for c in pwd_str) and
                any(c.isdigit() for c in pwd_str) and
                any(not c.isalnum() for c in pwd_str)):
            return generate_password_from_markov(length)
        return pwd_str
    except Exception:
        return generate_random_password(length)

def generate_random_password(length=16):
    """Fallback: completely random password with all character classes."""
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits    = string.digits
    symbols   = "!@#$%^&*()_-+=<>?"
    all_chars = lowercase + uppercase + digits + symbols
    pwd = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(symbols),
    ]
    for _ in range(length - 4):
        pwd.append(random.choice(all_chars))
    random.shuffle(pwd)
    return ''.join(pwd)

def improve_password(original, min_length=12):
    """Return a stronger version of the original password."""
    pwd = original
    has_lower  = any(c.islower() for c in pwd)
    has_upper  = any(c.isupper() for c in pwd)
    has_digit  = any(c.isdigit() for c in pwd)
    has_symbol = any(not c.isalnum() for c in pwd)
    symbols_set = "!@#$%^&*()_-+=<>?"

    if not has_upper:
        lower_indices = [i for i, c in enumerate(pwd) if c.islower()]
        if lower_indices:
            i = random.choice(lower_indices)
            pwd = pwd[:i] + pwd[i].upper() + pwd[i+1:]
        else:
            pos = random.randint(0, len(pwd))
            pwd = pwd[:pos] + random.choice(string.ascii_uppercase) + pwd[pos:]
    if not has_digit:
        pos = random.randint(0, len(pwd))
        pwd = pwd[:pos] + random.choice(string.digits) + pwd[pos:]
    if not has_symbol:
        pos = random.randint(0, len(pwd))
        pwd = pwd[:pos] + random.choice(symbols_set) + pwd[pos:]

    all_chars = string.ascii_letters + string.digits + symbols_set
    while len(pwd) < min_length:
        pos = random.randint(0, len(pwd))
        pwd = pwd[:pos] + random.choice(all_chars) + pwd[pos:]
    return pwd

def generate_suggestions_from_user(user_password, count=5):
    """Generate stronger password suggestions based on user's input."""
    suggestions = set()
    for _ in range(count * 3):
        improved = improve_password(user_password)
        suggestions.add(improved)
        if len(suggestions) >= count:
            break
    while len(suggestions) < count:
        suggestions.add(generate_random_password(16))
    return list(suggestions)[:count]

# ----------------------------------------------------------------------
# ✅ UPDATED: Main analysis function (now accepts user_info=None)
# ----------------------------------------------------------------------
def analyze_password(password, ml_model=None, hash_type='default', gpu_count=1, user_info=None):
    """
    Full password analysis.

    Parameters
    ----------
    password   : str   – the password to analyse
    ml_model   : fitted sklearn model or None
    hash_type  : str   – one of HASH_SPEEDS keys (default 'default')
    gpu_count  : int   – number of GPUs for crack-time estimation
    user_info  : dict or None – personal info dict with keys:
                     first_name, last_name, username, email,
                     birthdate, pet_name, city, company, phone

    Returns
    -------
    dict with keys:
        score, crack_time, crack_time_advanced, entropy,
        issues, ml_risk, ml_confidence, suggestions,
        user_info_matches   ← NEW
    """
    # ── Pattern analysis ─────────────────────────────────────────────
    pattern_issues = []
    if keyboard_sequences(password):
        pattern_issues.append("Contains keyboard sequence")
    if repeated_chars(password):
        pattern_issues.append("Contains repeated characters")
    if sequential_digits(password):
        pattern_issues.append("Contains sequential digits")
    if common_substitutions(password):
        pattern_issues.append("Uses common substitutions (leet)")
    name_hit, name_type = contains_name_or_popculture(password)
    if name_hit:
        pattern_issues.append(f"Contains a common {name_type}")

    # ── Dictionary check ─────────────────────────────────────────────
    substrings = find_dictionary_substrings(password)
    dict_issues = []
    if substrings:
        examples = ', '.join(substrings[:3])
        dict_issues.append(f"Contains common word(s): {examples}")
    if password.lower() in PASSWORDS or password in PASSWORDS:
        dict_issues.append("Password is a very common password (exact match)")

    # ── Rule-based match ─────────────────────────────────────────────
    rule_hit = rule_based_match(password)
    if rule_hit:
        dict_issues.append("Password is a common word with simple transformations (e.g., leet, appended digits)")

    # ── Breach check ─────────────────────────────────────────────────
    breached = check_hibp(password)
    if breached:
        dict_issues.append("Password has been exposed in data breaches")

    # ── NEW: User info check ─────────────────────────────────────────
    user_info_matches = []
    user_info_issues  = []
    if user_info:
        user_info_matches = check_user_info_in_password(password, user_info)
        if user_info_matches:
            shown = ', '.join(user_info_matches[:3])
            user_info_issues.append(
                f"Contains personal information: {shown}"
            )

    # ── Combine all issues ───────────────────────────────────────────
    all_issues = pattern_issues + dict_issues + user_info_issues
    dict_hit   = any("common word" in i or "common password" in i for i in all_issues)

    # ── Entropy ──────────────────────────────────────────────────────
    entropy = calculate_entropy(password)

    # ── Attack times ─────────────────────────────────────────────────
    crack_time_simple   = estimate_crack_time_simple(password, entropy, dict_hit)
    crack_time_advanced = estimate_crack_time_advanced(password, hash_type, gpu_count)

    # ── ML prediction ────────────────────────────────────────────────
    ml_risk = "unknown"
    ml_conf = 0.0
    if ml_model is not None:
        features = extract_features(
            password, all_issues, entropy,
            rule_hit, breached,
            user_info_hits=len(user_info_matches),   # ← NEW
        )
        probs  = ml_model.predict_proba([features])[0]
        pred   = ml_model.predict([features])[0]
        labels = {0: "weak", 1: "medium", 2: "strong"}
        ml_risk = labels.get(pred, "unknown")
        ml_conf = max(probs)

    # ── Overall score ────────────────────────────────────────────────
    score = compute_score(
        password, entropy, dict_hit, all_issues, ml_risk,
        rule_hit, breached,
        user_info_matches=user_info_matches,   # ← NEW
    )

    # ── Suggestions ──────────────────────────────────────────────────
    suggestions = generate_suggestions(
        all_issues, entropy, len(password),
        user_info_matches=user_info_matches,   # ← NEW
    )

    return {
        "score":              score,
        "crack_time":         crack_time_simple,
        "crack_time_advanced":crack_time_advanced,
        "entropy":            entropy,
        "issues":             all_issues,
        "ml_risk":            ml_risk,
        "ml_confidence":      round(ml_conf, 2),
        "suggestions":        suggestions,
        "user_info_matches":  user_info_matches,   # ← NEW
    }