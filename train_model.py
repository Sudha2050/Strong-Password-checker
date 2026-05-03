"""
train_model.py  —  User-Info-Aware Password Strength Classifier
================================================================
What's new vs original:
  * Training samples now include passwords that deliberately contain
    personal info tokens (name, date fragments, phone slices, etc.)
    so the model LEARNS that personal-info usage = weak, not just
    that a password has certain character classes.
  * extract_features() now has 15 features (user_info_hits = feature 14).
  * Synthetic user profiles are generated per sample so the model sees
    diverse personal-info patterns during training.
  * Full evaluation: accuracy, classification report, confusion matrix,
    feature importance bar chart, and a real-world sanity check.

Labels:
    0 = weak
    1 = medium
    2 = strong
"""

import numpy as np
import random
import string
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import password_analyzer as pa

# ── Load supporting data ────────────────────────────────────────────────────
pa.load_wordlists('data')
pa.load_rule_set('rule_set.pkl')
pa.load_markov_model('markov_model.json')

print("✅ Data loaded. Starting dataset generation...\n")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Synthetic user profile generator
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_FIRST_NAMES = [
    "sudha", "john", "priya", "amit", "sara", "michael", "lakshmi",
    "david", "ananya", "raj", "emma", "arjun", "linda", "vijay", "lisa",
    "rahul", "jessica", "kiran", "james", "pooja"
]
SAMPLE_LAST_NAMES = [
    "raju", "smith", "sharma", "patel", "jones", "kumar", "brown",
    "reddy", "wilson", "nair", "taylor", "das", "moore", "singh", "martin",
    "iyer", "anderson", "rao", "thomas", "gupta"
]
SAMPLE_PET_NAMES = ["buddy", "max", "charlie", "luna", "rocky", "bella", "tiger", "milo"]
SAMPLE_CITIES    = ["london", "mumbai", "delhi", "paris", "toronto", "sydney", "berlin"]
SAMPLE_COMPANIES = ["google", "infosys", "tcs", "amazon", "microsoft", "wipro", "accenture"]


def random_date():
    year  = random.randint(1970, 2005)
    month = random.randint(1, 12)
    day   = random.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def random_phone():
    return ''.join(str(random.randint(0, 9)) for _ in range(10))


def make_random_user() -> dict:
    first = random.choice(SAMPLE_FIRST_NAMES)
    last  = random.choice(SAMPLE_LAST_NAMES)
    dob   = random_date()
    phone = random_phone()
    return {
        "first_name": first,
        "last_name":  last,
        "username":   first + str(random.randint(100, 999)),
        "email":      f"{first}.{last}{random.randint(1,99)}@gmail.com",
        "birthdate":  dob,
        "pet_name":   random.choice(SAMPLE_PET_NAMES),
        "city":       random.choice(SAMPLE_CITIES),
        "company":    random.choice(SAMPLE_COMPANIES),
        "phone":      phone,
    }


def get_date_fragments(dob: str) -> list:
    """Extract common date fragments people use in passwords."""
    parts = dob.split('-')
    if len(parts) != 3:
        return []
    year, month, day = parts
    sy = year[-2:]
    return [
        month + day,
        day + month,
        sy,
        year,
        day + month + sy,
        month + day + sy,
        day + month + year,
    ]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Password generators (personal-info-aware)
# ═══════════════════════════════════════════════════════════════════════════

def generate_weak_password(user: dict):
    """
    Weak: common words, keyboard sequences, OR pure personal info tokens.
    Returns (password, user_info_hits).
    """
    strategy = random.choice(['dict', 'personal_pure', 'personal_name_date', 'keyboard'])

    if strategy == 'dict':
        word = random.choice(list(pa.ALL_WORDS))
        return word, 0

    elif strategy == 'personal_pure':
        # Just a raw personal token: first name, pet name, birth year, etc.
        choices = [
            user['first_name'],
            user['last_name'],
            user['pet_name'],
            user['city'],
        ] + get_date_fragments(user['birthdate'])
        pwd = random.choice([c for c in choices if len(c) >= 3])
        hits = pa.check_user_info_in_password(pwd, user)
        return pwd, len(hits)

    elif strategy == 'personal_name_date':
        # Name + date fragment — extremely common real-world weak pattern
        first = user['first_name']
        frags = get_date_fragments(user['birthdate'])
        frag  = random.choice(frags) if frags else str(random.randint(10, 99))
        pwd   = first + frag
        hits  = pa.check_user_info_in_password(pwd, user)
        return pwd, len(hits)

    else:  # keyboard
        seqs = ['qwerty', 'asdfgh', '123456', 'abcdef', 'zxcvbn', '111111', 'password']
        return random.choice(seqs), 0


def generate_medium_password(user: dict):
    """
    Medium: dict word + rules, OR personal info dressed up with symbols/digits.
    Still guessable by a targeted attacker. Returns (password, user_info_hits).
    """
    strategy = random.choice(['dict_rule', 'personal_dressed', 'personal_mixed'])

    if strategy == 'dict_rule':
        base = random.choice(list(pa.ALL_WORDS)).capitalize()
        if random.random() > 0.5:
            base = base.replace('a', '@').replace('o', '0').replace('e', '3')
        base += str(random.randint(10, 99))
        if random.random() > 0.5:
            base += random.choice(['!', '@', '#', '$'])
        return base, 0

    elif strategy == 'personal_dressed':
        # Personal info + symbols — looks decent but is personally guessable
        first = user['first_name'].capitalize()
        frags = get_date_fragments(user['birthdate'])
        frag  = random.choice(frags) if frags else '99'
        sym   = random.choice(['!', '@', '#', '$', '&'])
        combos = [
            first + user['last_name'][:3].capitalize() + frag,
            first + frag + sym,
            user['pet_name'].capitalize() + frag + sym,
            first + user['city'][:3] + frag,
            user['last_name'].capitalize() + frag + sym,
        ]
        pwd  = random.choice(combos)
        hits = pa.check_user_info_in_password(pwd, user)
        return pwd, len(hits)

    else:  # personal_mixed
        # Random word base + personal date fragment
        word  = random.choice(list(pa.ALL_WORDS)).capitalize()
        frags = get_date_fragments(user['birthdate'])
        frag  = random.choice(frags) if frags else str(random.randint(10, 99))
        pwd   = word + frag
        hits  = pa.check_user_info_in_password(pwd, user)
        return pwd, len(hits)


def generate_strong_password(user: dict):
    """
    Strong: fully random, no personal info, no dictionary. Returns (password, 0).
    """
    length = random.randint(12, 22)
    chars  = string.ascii_letters + string.digits + "!@#$%^&*()_-+=<>?"
    for _ in range(100):   # retry loop
        pwd = ''.join(random.choice(chars) for _ in range(length))
        if (any(c.islower() for c in pwd) and
            any(c.isupper() for c in pwd) and
            any(c.isdigit() for c in pwd) and
            any(not c.isalnum() for c in pwd) and
            not pa.check_user_info_in_password(pwd, user)):
            return pwd, 0
    # Fallback — guaranteed strong structure
    base = (random.choice(string.ascii_lowercase) +
            random.choice(string.ascii_uppercase) +
            random.choice(string.digits) +
            random.choice("!@#$%^&*"))
    base += ''.join(random.choice(chars) for _ in range(length - 4))
    return ''.join(random.sample(base, len(base))), 0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Build dataset
# ═══════════════════════════════════════════════════════════════════════════

X, y = [], []
NUM_SAMPLES = 1500   # per class

print(f"Generating {NUM_SAMPLES * 3} training samples ({NUM_SAMPLES} per class)...")

skipped = 0
for i in range(NUM_SAMPLES):
    user = make_random_user()

    for generator, label in [
        (generate_weak_password,   0),
        (generate_medium_password, 1),
        (generate_strong_password, 2),
    ]:
        try:
            pwd, user_info_hits = generator(user)
        except Exception as e:
            skipped += 1
            continue

        entropy  = pa.calculate_entropy(pwd)
        rule_hit = pa.rule_based_match(pwd)
        issues   = []
        if pwd.lower() in pa.PASSWORDS:
            issues.append("common password")

        feats = pa.extract_features(
            pwd, issues, entropy,
            rule_hit=rule_hit,
            breached=False,
            user_info_hits=user_info_hits,
        )
        X.append(feats)
        y.append(label)

    if (i + 1) % 300 == 0:
        print(f"  ... {i+1}/{NUM_SAMPLES} user profiles processed")

print(f"\n✅ Dataset ready: {len(X)} samples  ({skipped} skipped)")
dist = Counter(y)
print(f"   Class distribution → weak: {dist[0]}  medium: {dist[1]}  strong: {dist[2]}")

X = np.array(X, dtype=np.float32)
y = np.array(y)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Train / test split
# ═══════════════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n   Train: {len(X_train)}  |  Test: {len(X_test)}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Train
# ═══════════════════════════════════════════════════════════════════════════

print("\n⏳ Training Random Forest (200 trees)...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)
print("✅ Training complete.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Evaluate
# ═══════════════════════════════════════════════════════════════════════════

y_pred   = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)

print(f"\n{'='*55}")
print(f"  Test Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"{'='*55}")

print("\n── Classification Report ──────────────────────────────")
print(classification_report(y_test, y_pred,
      target_names=["weak", "medium", "strong"]))

print("── Confusion Matrix ───────────────────────────────────")
cm     = confusion_matrix(y_test, y_pred)
labels = ["weak", "medium", "strong"]
header = f"{'':14s}" + "  ".join(f"{'pred_'+l:>10s}" for l in labels)
print(header)
for i, row in enumerate(cm):
    print(f"  {'real_'+labels[i]:12s}" + "  ".join(f"{v:>10d}" for v in row))

print("\n── 5-Fold Cross-Validation ────────────────────────────")
cv = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"  Scores  : {cv.round(4)}")
print(f"  Mean    : {cv.mean():.4f}   Std: {cv.std():.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — Feature importance
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "length",           # 0
    "entropy",          # 1
    "num_char_classes", # 2
    "has_upper",        # 3
    "has_lower",        # 4
    "has_digit",        # 5
    "has_special",      # 6
    "digit_ratio",      # 7
    "special_ratio",    # 8
    "has_pattern",      # 9
    "dict_hit",         # 10
    "name_hit",         # 11
    "rule_hit",         # 12
    "breached",         # 13
    "user_info_hits",   # 14  NEW
]

print("\n── Feature Importances (sorted) ───────────────────────")
for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 60)
    print(f"  {name:20s}: {imp:.4f}  {bar}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — Real-world sanity check (mirrors the screenshot user)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Sanity Check (using screenshot user profile) ───────")

sample_user = {
    "first_name": "sudha",
    "last_name":  "raju",
    "username":   "sudha434",
    "email":      "gnoob4626@gmail.com",
    "birthdate":  "2005-06-01",
    "pet_name":   "max",
    "city":       "paralakhumun",
    "company":    "cutm",
    "phone":      "07299961484",
}

test_cases = [
    ("sudha",                     "weak",   "pure first name"),
    ("max123",                    "weak",   "pet + digits"),
    ("Sudha1484@max",             "weak",   "name+phone+pet (your screenshot)"),
    ("SudhaRaju2005!",            "weak",   "full name + birth year + symbol"),
    ("Raju@0601",                 "medium", "last name + birth day+month"),
    ("P@ssw0rd!99",               "medium", "common pattern, no personal info"),
    ("xK9#mP2$qL7!nR4",          "strong", "fully random 16-char"),
    ("Tr0ub4dor&3!Xq7Vm$",       "strong", "long random strong"),
]

passed = 0
print(f"  {'STATUS':8s} {'CONF':6s} {'PASSWORD':32s} {'PRED':8s} {'NOTES'}")
print(f"  {'-'*80}")
for pwd, expected, desc in test_cases:
    hits    = pa.check_user_info_in_password(pwd, sample_user)
    entropy = pa.calculate_entropy(pwd)
    rule_h  = pa.rule_based_match(pwd)
    issues  = []
    if pwd.lower() in pa.PASSWORDS:
        issues.append("common password")
    feats   = pa.extract_features(pwd, issues, entropy,
                                   rule_hit=rule_h, breached=False,
                                   user_info_hits=len(hits))
    pred    = ["weak", "medium", "strong"][clf.predict([feats])[0]]
    conf    = max(clf.predict_proba([feats])[0])
    ok      = (pred == expected)
    status  = "PASS ✅" if ok else "FAIL ❌"
    if ok: passed += 1
    print(f"  {status:8s} {conf*100:4.0f}%  {pwd[:30]:32s} {pred:8s} {desc}")

print(f"\n  Result: {passed}/{len(test_cases)} sanity checks passed")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — Save
# ═══════════════════════════════════════════════════════════════════════════

joblib.dump(clf, 'model.pkl')
print(f"\n✅ model.pkl saved")
print(f"   Features        : {X.shape[1]}")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples    : {len(X_test)}")