'''
Use the six short documents below. You may use any library or write your own code; AI assistants are permitted. Represent each document as its set of words (ignore order and repetition).

d1: "the quick brown fox jumps over the lazy dog"
d2: "the quick brown fox leaps over the lazy dog"
d3: "a fast brown fox jumps over a sleepy dog"
d4: "machine learning models need careful evaluation"
d5: "careful evaluation is needed for machine learning models"
d6: "the stock market rallied after the announcement"
(a) Compute the exact Jaccard similarity for the pairs (d1, d2), (d4, d5), and (d1, d6).

(b) Estimate the same three similarities using MinHash. State the number of hash functions K you chose, and present a small table comparing your estimate to the exact value for each pair.
'''
from datasketch import MinHash
import pandas as pd

# 1. Definizione dei documenti e conversione in set (come nel punto a)
documents = {
    "d1": "the quick brown fox jumps over the lazy dog",
    "d2": "the quick brown fox leaps over the lazy dog",
    "d4": "machine learning models need careful evaluation",
    "d5": "careful evaluation is needed for machine learning models",
    "d6": "the stock market rallied after the announcement"
}

d1_set = set(documents["d1"].split())
d2_set = set(documents["d2"].split())
d4_set = set(documents["d4"].split())
d5_set = set(documents["d5"].split())
d6_set = set(documents["d6"].split())

def exact_jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

# 2. Configurazione MinHash (Scegliamo K = 200 funzioni hash)
K = 200

m1 = MinHash(num_perm=K, seed=42)
m2 = MinHash(num_perm=K, seed=42)
m4 = MinHash(num_perm=K, seed=42)
m5 = MinHash(num_perm=K, seed=42)
m6 = MinHash(num_perm=K, seed=42)

# Popoliamo i MinHash (ricorda di codificare le stringhe in byte)
for word in d1_set: m1.update(word.encode('utf-8'))
for word in d2_set: m2.update(word.encode('utf-8'))
for word in d4_set: m4.update(word.encode('utf-8'))
for word in d5_set: m5.update(word.encode('utf-8'))
for word in d6_set: m6.update(word.encode('utf-8'))

# 3. Raccolta dati per la tabella comparativa
pairs = [
    ("d1, d2", exact_jaccard(d1_set, d2_set), m1.jaccard(m2)),
    ("d4, d5", exact_jaccard(d4_set, d5_set), m4.jaccard(m5)),
    ("d1, d6", exact_jaccard(d1_set, d6_set), m1.jaccard(m6))
]

# Creazione della tabella comparativa usando un DataFrame di Pandas
df_res = pd.DataFrame(pairs, columns=["Pair", "Exact Jaccard", f"MinHash Estimate (K={K})"])

# Formattiamo i numeri decimali a 4 cifre per renderla più leggibile ed elegante
df_res["Exact Jaccard"] = df_res["Exact Jaccard"].round(4)
df_res[f"MinHash Estimate (K={K})"] = df_res[f"MinHash Estimate (K={K})"].round(4)

print(f"Number of hash functions chosen: K = {K}\n")
print(df_res.to_string(index=False))