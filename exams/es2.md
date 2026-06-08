# Near-Duplicate Listing Detection System

An architectural design for identifying and flagging near-duplicate item listings at scale within a second-hand marketplace, optimizing for user experience and system throughput.

---

## 1. Representation

**The Choice:** Word-level Shingles ($k=3$ or $k=4$).

* **Justification:** The core problem states that sellers create near-duplicates by making "small edits" (changing a single word, tweaking a price, adding a short caption). We are hunting for **syntactic near-duplicates** (nearly identical strings of text), not documents that merely share the same semantic topic. Shingles of length $k$ preserve the local structure and word ordering of the text.
* **Why reject the alternatives?**
    * *Bag-of-Words (Single Words / $k=1$):* This approach completely ignores word order. A malicious seller could simply scramble the word order of a listing. A bag-of-words representation would yield a 100% Jaccard similarity match, but we would lose the structural ability to catch clever rephrasings.
    * *Embeddings (e.g., Dense Vectors from BERT/SBERT):* Embeddings capture semantic meaning. If two completely different sellers are listing a "used PlayStation 5 in excellent condition," their dense vectors will be extremely close in the latent space. Using embeddings would cause massive over-clustering, flagging distinct, legitimate items from different sellers as duplicates. We need to catch the *same* seller spamming the *same* text literal.

---

## 2. Similarity & Scale

* **Similarity Measure:** **Jaccard Similarity** computed over the sets of shingles. It is the standard metric for measuring the token overlap between two sets, bound between 0 and 1, perfectly matching the definition of "near-duplicate detection."
* **Infeasibility of All-Pairs:** With $M = 3,000,000$ active listings, a naive all-pairs comparison requires an $O(M^2)$ brute-force approach:
    $$\frac{3,000,000 \times 2,999,999}{2} \approx 4.5 \times 10^{12} \text{ comparisons}$$
    Even if a highly optimized cluster could compute 1 million Jaccard similarities per second, the computation would take over **52 days**. This is impossible to scale in a production environment where items are constantly posted.
* **What Makes It Feasible:** The combination of **MinHash** and **Locality Sensitive Hashing (LSH) with Banding**. 
    1.  **MinHash** compresses large, variable-length shingle sets into short, fixed-length signatures (e.g., $K=200$ integers) while preserving the expected Jaccard similarity.
    2.  **LSH Banding** avoids comparing all signatures by hashing segments of the signatures into discrete buckets. Only listings that collide in at least one bucket become "candidate pairs." This drops the computational complexity from quadratic $O(M^2)$ to sub-linear, making the 3-million-node problem solvable in minutes.

---

## 3. Parameters & Threshold

The LSH banding technique provides three main tuning knobs:
1.  **$K$ (Signature Length):** Determines the precision and reduces the variance of our Jaccard approximation.
2.  **$b$ (Number of Bands)** and **$r$ (Rows per Band):** Where $K = b \times r$. These two parameters define the steepness and location of the S-curve probability threshold, approximated by $t \approx (1/b)^{1/r}$.

* **Tuning for Near-Duplicates:** Because the manager wants to flag *near-duplicates specifically* (and not just similar items), we need a very high similarity threshold (e.g., $s \ge 0.85$ or $s \ge 0.90$).
* **Parameter Configuration:** To push the threshold $t$ higher (to the right of the S-curve), we must choose a **higher number of rows $r$ per band** and fewer bands $b$ (for instance, $K=200, b=20, r=10$). By increasing $r$, we make the "hashing exam" much more stringent: two listings must match *exactly* on 10 consecutive MinHash values within a band to even be considered a candidate. This drastically isolates true duplicates and keeps unrelated items away.

---

## 4. No Ground Truth (Evaluation Strategy)

Without pre-existing labels, we must build an evaluation framework using **Statistical Sampling** combined with **Human Annotation (Gold Standard)**.

* **Sampling Strategy:** We run our LSH pipeline and randomly extract two distinct subsets of item pairs:
    1.  **Sample A:** 1,000 random pairs that our algorithm *flagged* as near-duplicates (LSH Candidates).
    2.  **Sample B:** 1,000 random pairs that our algorithm *ignored* (ideally focusing on pairs that had partial matches or fell just below our LSH threshold).
* **Human Evaluation:** Human moderators manually inspect these 2,000 pairs, answering a binary question: *"Are these listings spam duplicates of the exact same physical item? (Yes/No)"*.
* **Metrics to Measure:**
    * **Precision:** $\frac{\text{True Duplicates Identified}}{\text{Total Flagged by Algorithm}}$. If Precision is low, our algorithm has high *False Positives*, meaning we are accidentally hiding unique, legitimate listings from different sellers. In e-commerce, low precision hurts seller retention.
    * **Recall:** $\frac{\text{True Duplicates Identified}}{\text{Total True Duplicates in the Sample}}$. If Recall is low, our algorithm has high *False Negatives*, meaning malicious sellers are successfully bypassing our system with slightly smarter text edits.

---

## 5. Failure Modes

1.  **The Boilerplate/Template Inflation (High False Positives):**
    * *The Flaw:* Many professional commercial sellers list completely different physical items using identical stock manufacturer descriptions (e.g., listing an iPhone 14 Pro Max and an iPhone 14 Pro using the exact same generic technical specifications sheet). Our shingle-Jaccard approach will read a 95% text overlap and flag them as duplicates, wrongfully hiding legitimate listings.
    * *What to watch:* Monitor the ratio of cross-seller flags. If LSH flags a massive cluster of duplicates across *different* seller IDs, it is highly likely a generic product template rather than a single spammer gaming the system.
2.  **The Visual Spam/Adversarial Padding (High False Negatives):**
    * *The Flaw:* Malicious sellers can easily reverse-engineer a text-only filter. If they append a massive string of random, unique keywords, a unique quote, or large blocks of emojis at the bottom of their description, the denominator of our Jaccard equation (the Union) will artificially balloon. This forces the Jaccard similarity down below our high 0.85 threshold, allowing the duplicate listing to pass through undetected.
    * *What to watch:* Monitor user reports ("Report duplicate listing" button on the UI). If user complaints remain high but our LSH pipeline reports low duplicate counts, sellers have successfully shifted to adversarial padding strategies.