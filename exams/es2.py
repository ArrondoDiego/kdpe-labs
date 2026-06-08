'''
Scenario. You join a second-hand marketplace. Sellers game search by re-posting the same item many times with small edits — a changed word, a new caption, a tweaked price — so results fill up with near-duplicates and buyers complain. Your manager asks: "Flag near-duplicate listings so we show each item once." You have ~3 million active listings, each a short text blob (title + description). There are no labels telling you which are duplicates.

Your task — before writing any code, design the approach. Address each of the following:

Representation. How do you turn a listing into something comparable — words, shingles, embeddings? Why that choice for this problem?
Similarity & scale. What similarity measure fits "near-duplicate"? Why is all-pairs comparison infeasible at 3M listings, and what from the course makes it feasible?
Parameters & threshold. What knobs does your method have, and how do you set them to catch near-duplicates specifically (not merely "similar" items)?
No ground truth. You have no labels. How would you build a small evaluation, and what would you measure (think precision vs recall)?
Failure modes. Name two ways your design could be wrong, and what you'd watch for.
There is no single correct answer. Marks are for justified, internally consistent choices — and for spotting the weaknesses of your own design.
'''
