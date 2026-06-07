"""
demo_d3_user_movie_embeddings.py
================================

Tiny recommender embedding demo for Session 07.

Concept
-------
A matrix-factorization recommender is just two embedding tables — one for
users, one for movies — trained so that  user_vec · movie_vec  approximates
the rating.  Same idea as word2vec, just applied to (user, item) pairs.

We use **2-dimensional** embeddings on purpose so we can plot them directly,
and we **snapshot them during training** to watch random noise turn into
structure: same-genre movies cluster, each pure-taste user lands among the
movies they like, and **blended-taste users land between the genres they
enjoy**.

The whole model is ~8 lines.  The whole training loop is ~5 lines.
That is the teaching moment.

Run
---
    pip install torch numpy matplotlib
    python demo_d3_user_movie_embeddings.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. Synthetic world: 5 genres × 10 movies, 50 pure-taste + 12 blended users
# ---------------------------------------------------------------------------
GENRES = ["Action", "Romance", "Sci-Fi", "Comedy", "Horror"]
GENRE_COLORS = ["#e63946", "#e9a23b", "#2a9d8f", "#457b9d", "#7d3c98"]

MOVIES_BY_GENRE = {
    "Action":  ["Mad Max", "Die Hard", "John Wick", "Top Gun", "Rambo",
                "Speed", "Heat", "Kill Bill", "The Raid", "Mission Impossible"],
    "Romance": ["Notting Hill", "Titanic", "La La Land", "The Notebook",
                "Pride and Prejudice", "Casablanca", "Roman Holiday",
                "Before Sunrise", "Amelie", "Brooklyn"],
    "Sci-Fi":  ["Interstellar", "Blade Runner", "The Matrix", "Arrival",
                "Inception", "Dune", "2001", "Alien", "Ex Machina", "Gattaca"],
    "Comedy":  ["Anchorman", "Superbad", "Bridesmaids", "The Hangover",
                "Airplane", "Groundhog Day", "Some Like It Hot",
                "Step Brothers", "Booksmart", "Borat"],
    "Horror":  ["The Shining", "Hereditary", "Get Out", "The Exorcist",
                "Halloween", "A Quiet Place", "It Follows", "Midsommar",
                "The Witch", "Sinister"],
}

movie_title, movie_genre = [], []
for g_id, g_name in enumerate(GENRES):
    for title in MOVIES_BY_GENRE[g_name]:
        movie_title.append(title)
        movie_genre.append(g_id)
movie_genre = np.array(movie_genre)
N_M = len(movie_title)

# 10 pure fans per genre = 50 users; each user_taste is a *set* of liked genres
user_taste = []
for g in range(5):
    user_taste.extend([{g}] * 10)

# 12 blended-taste users across 3 cross-genre combinations
BLENDS = [
    ({0, 2}, "Action+Sci-Fi"),     # superhero / blockbuster fans
    ({1, 3}, "Romance+Comedy"),    # rom-com fans
    ({2, 4}, "Sci-Fi+Horror"),     # cosmic-horror fans
]
blend_label_per_user = [None] * len(user_taste)
for blend_set, label in BLENDS:
    for _ in range(4):
        user_taste.append(blend_set)
        blend_label_per_user.append(label)

N_U = len(user_taste)

# Build ratings: rating ~ 4.5 if any of user's preferred genres == movie genre, else ~ 1.5
rows = []
for u in range(N_U):
    for m in range(N_M):
        match = movie_genre[m] in user_taste[u]
        r = (4.5 if match else 1.5) + rng.normal(0, 0.4)
        rows.append((u, m, np.clip(r, 1.0, 5.0)))
ratings = np.array(rows, dtype=np.float32)

u_idx = torch.tensor(ratings[:, 0], dtype=torch.long)
m_idx = torch.tensor(ratings[:, 1], dtype=torch.long)
y     = torch.tensor(ratings[:, 2], dtype=torch.float32)
y_mean = y.mean()

print(f"World: {N_U} users  ×  {N_M} movies  ×  {len(GENRES)} genres "
      f"→ {len(ratings):,} ratings")

# ---------------------------------------------------------------------------
# 2. The whole model
# ---------------------------------------------------------------------------
DIM = 2  # 2D so we can plot directly

class MF(nn.Module):
    def __init__(self, n_users, n_movies, dim):
        super().__init__()
        self.U = nn.Embedding(n_users, dim)
        self.M = nn.Embedding(n_movies, dim)
        nn.init.normal_(self.U.weight, std=0.1)
        nn.init.normal_(self.M.weight, std=0.1)

    def forward(self, u, m):
        return (self.U(u) * self.M(m)).sum(-1) + y_mean   # dot product + global bias

model = MF(N_U, N_M, DIM)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

# ---------------------------------------------------------------------------
# 3. Train and snapshot the embeddings as they move
# ---------------------------------------------------------------------------
SNAPSHOTS = [0, 5, 25, 100, 500]
N_EPOCHS = SNAPSHOTS[-1] + 1

snaps = {}
losses = []
for epoch in range(N_EPOCHS):
    if epoch in SNAPSHOTS:
        snaps[epoch] = (
            model.U.weight.detach().numpy().copy(),
            model.M.weight.detach().numpy().copy(),
        )
    pred = model(u_idx, m_idx)
    loss = ((pred - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

print(f"final MSE = {losses[-1]:.3f}   (rating range 1-5)")

# ---------------------------------------------------------------------------
# 4. Visualise: one panel per snapshot, plus a loss curve underneath
# ---------------------------------------------------------------------------

# Per-user colour: pure fan → genre colour; blended → mean of the two genre colours.
def user_color(taste_set):
    rgbs = np.array([to_rgb(GENRE_COLORS[g]) for g in taste_set])
    return rgbs.mean(axis=0)

user_colors = np.array([user_color(t) for t in user_taste])
is_blend = np.array([len(t) > 1 for t in user_taste])

fig = plt.figure(figsize=(4 * len(SNAPSHOTS), 5.8))
gs = fig.add_gridspec(2, len(SNAPSHOTS), height_ratios=[4, 1], hspace=0.55)

# shared axis limits across snapshots so the "growing into shape" effect is visible
all_pts = np.vstack([np.vstack([U, M]) for (U, M) in snaps.values()])
lim = max(abs(all_pts.min()), abs(all_pts.max())) * 1.15

for col, ep in enumerate(SNAPSHOTS):
    ax = fig.add_subplot(gs[0, col])
    U, M = snaps[ep]

    # movies as squares, coloured by genre
    for g in range(len(GENRES)):
        mask = movie_genre == g
        ax.scatter(M[mask, 0], M[mask, 1],
                   c=GENRE_COLORS[g], s=70, marker='s',
                   edgecolor='black', linewidth=0.4, alpha=0.85,
                   label=f'{GENRES[g]} movie' if col == 0 else None,
                   zorder=2)

    # pure-taste users as triangles (genre colour); blended users as stars (mixed colour)
    pure = ~is_blend
    ax.scatter(U[pure, 0], U[pure, 1],
               c=user_colors[pure], s=55, marker='^',
               edgecolor='black', linewidth=0.3, alpha=0.85,
               label='pure-taste user' if col == 0 else None, zorder=3)
    ax.scatter(U[is_blend, 0], U[is_blend, 1],
               c=user_colors[is_blend], s=130, marker='*',
               edgecolor='black', linewidth=0.5,
               label='blended-taste user' if col == 0 else None, zorder=4)

    # On the final panel, annotate one example blended user per blend type so we
    # can point out: "this user likes two genres → lands between the two clusters".
    if col == len(SNAPSHOTS) - 1:
        seen = set()
        for u in range(N_U):
            label = blend_label_per_user[u]
            if label is not None and label not in seen:
                seen.add(label)
                ax.annotate(label, (U[u, 0], U[u, 1]),
                            fontsize=7, alpha=0.9, fontweight='bold',
                            xytext=(8, 8), textcoords='offset points',
                            arrowprops=dict(arrowstyle='-', color='black',
                                            lw=0.5, alpha=0.7))

    ax.set_title(f'Epoch {ep}', fontsize=11)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.axhline(0, c='gray', lw=0.5)
    ax.axvline(0, c='gray', lw=0.5)
    if col == 0:
        ax.legend(fontsize=6.5, loc='lower left', framealpha=0.85)

ax_loss = fig.add_subplot(gs[1, :])
ax_loss.plot(losses, c='#264653')
for ep in SNAPSHOTS:
    ax_loss.axvline(ep, c='gray', lw=0.6, alpha=0.6)
ax_loss.set_xlabel('epoch')
ax_loss.set_ylabel('MSE loss')
ax_loss.set_title('Training loss')
ax_loss.grid(alpha=0.3)

fig.suptitle(
    'Recommender embeddings learn genre structure from ratings alone\n'
    f'({N_U} users × {N_M} movies × {len(GENRES)} genres, '
    f'dim={DIM}, dot-product matrix factorisation)',
    fontsize=13,
)
out = 'embeddings_evolution.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
print(f'Saved → {out}')
plt.show()
