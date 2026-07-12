# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Real-world platforms like Spotify and YouTube predict what you'll love next by blending two ideas: **collaborative filtering** (learning from the behavior of millions of listeners — what people with similar taste play, like, skip, and group into playlists) and **content-based filtering** (matching the actual attributes of songs, such as genre, tempo, mood, and acoustic qualities). They fuse these into large hybrid models that also factor in context like time of day and recent activity. My version is a small, transparent **content-based** recommender: with no crowd data available, it prioritizes matching a single user's stated taste to the measurable attributes of each song, and — just as importantly — it prioritizes being *explainable*, so every recommendation comes with a plain-language reason for why it was chosen. It favors clarity over sophistication: a simple weighted score that a person can read and understand, rather than a black box.

### Features used in the simulation

Each **`Song`** stores these attributes (from `data/songs.csv`):

- `id`, `title`, `artist` — identity/display only
- `genre` — e.g. pop, lofi, rock (primary matching signal)
- `mood` — e.g. happy, chill, intense
- `energy` — 0.0–1.0 intensity
- `tempo_bpm` — beats per minute
- `valence` — 0.0–1.0 musical positivity
- `danceability` — 0.0–1.0
- `acousticness` — 0.0–1.0 (acoustic vs. produced/electronic)

Each **`UserProfile`** stores the taste preferences the score is built from:

- `favorite_genre` — the genre the user most wants to hear
- `favorite_mood` — the feeling/context they're after
- `target_energy` — the energy level they're aiming for (0.0–1.0)
- `likes_acoustic` — whether they prefer an acoustic or a produced sound

### How a score is computed

The `Recommender` applies a **weighted scoring rule** to each song. Every song's score is the sum of four terms, with genre weighted most heavily:

| Feature | Weight | How it scores |
|---|---|---|
| genre | **3.0** | exact match → full points; otherwise **partial credit** by similarity (see below) |
| mood | **2.0** | exact match → full points; otherwise **partial credit** by similarity |
| energy | 2.0 | `2 × (1 − |song.energy − target_energy|)` — closer is better |
| acousticness | 1.0 | rewards high values if `likes_acoustic`, low values otherwise |

**Partial credit for genre & mood (additive similarity).** An exact genre or mood match still earns the full weight, so exact matches always win. When a song's genre/mood *doesn't* match, instead of scoring zero it earns a fraction of the weight based on how similar it *sounds* to the user's favorite. That similarity is **objective and data-derived**, not hand-tuned:

- For the favorite genre (and, separately, the favorite mood), the system computes a **centroid** — the average feature vector of every song in the catalog carrying that label.
- A candidate song's closeness is `1 − mean(|difference|)` between its features and that centroid, giving a value in 0–1.
- The similarity is only measured over `tempo_bpm`, `valence`, and `danceability` — the three features **not** already scored elsewhere, so nothing is double-counted. `tempo_bpm` is scaled to 0–1 using the catalog's own min/max; the other two are already on a 0–1 scale.
- The closeness is **squared** before use, so strong matches stay strong while mediocre ones are pushed down. This keeps exact matches clearly ahead and prevents the ranking from flattening.

Each contributing term also records a short reason (e.g. *"matches your favorite genre (lofi)"* or *"sounds similar to your favorite genre (lofi)"*), and those reasons become the explanation shown with the recommendation.

### Finalized Algorithm Recipe

```
For each song, score = genre_term + mood_term + energy_term + acoustic_term

  genre_term    = 3.0                          if song.genre == favorite_genre
                = 3.0 × similarity(song, favorite_genre_centroid)²   otherwise
  mood_term     = 2.0                          if song.mood == favorite_mood
                = 2.0 × similarity(song, favorite_mood_centroid)²    otherwise
  energy_term   = 2.0 × (1 − |song.energy − target_energy|)
  acoustic_term = 1.0 × (song.acousticness if likes_acoustic else 1 − song.acousticness)

  where similarity(song, centroid) = 1 − mean(|Δtempo_norm|, |Δvalence|, |Δdanceability|)
        and tempo is min/max-scaled to 0–1 across the catalog.
```

### How songs are chosen

A separate **ranking rule** scores *every* song with the recipe above, sorts them from highest to lowest, and returns the top `k`. Splitting scoring (judging one song) from ranking (comparing and selecting the whole list) keeps the scoring logic simple and reusable, and makes it easy to change weights or selection behavior independently.

```
recommend_songs (ranking)
   ├── build data-derived centroids once from the catalog
   ├── score_song(user, song)  ← run once per song
   ├── sort by score, high → low
   └── return top k
```

### Potential biases I expect

This system makes deliberate choices, and each one introduces a bias worth naming:

- **Genre is weighted highest (3.0).** The system may over-prioritize genre and bury great songs that nail the user's *mood* but sit in a different genre. A perfect chill track in an "unexpected" genre can lose to a mediocre in-genre one.
- **Popularity/representation bias in the centroids.** A genre or mood with only one or two songs in the catalog has a shaky, noisy centroid (a 1-song genre's centroid *is* that song). Well-represented genres get more stable, more trustworthy similarity scores — so the recommender is implicitly fairer to whatever is already common in the catalog.
- **Filter bubble / over-specialization.** Because it optimizes for "more of what you already like," it rarely broadens taste. Highly relevant surprises from adjacent styles are always ranked below the exact-match block.
- **The similarity features are weak genre discriminators.** By excluding energy and acousticness (to avoid double-counting), similarity rests on tempo, valence, and danceability — which don't always separate very different genres well, so an occasional odd "sounds similar" pairing can slip through.
- **Only measurable attributes count.** The system has no understanding of lyrics, language, culture, or *why* a person loves a song — so it can systematically under-serve tastes that don't reduce to these numeric features.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Output of `python -m src.main` for the default profile `{genre: pop, mood: happy, energy: 0.8}`:

```
Loaded songs: 18

Top recommendations:

Sunrise City - Score: 7.78
Because: matches your favorite genre (pop), fits the happy mood you like, energy level is close to what you want, has the produced/electronic sound you prefer

Rooftop Lights - Score: 7.51
Because: sounds similar to your favorite genre (pop), fits the happy mood you like, energy level is close to what you want, has the produced/electronic sound you prefer

Gym Hero - Score: 7.39
Because: matches your favorite genre (pop), has a feel similar to the happy mood you like, energy level is close to what you want, has the produced/electronic sound you prefer

Concrete Sunrise - Score: 6.68
Because: sounds similar to your favorite genre (pop), has a feel similar to the happy mood you like, energy level is close to what you want, has the produced/electronic sound you prefer

Basement Pulse - Score: 6.28
Because: sounds similar to your favorite genre (pop), has a feel similar to the happy mood you like, energy level is close to what you want, has the produced/electronic sound you prefer
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



