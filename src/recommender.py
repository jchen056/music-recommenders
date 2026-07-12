import csv
from typing import List, Dict, Tuple, Optional, Iterable, Any
from dataclasses import dataclass

# --- Scoring weights -------------------------------------------------------
# A matching genre is worth more than a matching mood, which in turn is
# weighted alongside how closely the song's energy matches the user's target.
GENRE_WEIGHT = 3.0
MOOD_WEIGHT = 2.0
ENERGY_WEIGHT = 2.0
ACOUSTIC_WEIGHT = 1.0

# Numeric fields loaded from the CSV that should be parsed as floats.
NUMERIC_FIELDS = ("energy", "tempo_bpm", "valence", "danceability", "acousticness")

# Features used ONLY for the objective genre/mood similarity fallback.
# Deliberately excludes energy and acousticness (already scored directly) so
# nothing is double-counted. tempo_bpm is scaled to 0-1; valence and
# danceability are already on a 0-1 scale.
SIMILARITY_FIELDS = ("tempo_bpm", "valence", "danceability")

# A non-exact genre/mood match only earns a spoken "sounds similar" reason
# once its similarity clears this bar (it still contributes points below it).
SIMILARITY_REASON_THRESHOLD = 0.5


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


def _read(song: Any, field: str, default: Any = None) -> Any:
    """Read a field from either a Song dataclass or a plain dict."""
    if isinstance(song, dict):
        return song.get(field, default)
    return getattr(song, field, default)


def _as_features(song: Any) -> Dict[str, Any]:
    """Normalize a Song or dict into a uniform feature dict of the right types."""
    return {
        "genre": _read(song, "genre", "") or "",
        "mood": _read(song, "mood", "") or "",
        "energy": float(_read(song, "energy", 0.0) or 0.0),
        "acousticness": float(_read(song, "acousticness", 0.0) or 0.0),
        "tempo_bpm": float(_read(song, "tempo_bpm", 0.0) or 0.0),
        "valence": float(_read(song, "valence", 0.0) or 0.0),
        "danceability": float(_read(song, "danceability", 0.0) or 0.0),
    }


class CatalogStats:
    """
    Objective, data-derived statistics used for the genre/mood similarity
    fallback. All numbers come from the catalog itself -- nothing is
    hand-tuned. Centroids are the average feature vector of every song sharing
    a given genre (or mood), computed over SIMILARITY_FIELDS.
    """

    def __init__(self, songs: Iterable[Any]):
        """Derive tempo scaling bounds and per-genre/mood centroids from the catalog."""
        feats = [_as_features(s) for s in songs]
        tempos = [f["tempo_bpm"] for f in feats]
        # tempo is scaled to 0-1 using the catalog's own min/max (data-driven).
        self.tempo_min = min(tempos) if tempos else 0.0
        self.tempo_max = max(tempos) if tempos else 0.0
        self.genre_centroids = self._centroids(feats, "genre")
        self.mood_centroids = self._centroids(feats, "mood")

    def _norm_vec(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Scale the similarity features to 0-1 (only tempo needs scaling)."""
        span = self.tempo_max - self.tempo_min
        tempo_n = (features["tempo_bpm"] - self.tempo_min) / span if span else 0.0
        return {
            "tempo_bpm": tempo_n,
            "valence": features["valence"],
            "danceability": features["danceability"],
        }

    def _centroids(self, feats: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
        """Average the normalized similarity features of every song sharing each genre/mood label."""
        groups: Dict[str, List[Dict[str, float]]] = {}
        for f in feats:
            groups.setdefault(f[key], []).append(self._norm_vec(f))
        centroids: Dict[str, Dict[str, float]] = {}
        for label, vecs in groups.items():
            centroids[label] = {
                field: sum(v[field] for v in vecs) / len(vecs)
                for field in SIMILARITY_FIELDS
            }
        return centroids

    def similarity(self, features: Dict[str, Any], centroid: Dict[str, float]) -> float:
        """
        Objective closeness in 0-1: 1 minus the mean absolute difference
        between this song's normalized features and a centroid.
        """
        vec = self._norm_vec(features)
        mean_diff = sum(abs(vec[f] - centroid[f]) for f in SIMILARITY_FIELDS) / len(SIMILARITY_FIELDS)
        return 1.0 - mean_diff


def _score_attributes(
    *,
    features: Dict[str, Any],
    favorite_genre: Optional[str],
    favorite_mood: Optional[str],
    target_energy: Optional[float],
    likes_acoustic: bool,
    stats: Optional[CatalogStats] = None,
) -> Tuple[float, List[str]]:
    """
    Core content-based scoring shared by the functional and OOP APIs.

    Genre and mood use additive partial credit: an exact match earns the full
    weight; a non-exact match earns weight x similarity, where similarity is an
    objective, data-derived closeness to the favorite genre's/mood's centroid
    (see CatalogStats). With no stats provided, only exact matches score.

    Returns (score, reasons) where reasons lists the positive contributors
    that can be shown to the user as an explanation.
    """
    score = 0.0
    reasons: List[str] = []
    genre = features["genre"]
    mood = features["mood"]

    # Genre: exact match earns full weight; otherwise partial credit by
    # objective similarity to the favorite genre's centroid.
    if favorite_genre and genre == favorite_genre:
        score += GENRE_WEIGHT
        reasons.append(f"matches your favorite genre ({genre})")
    elif favorite_genre and stats and favorite_genre in stats.genre_centroids:
        # Square the similarity so strong matches stay strong while mediocre
        # ones are pushed down -- keeps exact matches clearly ahead and re-widens
        # the ranking, without a hard cap (full-range partial credit preserved).
        sim = stats.similarity(features, stats.genre_centroids[favorite_genre]) ** 2
        score += GENRE_WEIGHT * sim
        if sim >= SIMILARITY_REASON_THRESHOLD:
            reasons.append(f"sounds similar to your favorite genre ({favorite_genre})")

    # Mood: same additive scheme.
    if favorite_mood and mood == favorite_mood:
        score += MOOD_WEIGHT
        reasons.append(f"fits the {mood} mood you like")
    elif favorite_mood and stats and favorite_mood in stats.mood_centroids:
        sim = stats.similarity(features, stats.mood_centroids[favorite_mood]) ** 2
        score += MOOD_WEIGHT * sim
        if sim >= SIMILARITY_REASON_THRESHOLD:
            reasons.append(f"has a feel similar to the {favorite_mood} mood you like")

    # Energy: reward closeness to the target (both are on a 0-1 scale).
    if target_energy is not None:
        energy = features["energy"]
        score += ENERGY_WEIGHT * (1.0 - abs(energy - target_energy))
        if abs(energy - target_energy) <= 0.2:
            reasons.append("energy level is close to what you want")

    # Acousticness: reward high values for acoustic lovers, low values otherwise.
    acousticness = features["acousticness"]
    acoustic_fit = acousticness if likes_acoustic else (1.0 - acousticness)
    score += ACOUSTIC_WEIGHT * acoustic_fit
    if likes_acoustic and acousticness >= 0.6:
        reasons.append("has the acoustic sound you prefer")
    elif not likes_acoustic and acousticness <= 0.4:
        reasons.append("has the produced/electronic sound you prefer")

    return score, reasons


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Store the catalog and derive its similarity stats once for reuse."""
        self.songs = songs
        # Centroids/stats are derived once from the catalog and reused.
        self._stats = CatalogStats(songs) if songs else None

    def _score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Score one song for a user, returning (score, reasons)."""
        return _score_attributes(
            features=_as_features(song),
            favorite_genre=user.favorite_genre,
            favorite_mood=user.favorite_mood,
            target_energy=user.target_energy,
            likes_acoustic=user.likes_acoustic,
            stats=self._stats,
        )

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs for a user, ranked by score descending."""
        ranked = sorted(
            self.songs,
            key=lambda song: self._score(user, song)[0],
            reverse=True,
        )
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Build a human-readable one-line explanation of a song's score for a user."""
        score, reasons = self._score(user, song)
        if reasons:
            return f"{song.title} (score {score:.2f}): " + ", ".join(reasons) + "."
        return f"{song.title} (score {score:.2f}): no strong match, but it's a reasonable option."


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song = dict(row)
            song["id"] = int(song["id"])
            for field in NUMERIC_FIELDS:
                if field in song and song[field] != "":
                    song[field] = float(song[field])
            songs.append(song)
    return songs


def score_song(user_prefs: Dict, song: Dict, catalog: Optional[List[Dict]] = None) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py

    user_prefs keys (all optional): "genre", "mood", "energy", "likes_acoustic".
    Pass `catalog` (the full song list) to enable the objective genre/mood
    similarity fallback; without it, only exact genre/mood matches score.
    Returns (score, reasons).
    """
    stats = CatalogStats(catalog) if catalog else None
    return _score_attributes(
        features=_as_features(song),
        favorite_genre=user_prefs.get("genre"),
        favorite_mood=user_prefs.get("mood"),
        target_energy=user_prefs.get("energy"),
        likes_acoustic=bool(user_prefs.get("likes_acoustic", False)),
        stats=stats,
    )


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Returns a list of (song_dict, score, explanation) sorted by score desc.
    """
    # Build the data-derived stats once and reuse across all songs.
    stats = CatalogStats(songs) if songs else None

    scored = []
    for song in songs:
        score, reasons = _score_attributes(
            features=_as_features(song),
            favorite_genre=user_prefs.get("genre"),
            favorite_mood=user_prefs.get("mood"),
            target_energy=user_prefs.get("energy"),
            likes_acoustic=bool(user_prefs.get("likes_acoustic", False)),
            stats=stats,
        )
        explanation = ", ".join(reasons) if reasons else "no strong match, but it's a reasonable option"
        scored.append((song, score, explanation))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
