import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WEIGHTS = {"like": 2.0, "view": 1.0}

def load():
    ev = pd.read_csv("data/events.csv")
    ev["text"] = (ev.category + " ") * 2 + ev.tags + " " + ev.description
    return ev, pd.read_csv("data/interactions.csv")

def recommend(user_id, top_n=3):
    events, inter = load()
    X = TfidfVectorizer(stop_words="english").fit_transform(events.text)
    hist = inter[inter.user_id == user_id]
    if hist.empty:                                   # cold start -> popularity
        pop = inter.event_id.value_counts().index.tolist()
        return events[events.event_id.isin(pop)].head(top_n)[["title", "category"]]
    idx = events.reset_index().set_index("event_id").loc[hist.event_id, "index"].values
    w = hist.action.map(WEIGHTS).values
    profile = np.asarray((X[idx].multiply(w[:, None])).sum(axis=0))   # weighted user profile
    scores = cosine_similarity(profile, X).ravel()
    scores[idx] = -1                                 # don't recommend already-seen events
    best = scores.argsort()[::-1][:top_n]
    out = events.iloc[best][["title", "category"]].copy()
    out["score"] = scores[best].round(3)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    print(recommend(a.user, a.top).to_string(index=False))
