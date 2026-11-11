# AI-Based Personalized Event Guide
Machine-learning recommendation engine that learns a user's preferences from their behaviour and suggests events.

**Stack:** Python, scikit-learn, pandas, Recommendation algorithms

## Approach
- **Content-based filtering:** TF-IDF on event category/tags/description + cosine similarity.
- **User profile:** weighted average of the vectors of events the user liked/attended (likes weigh more than views).
- **Cold start:** falls back to popularity when a user has no history.

## Run
```bash
pip install -r requirements.txt
python recommender.py --user u1 --top 3
```
