"""
app.py — Sentiment Analysis Web Deployment
------------------------------------------
Run:  python app.py
Open: http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template_string
import pickle, re, os
import nltk
from nltk.corpus import stopwords
from nltk.stem   import PorterStemmer

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# ── Load model & vectoriser ───────────────────────────────────────────────────
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'<.*?>',     ' ', text)
    text = re.sub(r'&[a-z]+;',  ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+',       ' ', text).strip()
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sentiment Analysis — IMDB Review Classifier</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0D1117;
      color: #E6EDF3;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    /* ── Header ── */
    header {
      width: 100%;
      background: #161B22;
      border-bottom: 2px solid #00B4D8;
      padding: 18px 40px;
      display: flex;
      align-items: center;
      gap: 14px;
    }
    header h1 { font-size: 1.5rem; color: #E6EDF3; }
    header span { font-size: 1.8rem; }
    .badge {
      margin-left: auto;
      background: #1C2333;
      border: 1px solid #30363D;
      color: #8B949E;
      font-size: 0.78rem;
      padding: 5px 14px;
      border-radius: 20px;
    }

    /* ── Main card ── */
    main {
      width: 100%;
      max-width: 820px;
      padding: 40px 20px;
    }

    .card {
      background: #161B22;
      border: 1px solid #30363D;
      border-radius: 12px;
      padding: 32px;
      margin-bottom: 24px;
    }

    .card h2 {
      font-size: 1.1rem;
      color: #00B4D8;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    textarea {
      width: 100%;
      min-height: 160px;
      background: #0D1117;
      border: 1px solid #30363D;
      border-radius: 8px;
      color: #E6EDF3;
      font-size: 1rem;
      padding: 16px;
      resize: vertical;
      outline: none;
      transition: border-color 0.2s;
      font-family: 'Segoe UI', sans-serif;
      line-height: 1.6;
    }
    textarea:focus { border-color: #00B4D8; }
    textarea::placeholder { color: #484F58; }

    .char-count {
      text-align: right;
      font-size: 0.8rem;
      color: #484F58;
      margin-top: 6px;
    }

    /* ── Buttons ── */
    .btn-row { display: flex; gap: 12px; margin-top: 20px; }

    button {
      padding: 12px 28px;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-primary {
      background: #00B4D8;
      color: #0D1117;
      flex: 1;
    }
    .btn-primary:hover { background: #0096C7; }
    .btn-secondary {
      background: #1C2333;
      color: #8B949E;
      border: 1px solid #30363D;
    }
    .btn-secondary:hover { background: #21262D; }

    /* ── Result ── */
    #result-card { display: none; }

    .result-positive {
      border-color: #2ECC71 !important;
      background: linear-gradient(135deg, #161B22 0%, #0d2218 100%) !important;
    }
    .result-negative {
      border-color: #E74C3C !important;
      background: linear-gradient(135deg, #161B22 0%, #2d0d0d 100%) !important;
    }

    .result-label {
      font-size: 2.2rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }
    .positive-text { color: #2ECC71; }
    .negative-text { color: #E74C3C; }

    .confidence-bar-bg {
      background: #0D1117;
      border-radius: 99px;
      height: 12px;
      margin: 16px 0 6px;
      overflow: hidden;
    }
    .confidence-bar {
      height: 100%;
      border-radius: 99px;
      transition: width 0.8s ease;
    }
    .bar-positive { background: linear-gradient(90deg, #27AE60, #2ECC71); }
    .bar-negative { background: linear-gradient(90deg, #C0392B, #E74C3C); }

    .conf-label {
      font-size: 0.85rem;
      color: #8B949E;
      display: flex;
      justify-content: space-between;
    }

    /* ── Examples ── */
    .examples { display: flex; flex-direction: column; gap: 8px; }
    .example-btn {
      background: #1C2333;
      border: 1px solid #30363D;
      color: #8B949E;
      padding: 10px 16px;
      border-radius: 8px;
      text-align: left;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .example-btn:hover { border-color: #00B4D8; color: #E6EDF3; }
    .tag {
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 99px;
      margin-right: 8px;
    }
    .tag-pos { background: #0d2218; color: #2ECC71; }
    .tag-neg { background: #2d0d0d; color: #E74C3C; }

    /* ── Spinner ── */
    .spinner { display: none; text-align: center; padding: 16px; color: #00B4D8; }

    footer {
      color: #484F58;
      font-size: 0.8rem;
      padding: 20px;
      text-align: center;
    }
  </style>
</head>
<body>

<header>
  <span>◈</span>
  <h1>NLP Sentiment Analysis — IMDB Review Classifier</h1>
</header>

<main>
  <!-- Input card -->
  <div class="card">
    <h2>◉ Enter Movie Review</h2>
    <textarea id="review-input"
              placeholder="Type or paste a movie review here to analyse its sentiment…"
              oninput="updateCount()"></textarea>
    <div class="char-count"><span id="char-count">0</span> characters</div>
    <div class="btn-row">
      <button class="btn-primary" onclick="analyseReview()">▶  Analyse Sentiment</button>
      <button class="btn-secondary" onclick="clearInput()">✖ Clear</button>
    </div>
  </div>

  <!-- Spinner -->
  <div class="spinner" id="spinner">⟳  Analysing…</div>

  <!-- Result card -->
  <div class="card" id="result-card">
    <h2>◈ Result</h2>
    <div class="result-label" id="result-label"></div>
    <div class="confidence-bar-bg">
      <div class="confidence-bar" id="conf-bar" style="width:0%"></div>
    </div>
    <div class="conf-label">
      <span>Confidence</span>
      <span id="conf-text"></span>
    </div>
  </div>

  <!-- Example reviews -->
  <div class="card">
    <h2>◈ Try Example Reviews</h2>
    <div class="examples">
      <button class="example-btn" onclick="loadExample(this.dataset.text)"
              data-text="This movie was absolutely fantastic! The acting was superb, the storyline kept me engaged throughout, and the visuals were breathtaking. One of the best films I have seen in years. Highly recommended!">
        <span class="tag tag-pos">POSITIVE</span>
        "This movie was absolutely fantastic! The acting was superb…"
      </button>
      <button class="example-btn" onclick="loadExample(this.dataset.text)"
              data-text="What a complete waste of time. The plot made no sense, the acting was terrible, and the special effects looked cheap. I walked out halfway through and I want my money back. Avoid at all costs.">
        <span class="tag tag-neg">NEGATIVE</span>
        "What a complete waste of time. The plot made no sense…"
      </button>
      <button class="example-btn" onclick="loadExample(this.dataset.text)"
              data-text="A masterpiece of cinema. The director has crafted something truly special here. Every scene is beautifully shot and the performances from the entire cast are outstanding. This film will stay with you long after the credits roll.">
        <span class="tag tag-pos">POSITIVE</span>
        "A masterpiece of cinema. The director has crafted something truly special…"
      </button>
      <button class="example-btn" onclick="loadExample(this.dataset.text)"
              data-text="I cannot believe how bad this film was. The script was lazy, the direction was sloppy, and none of the characters were believable. Save yourself two hours and watch something else. Deeply disappointing.">
        <span class="tag tag-neg">NEGATIVE</span>
        "I cannot believe how bad this film was. The script was lazy…"
      </button>
    </div>
  </div>
</main>

<footer>NLP Assignment — Sentiment Analysis System  •  IMDB Dataset (50,000 reviews)  •  TF-IDF + ML Pipeline</footer>

<script>
  function updateCount() {
    const n = document.getElementById('review-input').value.length;
    document.getElementById('char-count').textContent = n;
  }

  function clearInput() {
    document.getElementById('review-input').value = '';
    document.getElementById('char-count').textContent = '0';
    document.getElementById('result-card').style.display = 'none';
    document.getElementById('spinner').style.display = 'none';
  }

  function loadExample(text) {
    document.getElementById('review-input').value = text;
    updateCount();
    document.getElementById('result-card').style.display = 'none';
  }

  async function analyseReview() {
    const text = document.getElementById('review-input').value.trim();
    if (!text) { alert('Please enter a review first.'); return; }

    document.getElementById('result-card').style.display = 'none';
    document.getElementById('spinner').style.display    = 'block';

    const resp = await fetch('/predict', {
      method : 'POST',
      headers: {'Content-Type':'application/json'},
      body   : JSON.stringify({ text }),
    });
    const data = await resp.json();

    document.getElementById('spinner').style.display = 'none';

    const card    = document.getElementById('result-card');
    const label   = document.getElementById('result-label');
    const bar     = document.getElementById('conf-bar');
    const confTxt = document.getElementById('conf-text');

    card.className = 'card ' + (data.sentiment === 'Positive' ? 'result-positive' : 'result-negative');
    label.innerHTML = data.sentiment === 'Positive'
      ? '<span>😊</span><span class="positive-text">POSITIVE</span>'
      : '<span>😞</span><span class="negative-text">NEGATIVE</span>';

    const pct = (data.confidence * 100).toFixed(1);
    bar.style.width = pct + '%';
    bar.className   = 'confidence-bar ' + (data.sentiment === 'Positive' ? 'bar-positive' : 'bar-negative');
    confTxt.textContent = pct + '%';

    card.style.display = 'block';
  }
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json()
    text   = data.get('text', '')
    clean  = preprocess(text)
    vec    = vectorizer.transform([clean])
    pred   = model.predict(vec)[0]

    # Confidence — use predict_proba if available (LR, NB) else decision_function (SVM)
    try:
        prob = model.predict_proba(vec)[0]
        conf = float(max(prob))
    except AttributeError:
        df_val = model.decision_function(vec)[0]
        conf   = float(1 / (1 + abs(df_val)))   # rough normalisation for SVM

    sentiment = 'Positive' if pred == 1 else 'Negative'
    return jsonify({'sentiment': sentiment, 'confidence': conf})

if __name__ == '__main__':
    print('\n NLP Sentiment Analysis — Web App')
    print(' Open: http://127.0.0.1:5000\n')
    app.run(debug=True)
