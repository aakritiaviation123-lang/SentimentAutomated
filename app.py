import os
import io
import base64
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

LABELS = ['Negative', 'Neutral', 'Positive']
COLORS = {'Negative': '#F08080', 'Neutral': '#F0E68C', 'Positive': '#66CDAA'}


def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    probs = softmax(logits)
    return LABELS[probs.argmax()], float(probs.max())


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze/text', methods=['POST'])
def analyze_text():
    """Single-text quick analysis."""
    data = request.get_json()
    text = (data or {}).get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    sentiment, confidence = predict(text)
    return jsonify({'sentiment': sentiment, 'confidence': round(confidence * 100, 2)})


@app.route('/columns', methods=['POST'])
def get_columns():
    """Return CSV column names so the user can pick the text column."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    try:
        df = pd.read_csv(path, nrows=0)
        return jsonify({'columns': list(df.columns), 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze/csv', methods=['POST'])
def analyze_csv():
    """Full CSV analysis — returns summary stats + chart PNGs as base64."""
    filename = request.form.get('filename')
    col = request.form.get('column')
    mode = request.form.get('mode', 'limited')   # 'limited' = 5000 rows, 'full' = all

    if not filename or not col:
        return jsonify({'error': 'Missing filename or column'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.isfile(path):
        return jsonify({'error': 'File not found. Please re-upload.'}), 404

    if mode == 'limited':
        df = pd.read_csv(path, nrows=5000)
    else:
        df = pd.read_csv(path)
    if col not in df.columns:
        return jsonify({'error': f'Column "{col}" not found'}), 400

    if mode == 'limited':
        pass  # already limited
    else:
        pass  # full

    texts = df[col].astype(str)
    results = [predict(t) for t in texts]
    df[['Sentiment', 'Confidence']] = results
    df['score'] = df['Sentiment'].map({'Negative': -1, 'Neutral': 0, 'Positive': 1})

    # Save results CSV
    results_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    df.to_csv(results_path, index=False)

    sent_counts = df['Sentiment'].value_counts().reindex(LABELS, fill_value=0)
    avg_conf = round(df['Confidence'].mean() * 100, 2)
    high_conf = int(df[df['Confidence'] > 0.9].shape[0])

    charts = {}
    plt.style.use('dark_background')

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    bars = ax.bar(LABELS, sent_counts.values,
                  color=[COLORS[l] for l in LABELS], edgecolor='#FF6B00', linewidth=0.8)
    ax.set_title('Sentiment Count', color='#FF6B00', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#ccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    charts['bar'] = fig_to_b64(fig)
    plt.close(fig)

    # Pie chart
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.pie(sent_counts.values, labels=LABELS, autopct='%1.1f%%',
           colors=[COLORS[l] for l in LABELS],
           textprops={'color': '#eee'}, wedgeprops={'edgecolor': '#1a1a1a', 'linewidth': 2})
    ax.set_title('Sentiment Proportion', color='#FF6B00', fontsize=13, fontweight='bold')
    charts['pie'] = fig_to_b64(fig)
    plt.close(fig)

    # Confidence box plot
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    sns.boxplot(x='Sentiment', y='Confidence', data=df, order=LABELS,
                palette=COLORS, ax=ax, linewidth=1.2)
    ax.set_title('Confidence by Sentiment', color='#FF6B00', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#ccc')
    ax.set_xlabel('Sentiment', color='#ccc')
    ax.set_ylabel('Confidence', color='#ccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    charts['box'] = fig_to_b64(fig)
    plt.close(fig)

    return jsonify({
        'summary': {
            'total': len(df),
            'positive': int(sent_counts['Positive']),
            'neutral': int(sent_counts['Neutral']),
            'negative': int(sent_counts['Negative']),
            'avg_confidence': avg_conf,
            'high_confidence': high_conf,
            'dominant': sent_counts.idxmax()
        },
        'charts': charts
    })


@app.route('/download/results')
def download_results():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.isfile(path):
        return 'No results yet.', 404
    return send_file(path, as_attachment=True, download_name='results.csv')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
