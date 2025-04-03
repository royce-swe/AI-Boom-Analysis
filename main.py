import arxiv
from arxiv import Client, Search

import pandas as pd

import nltk
import spacy
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK resources
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# arxiv search for AI/Machine Learning - related papers
search = Search(
    query="artificial intelligence OR machine learning",
    max_results=500,
    sort_by=arxiv.SortCriterion.Relevance
)

client = Client()
results = list(client.results(search))

# Fetch abstracts and metadata
papers = []
for result in results:
    papers.append({
        "title": result.title,
        "abstract": result.summary,
        "published": result.published
    })

df = pd.DataFrame(papers)
print(df.head())

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Handle potential NaN values
    if pd.isna(text):
        return ""
    
    text = str(text).lower()  # Ensure text is a string and lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    
    # Use regex for tokenization
    words = re.findall(r'\b\w+\b', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    
    # Process text with spaCy in one go
    doc = nlp(" ".join(words))
    words = [token.lemma_ for token in doc]  # Lemmatization

    return " ".join(words)

df["cleaned_abstract"] = df["abstract"].apply(preprocess_text)
print(df["cleaned_abstract"].head())

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Convert text to matrix of token counts
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
doc_term_matrix = vectorizer.fit_transform(df["cleaned_abstract"])

# Apply LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(doc_term_matrix)

# Display topics
words = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    print(f"Topic {i+1}: {[words[i] for i in topic.argsort()[-10:]]}")

df["published"] = pd.to_datetime(df["published"])
df["year"] = df["published"].dt.year

# Count papers per year
yearly_counts = df["year"].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.plot(yearly_counts.index, yearly_counts.values, marker='o')
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.title("Number of AI/ML Papers Published Over Time")
plt.tight_layout()
plt.savefig('ai_ml_papers_trend.png')
plt.close()

print(yearly_counts)
print("Analysis complete. Trend plot saved as 'ai_ml_papers_trend.png'.")

# ============================
# 🚀 Trending Phrases (Bigrams)
# ============================

# Extract bigrams
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words="english")
X_bigrams = bigram_vectorizer.fit_transform(df["cleaned_abstract"])

# Convert to DataFrame
bigram_counts = pd.DataFrame(X_bigrams.toarray(), columns=bigram_vectorizer.get_feature_names_out())
bigram_counts["year"] = df["year"]

# Aggregate by year
bigram_trends = bigram_counts.groupby("year").sum().T

# Select top 10 most frequent bigrams
top_bigrams = bigram_trends.sum(axis=1).sort_values(ascending=False).head(10).index
filtered_trends = bigram_trends.loc[top_bigrams]

# 📈 Plot the trends of top bigrams
plt.figure(figsize=(12, 6))
for bigram in filtered_trends.index:
    plt.plot(filtered_trends.columns, filtered_trends.loc[bigram], marker='o', label=bigram)

plt.xlabel("Year")
plt.ylabel("Frequency")
plt.title("Trending Bigrams in AI Research Over Time")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('trending_ngrams.png')
plt.close()

print("N-gram trend graph saved as 'trending_ngrams.png'.")

# ============================
# 🌎 AI Policy Data
# ============================

# Search for AI regulation/policy-related papers
search = arxiv.Search(
    query='"AI regulation" OR "AI governance" OR "AI policy" OR "AI ethics"',
    max_results=500,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

client = arxiv.Client()
results = list(client.results(search))

# Extract metadata
papers = []
for result in results:
    papers.append({
        "title": result.title,
        "abstract": result.summary,
        "published": result.published
    })

# Convert to DataFrame
df = pd.DataFrame(papers)

# Convert published date to year
df["published"] = pd.to_datetime(df["published"])
df["year"] = df["published"].dt.year

# Count papers per year
yearly_counts = df["year"].value_counts().sort_index()

# 📊 Plot trend over time
plt.figure(figsize=(10, 5))
plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.title("AI Regulation & Policy Research Over Time")
plt.grid(False)
plt.tight_layout()
plt.savefig('ai_policy_trend.png')
plt.close()

print("Trend analysis complete. Graph saved as 'ai_policy_trend.png'.")