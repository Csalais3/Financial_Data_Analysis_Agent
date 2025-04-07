import numpy as np
import pandas as pd
import requests
import csv
import os
from collections import Counter
import umap.umap_ as umap
import plotly.express as px
import re
from sklearn.preprocessing import normalize
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity
import yaml

API_KEY = '3ea792e2a21c4a54a621191dd55a283d'
NEWSAPI_URL = 'https://newsapi.org/v2/Top-headlines'
PARAMS = {
    'q': '(business OR finance) AND (stock OR market OR economy)',
    'language': 'en',
    'sortBy': 'relevancy',
    'pageSize': 25,  
    'apiKey': API_KEY
}

DATA_PATH = "/Users/csalais3/Downloads/Financial_Data_Analysis_Agents/Data/news_data.csv"
CLUSTERED_PATH = "/Users/csalais3/Downloads/Financial_Data_Analysis_Agents/Data/clustered_news.csv"

# Load industry taxonomy
with open("/Users/csalais3/Downloads/Financial_Data_Analysis_Agents/Data/industry_taxonomy.yaml") as f:
    INDUSTRY_TAXONOMY = yaml.safe_load(f)

class TextProcessor:
    def __init__(self):
        if not INDUSTRY_TAXONOMY or 'industries' not in INDUSTRY_TAXONOMY:
            raise ValueError("Invalid industry taxonomy format")

        # Extract and flatten industry terms
        self.industry_terms = []
        for industry in INDUSTRY_TAXONOMY['industries']:
            self.industry_terms.append(industry['name'].lower())
            self.industry_terms.extend([kw.lower() for kw in industry.get('keywords', [])])
        
        self.industry_terms = list(set(self.industry_terms))  # Deduplicate
        
        # Create regex pattern for term boosting
        pattern = r'\b(' + '|'.join(map(re.escape, self.industry_terms)) + r')\b'
        self.boost_pattern = re.compile(pattern, flags=re.IGNORECASE)
        
        # Initialize vectorizer with combined stop words
        custom_stop_words = ['business', 'company', 'plan', 'said', 'new']
        default_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words())
        all_stop_words = list(set(default_stop_words + custom_stop_words))
        
        self.vectorizer = TfidfVectorizer(
            max_features=800,
            stop_words=all_stop_words,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
        )

    def preprocess_text(self, text):
        """Enhanced text preprocessing pipeline"""
        if pd.isna(text):
            return ''
            
        # Clean text
        text = self.boost_pattern.sub(lambda m: m.group().upper(), text)
        text = re.sub(r'\b(?:said|reported|announced|according|plans)\b', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        # Remove short words and empty strings
        cleaned = ' '.join([word for word in text.split() if len(word) > 3])
        return cleaned if cleaned.strip() else ''

class IndustryClusterer:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self._prepare_industry_embeddings()
        
        self.cluster_params = {
            'min_cluster_size': 10,
            'min_samples': 2,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom'
        }

    def _prepare_industry_embeddings(self):
        """Create industry reference embeddings"""
        industry_docs = [
            " ".join([ind['name']] + ind.get('keywords', []))
            for ind in INDUSTRY_TAXONOMY['industries']
        ]
        self.industry_embeddings = self.vectorizer.fit_transform(industry_docs)
        self.industry_names = [ind['name'] for ind in INDUSTRY_TAXONOMY['industries']]

    def _label_cluster(self, keywords):
        """Match cluster keywords to industry taxonomy"""
        cluster_text = " ".join(keywords)
        cluster_vec = self.vectorizer.transform([cluster_text])
        
        similarities = cosine_similarity(cluster_vec, self.industry_embeddings)
        best_match_idx = np.argmax(similarities)
        
        if similarities[0, best_match_idx] > 0.3:
            return self.industry_names[best_match_idx]
        else:
            return "Other"

    def cluster(self, texts):
        """
        Full clustering pipeline with validity checks.
        Returns a list of cluster labels (one per document).
        """
        if not texts or len(texts) < 10:
            raise ValueError("Insufficient data for clustering (minimum 10 documents)")
            
        # Vectorize
        tfidf_matrix = self.vectorizer.transform(texts)
        if tfidf_matrix.shape[0] == 0:
            raise ValueError("TF-IDF matrix has zero samples")
            
        # Dimensionality reduction
        reducer = umap.UMAP(
            n_components=min(15, len(texts)-1),
            metric='cosine',
            random_state=42
        )
        embeddings = reducer.fit_transform(tfidf_matrix)
        
        # HDBSCAN
        clusterer = hdbscan.HDBSCAN(**self.cluster_params)
        labels = clusterer.fit_predict(embeddings)  # shape: (# docs,)
        
        # Extract cluster keywords
        cluster_keywords = self._extract_cluster_keywords(tfidf_matrix, labels)
        
        # Build a dict: cluster_id -> label from taxonomy
        cluster_label_map = {}
        unique_clusters = np.unique(labels)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                # Outliers
                cluster_label_map[cluster_id] = "Outlier"
            else:
                keywords = cluster_keywords.get(cluster_id, [])
                cluster_label_map[cluster_id] = self._label_cluster(keywords)
        
        # Map each document's cluster ID to the label
        doc_labels = [cluster_label_map[l] for l in labels]
        return doc_labels

    def _extract_cluster_keywords(self, tfidf_matrix, labels):
        """
        Get top keywords for each cluster.
        Returns a dict: cluster_id -> list of top keywords
        """
        features = self.vectorizer.get_feature_names_out()
        cluster_keywords = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # ignore outliers for keywords
            mask = (labels == cluster_id)
            # Average TF-IDF across docs in this cluster
            scores = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
            top_indices = np.argsort(scores)[-15:][::-1]
            
            keywords = [features[idx] for idx in top_indices]
            cluster_keywords[cluster_id] = keywords[:10]
            
        return cluster_keywords

def fetch_articles():
    """Fetch articles with proper error handling"""
    all_articles = []
    page = 1
    headers = {'User-Agent': 'NewsClustering/1.0'}
    
    try:
        while len(all_articles) < 100:  # Limit to 100 articles
            PARAMS['page'] = page
            response = requests.get(
                NEWSAPI_URL,
                params=PARAMS,
                headers=headers,
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                break
                
            data = response.json()
            articles = data.get('articles', [])
            if not articles:
                break
                
            all_articles.extend(articles)
            page += 1
            
            if page > 5:
                break

    except Exception as e:
        print(f"Fetch Error: {str(e)}")
    
    return all_articles[:100]

def save_articles(articles):
    """Save articles to CSV with validation"""
    if not articles:
        print("No articles to save")
        return False
        
    try:
        with open(DATA_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['source', 'title', 'description'])
            writer.writeheader()
            for article in articles:
                writer.writerow({
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'title': article.get('title', ''),
                    'description': article.get('description', '')
                })
        return True
    except Exception as e:
        print(f"Save Error: {str(e)}")
        return False

def analyze_results(df):
    """Display cluster distribution and samples"""
    print("\nIndustry Distribution:")
    print(df['industry'].value_counts())
    
    print("\nSample Articles Per Industry:")
    for industry in df['industry'].unique():
        subset = df[df['industry'] == industry]
        samples = subset.sample(min(2, len(subset)), random_state=42)
        print(f"\nIndustry: {industry}")
        print(samples[['title', 'source']].to_string(index=False))

def visualize_clusters(df, texts):
    """ 3D visualization with Plotly + UMAP where label = industry."""
    try:
        # 1. Vectorize the final texts
        vectorizer = TfidfVectorizer(max_features=500)
        tfidf = vectorizer.fit_transform(texts)
        
        # 2. Reduce to 3D
        reducer = umap.UMAP(n_components=3, metric='cosine')
        embeddings = reducer.fit_transform(tfidf)
        
        # 3. Plotly 3D scatter
        fig = px.scatter_3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            color=df['industry'],               # color by industry
            hover_name=df['industry'],          # show industry as primary hover label
            hover_data={'title': df['title']},  # also show article title
            title="News Article Clusters by Industry",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
        fig.show()

    except Exception as e:
        print(f"Visualization Error: {str(e)}")

def main():
    """Main workflow with comprehensive validation"""
    # 1. Fetch and save articles
    articles = fetch_articles()
    if not articles:
        print("No articles fetched")
        return
        
    if not save_articles(articles):
        return

    # 2. Load and prepare data
    try:
        df = pd.read_csv(DATA_PATH)
        df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')
        df = df[df['text'].str.strip().ne('')]
        if df.empty:
            print("No valid articles after preprocessing.")
            return
    except Exception as e:
        print(f"Data Loading Error: {str(e)}")
        return

    # 3. Process text
    try:
        processor = TextProcessor()
        cleaned_texts = [processor.preprocess_text(text) for text in df['text']]
        
        # Filter out empty texts after processing
        valid_mask = [bool(t.strip()) for t in cleaned_texts]
        df = df[valid_mask].copy()
        cleaned_texts = [t for t in cleaned_texts if t.strip()]
        
        if len(cleaned_texts) < 10:
            print(f"Only {len(cleaned_texts)} valid documents. Need at least 10.")
            return
    except Exception as e:
        print(f"Processing Error: {str(e)}")
        return

    # 4. Cluster articles
    try:
        clusterer = IndustryClusterer(processor.vectorizer)
        doc_labels = clusterer.cluster(cleaned_texts)  # one label per doc
        df['industry'] = doc_labels
    except Exception as e:
        print(f"Clustering Error: {str(e)}")
        return

    # 5. Analyze and visualize
    analyze_results(df)
    visualize_clusters(df, df['text'].tolist())
    
    # 6. Save results
    try:
        df.to_csv(CLUSTERED_PATH, index=False)
        print(f"\nResults saved to {CLUSTERED_PATH}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    main()