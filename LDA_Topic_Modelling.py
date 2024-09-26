import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import matplotlib.pyplot as plt
import nltk

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

if __name__ == '__main__':
    movie_reviews_path = 'IMDB Dataset.csv'
    movie_reviews_df = pd.read_csv(movie_reviews_path)
    movie_reviews_df = movie_reviews_df.head(1000)

    # Stopwords
    stop_words = set(stopwords.words('english'))
    custom_stop_words = stop_words.union({'movie', 'film', 'scene', 'character', 'plot', 'actor'})

    # Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Text Preprocessing Function
    def preprocess_text(text):
        sentences = sent_tokenize(text)
        processed_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            filtered_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in custom_stop_words]
            processed_sentences.append(' '.join(filtered_words))

        return ' '.join(processed_sentences)

    # Apply Preprocessing
    movie_reviews_df['Processed Review'] = movie_reviews_df['review'].apply(preprocess_text)

    # Try TF-IDF instead of CountVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=5, stop_words=custom_stop_words)
    doc_term_matrix = vectorizer.fit_transform(movie_reviews_df['Processed Review'])

    terms = vectorizer.get_feature_names_out()
    id2word = corpora.Dictionary([doc.split() for doc in movie_reviews_df['Processed Review']])
    corpus_gensim = [id2word.doc2bow(doc.split()) for doc in movie_reviews_df['Processed Review']]

    # Optimize the number of topics and hyperparameters
    best_coherence = 0
    best_model = None
    best_num_topics = 0

    for num_topics in range(5, 21):  # Try from 5 to 20 topics
        lda_model = LdaModel(corpus=corpus_gensim, id2word=id2word, num_topics=num_topics, 
                             alpha='auto', eta='auto', passes=40, iterations=800, random_state=100)
        coherence_model_lda = CoherenceModel(model=lda_model, 
                                             texts=[doc.split() for doc in movie_reviews_df['Processed Review']], 
                                             dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print(f'Number of Topics: {num_topics}, Coherence Score: {coherence_lda}')
        
        if coherence_lda > best_coherence:
            best_coherence = coherence_lda
            best_model = lda_model
            best_num_topics = num_topics

    print(f'Best Number of Topics: {best_num_topics}, Best Coherence Score: {best_coherence}')

    # Print the best model's topics
    topics = best_model.print_topics(num_words=5)
    print("\nMost Common Topics in Movie Reviews:")
    for topic_num, topic in topics:
        print(f"\nTopic {topic_num + 1}:")
        topic_words = topic.split('+')
        for word in topic_words:
            word = word.split('*')[1].replace('"', '').strip()
            print(f"  - {word}")

    # Visualize the best model
    lda_display = gensimvis.prepare(best_model, corpus_gensim, id2word, sort_topics=False)
    pyLDAvis.display(lda_display)
    pyLDAvis.save_html(lda_display, 'lda_vis.html')
