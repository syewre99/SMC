import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

movie_reviews_path = 'IMDB Dataset.csv'
movie_reviews_df = pd.read_csv(movie_reviews_path)

#movie_reviews_df = movie_reviews_df.head(1000)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

emotion_lexicon = {
    'love': ['love', 'affection', 'fondness'],
    'joy': ['joy', 'happy', 'delight', 'pleasure', 'glad'],
    'surprise': ['surprise', 'astonish', 'shock'],
    'anger': ['anger', 'angry', 'rage', 'annoyance'],
    'fear': ['fear', 'worry', 'terror', 'scare', 'anxiety'],
    'sadness': ['sad', 'grief', 'sorrow', 'unhappy']
}

def preprocess_text(text):
    sentences = sent_tokenize(text)
    processed_sentences = []
    
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
        processed_sentences.append(' '.join(filtered_words))

    return ' '.join(processed_sentences)

movie_reviews_df['Processed Review'] = movie_reviews_df['review'].apply(preprocess_text)

def classify_emotion(text):
    words = text.split()
    emotion_count = defaultdict(int)
    
    for word in words:
        for emotion, keywords in emotion_lexicon.items():
            if word.lower() in keywords:
                emotion_count[emotion] += 1
                
    return emotion_count

movie_reviews_df['Emotion Count'] = movie_reviews_df['Processed Review'].apply(classify_emotion)

total_emotions = defaultdict(int)
for emotion_dict in movie_reviews_df['Emotion Count']:
    for emotion, count in emotion_dict.items():
        total_emotions[emotion] += count

for emotion in emotion_lexicon.keys():
    if emotion not in total_emotions:
        total_emotions[emotion] = 0

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    total_emotions.values(), 
    labels=total_emotions.keys(), 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=['#FF6347', '#FFD700', '#87CEEB', '#FF4500', '#4169E1', '#708090']
)

for text in texts + autotexts:
    text.set_fontweight('bold')

plt.title('Emotion Proportions in Movie Reviews', fontweight='bold')
plt.axis('equal')
plt.show()
plt.savefig('emotion_proportions_piechart.png', format='png', dpi=300)