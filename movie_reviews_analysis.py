import pandas as pd
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

movie_reviews_path = 'IMDB Dataset.csv'
movie_reviews_df = pd.read_csv(movie_reviews_path)

#movie_reviews_df = movie_reviews_df.head(1000)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    sentences = sent_tokenize(text)
    processed_sentences = []
    
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
        processed_sentences.append(' '.join(filtered_words))

    return ' '.join(processed_sentences)

movie_reviews_df['Processed Review'] = movie_reviews_df['review'].apply(preprocess_text)

def classify_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    if sentiment_polarity > 0.1:
        return 'Positive'
    elif sentiment_polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

movie_reviews_df['Predicted Sentiment'] = movie_reviews_df['Processed Review'].apply(classify_sentiment)

positive_reviews = movie_reviews_df[movie_reviews_df['Predicted Sentiment'] == 'Positive']['Processed Review']
negative_reviews = movie_reviews_df[movie_reviews_df['Predicted Sentiment'] == 'Negative']['Processed Review']
neutral_reviews = movie_reviews_df[movie_reviews_df['Predicted Sentiment'] == 'Neutral']['Processed Review']

positive_reviews_text = ' '.join(positive_reviews.tolist())
negative_reviews_text = ' '.join(negative_reviews.tolist())
neutral_reviews_text = ' '.join(neutral_reviews.tolist())

wordcloud_positive = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(positive_reviews_text)
wordcloud_negative = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(negative_reviews_text)
wordcloud_neutral = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(neutral_reviews_text)

positive_count = len(positive_reviews)
negative_count = len(negative_reviews)
neutral_count = len(neutral_reviews)

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis("off")
plt.title(f'Word Cloud of Positive Sentiment Reviews (Total: {positive_count})')

plt.subplot(3, 1, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis("off")
plt.title(f'Word Cloud of Negative Sentiment Reviews (Total: {negative_count})')

plt.subplot(3, 1, 3)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis("off")
plt.title(f'Word Cloud of Neutral Sentiment Reviews (Total: {neutral_count})')

plt.tight_layout()
plt.show()

wordcloud_positive.to_file('wordcloud_positive.png')
wordcloud_negative.to_file('wordcloud_negative.png')
wordcloud_neutral.to_file('wordcloud_neutral.png')
