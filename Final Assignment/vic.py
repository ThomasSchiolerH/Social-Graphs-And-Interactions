import os
import json
import math
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# Aggregate text and build TF list for each media type
def aggregate_tf_for_media(media_name, media_files):
    media_text = ""
    for file_path in media_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                media_text += file.read() + " "
    
    # Preprocess aggregated text
    tokens = preprocess_text(media_text)
    tf_list = Counter(tokens)
    
    # Filter rare words
    tf_list_filtered = {word: count for word, count in tf_list.items() if count >= 5}
    return tf_list_filtered

# Save TF list for each media type
def save_tf_list(media_name, tf_list, output_folder):
    file_path = os.path.join(output_folder, f'{media_name}_tf_list.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        for word, count in tf_list.items():
            file.write(f"{word} {count}\n")

# File paths for Democratic and Republican media
democratic_files = ["Final Assignment/democratic_media_text.txt"]  # Replace with your files
republican_files = ["Final Assignment/republican_media_text.txt"]  # Replace with your files

output_folder = 'tf_lists'
os.makedirs(output_folder, exist_ok=True)

# Generate and save TF lists
democratic_tf_list = aggregate_tf_for_media("democratic", democratic_files)
save_tf_list("democratic", democratic_tf_list, output_folder)

republican_tf_list = aggregate_tf_for_media("republican", republican_files)
save_tf_list("republican", republican_tf_list, output_folder)

# Load TF lists and calculate TF-IDF
media_files = ['democratic', 'republican']
tf_by_media = {}

for media_name in media_files:
    filename = os.path.join(output_folder, f'{media_name}_tf_list.txt')
    tf_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word, count = line.strip().split()
            tf_dict[word] = 1 + math.log(int(count))  # Log scaling for TF
    tf_by_media[media_name] = tf_dict

# Calculate IDF across all media types
total_documents = len(tf_by_media)
df_counts = Counter()
for tf_dict in tf_by_media.values():
    unique_words = set(tf_dict.keys())
    for word in unique_words:
        df_counts[word] += 1

idf = {word: math.log(total_documents / df_count) for word, df_count in df_counts.items()}

# Calculate TF-IDF
tfidf_by_media = {}
for media_name, tf_values in tf_by_media.items():
    tfidf_by_media[media_name] = {word: tf * idf[word] for word, tf in tf_values.items()}

# Get top 10 TF-IDF words for each media type
top_words_by_media = {}
for media_name, tfidf in tfidf_by_media.items():
    sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    top_words = sorted_tfidf[:10]
    top_words_by_media[media_name] = top_words

# Display results in a table
table_data = []
for media_name, words in top_words_by_media.items():
    words = [f"{word} ({score:.2f})" for word, score in words]
    table_data.append(words)
df = pd.DataFrame(table_data, index=[media.title() for media in top_words_by_media],
                  columns=[f"Word #{i+1}" for i in range(10)])
print(df)

# Create word clouds
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
media_names = list(tfidf_by_media.keys())

for idx, media_name in enumerate(media_names):
    ax = axes[idx]
    top_tfidf_scores = dict(sorted(tfidf_by_media[media_name].items(), key=lambda x: x[1], reverse=True)[:100])
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False)
    wordcloud.generate_from_frequencies(top_tfidf_scores)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(media_name.title(), fontsize=16, pad=10)

plt.suptitle('Word Clouds for the Top 10 Words by TF-IDF', fontsize=20, y=1.02)
plt.tight_layout()
plt.show()
