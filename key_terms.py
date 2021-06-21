from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from lxml import etree
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest


def calc_tfidf(dataset, top_n):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset)
    terms_array = vectorizer.get_feature_names()
    all_relevant_words = []
    for document in tfidf_matrix.toarray():
        nmax = nlargest(top_n, enumerate(document), key=lambda x: (x[1], terms_array[x[0]]))
        relevant_words = []
        for n in nmax:
            pos = n[0]
            relevant_words.append(terms_array[pos])
        all_relevant_words.append(relevant_words)
    return all_relevant_words


def correct_postag(word):
    return pos_tag([word])[0][1] == desired_tag


def process_text(text, unallowed):
    unprocessed = word_tokenize(text)
    processed = []
    lemmatizer = WordNetLemmatizer()
    for word in unprocessed:
        word = lemmatizer.lemmatize(word)
        if word not in unallowed and correct_postag(word):
            processed.append(word)
    return processed


def collect_frequencies(path, limit):
    tree = etree.parse(path)
    root = tree.getroot()
    news = root[0]
    headlines = []
    only_content = []
    for new in news:
        title = new[0].text
        content = new[1].text.lower()
        processed_content = process_text(content, stopwords.words('english') + list(string.punctuation))
        headlines.append(title)
        only_content.append(" ".join(processed_content))
    return headlines, calc_tfidf(only_content, limit)


def output_text(titles, content_list):
    for i in range(len(content_list)):
        print(titles[i] + ":")
        print(" ".join(content_list[i]))
        print()


# main
article_path = 'news.xml'
count = 5
desired_tag = 'NN'
headlines, content_only = collect_frequencies(article_path, count)
output_text(headlines, content_only)
