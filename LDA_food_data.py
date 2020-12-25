#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -r requirements.txt')


# In[1]:


import gensim
import numpy as np
import nltk
import pandas as pd
import pyLDAvis.gensim
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from gensim.models import CoherenceModel
nltk.download('wordnet')
np.random.seed(2018)


# In[2]:


data = pd.read_parquet("./products.parquet.gz")


# ### Data Preprocessing
# 
# * Concatenate product name and description to include all information in one sentence
# * Use gensim utils simple preprocessing to convert text to lowercase
# * Remove stopwords from the text 
# * Lemmatizing and stemming the final tokens

# In[3]:


data['product_description'] = data['product_description'].fillna(value = "")


# In[4]:


data["product_concat"] = data['product_name'] + data['product_description']


# In[5]:


data


# In[6]:


data.info()


# In[7]:


def lemmatize_stemming(text):
    porter = PorterStemmer()
    return porter.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# In[8]:


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[9]:


processed_docs = data['product_concat'].map(preprocess)


# In[10]:


processed_docs


# In[11]:


pickle.dump(processed_docs, open( "processed_docs.p", "wb" ))


# In[12]:


processed_docs = pickle.load(open("processed_docs.p", "rb"))


# In[13]:


from wordcloud import WordCloud
all_word = []
for word in processed_docs:
    all_word.extend(word)
all_words = ''.join(all_word)


# In[14]:


word_dict = dict(Counter(all_word))


# In[15]:


wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, background_color="white").generate_from_frequencies(word_dict)


# In[16]:


plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Some frequent words used in the descriptions", weight='bold', fontsize=14)
plt.show()


# ### Modelisation
# * Prepare dictionary of all the words in the text
# * Filter out words from the dictionary that appear less than 15 times in the text and more than 50% of all texts
# * Convert dictionary to bag of words
# * Get tf-idf representation from bag of words
# * Train LDA(Latent Dirichlet Allocation) model with 3 topics

# In[17]:


dictionary = gensim.corpora.Dictionary(processed_docs)


# In[18]:


len(dictionary)


# In[19]:


dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=20000)


# In[20]:


bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# In[25]:


tfidf = gensim.models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]


# In[26]:


lda_model_3 = gensim.models.LdaMulticore(corpus_tfidf, num_topics=3, id2word=dictionary, passes=3, workers=2)
for idx, topic in lda_model_3.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[27]:


lda_display = pyLDAvis.gensim.prepare(lda_model_3, corpus_tfidf, dictionary, sort_topics=False)


# In[28]:


pyLDAvis.display(lda_display)


# ### Model Inference on the original data

# In[29]:


topics = []
scores = []


# In[30]:


for scores_row in lda_model_3[corpus_tfidf]:
    sorted_row = sorted(scores_row, key=lambda tup: -1*tup[1])
    topics.append(sorted_row[0][0])
    scores.append(sorted_row[0][1])


# In[31]:


processed_df = data.copy()
processed_df['topics'] = topics
processed_df['scores'] = scores


# In[32]:


processed_df['topics'].value_counts()


# In[33]:


processed_df


# ### Checking top products in each topic ordered by score

# In[34]:


topic0 = processed_df.loc[processed_df['topics'] == 0].sort_values('scores', ascending = False)


# In[35]:


topic1 = processed_df.loc[processed_df['topics'] == 1].sort_values('scores', ascending = False)


# In[36]:


topic2 = processed_df.loc[processed_df['topics'] == 2].sort_values('scores', ascending = False)


# In[37]:


topic0[:500]['product_name'].value_counts()


# In[38]:


topic1[:5000]['product_name'].value_counts()


# In[39]:


topic2[:10000]['product_name'].value_counts()


# ### Inference
# * Topic 0 mostly contains medicines and daily groceries
# * Topic 1 mostly contains main course food items including local Singaporean foods
# * Topic 2 mostly contains Macdonald's items and drinks

# ### Vendor Recommendations
# 
# - Group topics by Geohash and calculate % of each topic in the geohash. 
# - For a given vendor, we can then make recommendations based on the popular topics in his geohash

# In[40]:


grouped_processed = processed_df.groupby(['vendor_geohash', 'topics']).agg({'product_name': 'count'})
grouped_pcts = grouped_processed.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))


# In[41]:


grouped_pcts = grouped_pcts.reset_index()


# In[42]:


grouped_pcts


# In[43]:


pivoted = grouped_pcts.pivot(index='vendor_geohash', columns='topics', values='product_name')


# In[44]:


pivoted = pivoted.fillna(0.0)


# In[45]:


pivoted.columns = ['T0: Daily Use Items' , 'T1: Main Course', 'T2: McDonalds and drinks']


# In[46]:


pivoted


# In[47]:


processed_df = processed_df.merge(pivoted, on = 'vendor_geohash')


# In[48]:


processed_df.sort_values('product_id')


# In[49]:


coherence_model_lda = CoherenceModel(model=lda_model_3, texts=processed_docs, dictionary=dictionary, coherence='c_v')


# In[50]:


coherence_lda = coherence_model_lda.get_coherence()


# In[51]:


coherence_lda


# ### Selecting the appropriate number of topics

# In[52]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=3, workers=2)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[53]:


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus_tfidf, texts=processed_docs, start=2, limit=25, step=5)


# In[54]:


limit=25; start=2; step=5;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

