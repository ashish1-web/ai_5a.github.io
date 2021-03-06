#!/usr/bin/env python
# coding: utf-8

# In[44]:


import nltk                                 # importing nltk library


# In[45]:


from nltk.tokenize import sent_tokenize     # we need to break the word so that the machine can read it / for cleaning the sentence


# In[46]:


text="""A Recurrent neural network is a class of neural network where connections between the nodes form a direct graph along a temporal sequence. This allows it to exbit temporal dynamic behaviour. Derived from feedforward neural network, Rnn can use their internal state to process variable length sequence of inputs."""


# In[47]:


tokenized_text=sent_tokenize(text)      # tokenizing the input text


# In[48]:


nltk.download('wordnet')            # wordnet is a lexical database for the english/ wordnet is a english dishnorey which is a part of ntlk
nltk.download('punkt')              # A tokenizer which divides a text into list of sentences which the help of an unsurpurives alogrihtm


# In[49]:


tokenized_text=sent_tokenize(text)    


# In[50]:


print(tokenized_text)           # printing the tokenized text


# In[51]:


from nltk.tokenize import word_tokenize         # to import the tokens 
tokenized_word=word_tokenize(text)              # to remove punctuations and numerical characters
print(tokenized_word)                           #printing the results


# In[52]:


# Counting word frequency
from nltk.probability import FreqDist          #ntlk library for counting the word frequency
fdist = FreqDist(tokenized_word)               # counting the frequency
print(fdist)


# In[53]:


fdist.most_common(5)                          # extracting the five most common words


# In[54]:


import matplotlib.pyplot as plt                 #for plotting the graphs


# In[55]:


fdist.plot(44,cumulative=False)                #
plt.show()


# In[56]:


from nltk.corpus import stopwords               # to remove sequences like i'm and so on
stop_words=set(stopwords.words("english"))      
print(stop_words)                               #printing the common stopwords


# In[57]:


nltk.download('stopwords')


# In[58]:


from nltk.corpus import stopwords


# In[59]:


stop_words=set(stopwords.words("english"))


# In[60]:


print(stop_words)


# In[61]:


filtered_sent=[]                #intiaizing  a empty list


# In[62]:


tokenized_word=word_tokenize(text)     


# In[63]:


for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)


# In[64]:


print("Tokenized Sentence:",tokenized_word)


# In[65]:


print("Filterd Sentence:",filtered_sent)


# In[66]:


from nltk.stem import PorterStemmer     # importing the stemmer


# In[67]:


from nltk.tokenize import sent_tokenize, word_tokenize


# In[68]:


ps = PorterStemmer()


# In[69]:


stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))


# In[70]:


print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)


# In[71]:


#Lexicon Normalization
#performing stemming and Lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()


# In[72]:


Lemmatize_words=[]
for w in filtered_sent:
    Lemmatize_words.append(lem.lemmatize(w))
   # print("Lemmatized Word:",lem.lemmatize(word,"v"))


# In[73]:


print("Filtered Sentence:",filtered_sent)
print("Lemmatized Word:",Lemmatize_words)


# In[74]:


print("Stemmed Sentence:",stemmed_words)
print("Lemmatized Word:",Lemmatize_words)


# In[ ]:





# In[ ]:




