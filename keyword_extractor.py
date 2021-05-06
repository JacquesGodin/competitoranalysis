import pandas
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import os
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import re
import seaborn as sns
import argparse


nltk.download('stopwords')
nltk.download('wordnet')


def analysis(file):
    #Get the name of the Competitor under analysis
    ent_name = file.split('\\')[-1].split('.')[0]
    
    #Get the path to the Competitor's folder
    ent_path = file.split(file.split('\\')[-1])[0]

    print(f"\n\n###########################\nAnalyzing {ent_name}\n###########################")

    #Load the dataset
    dataset = pandas.read_csv(file, delimiter = '\t', header=None)
    dataset.head()

    #Get the index of the max row of the dataset
    maxrow = len(dataset.index)
    print(f"\n\nMax row is: {maxrow}\n\n\n")
    
    ##Creating a list of stop words and adding custom stopwords
    stop_words = set(stopwords.words("english"))

    ##Creating a list of custom stopwords
    stop_words.remove("your")
    stop_words.remove("to")
    new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown", "cookies", "website", "get", "book", "demo", "see", "case", "study"]
    stop_words = stop_words.union(new_words)

    corpus = []
    for i in range(0, maxrow):
        #Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', dataset[0][i])
        
        #Convert to lowercase
        text = text.lower()
        
        #remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
        
        # remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ",text)
        
        ##Convert to list from string
        text = text.split()
        
        ##Stemming
        ps=PorterStemmer()
        
        #Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in  
                stop_words] 
        text = " ".join(text)
        corpus.append(text)
        
    #Word cloud
    # matplotlib inline
    
    print("\n\n###########################\nCreating WordCloud\n###########################\n\n")
    
    wordcloud = WordCloud(background_color='white', stopwords=stop_words, max_words=100, max_font_size=50,random_state=42).generate(str(corpus))

    print(wordcloud)
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    fig.savefig(os.path.join(ent_path,f"{ent_name} wordcloud.png"), dpi=1200)

    #Keywords Analysis
    cv = CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
    X = cv.fit_transform(corpus)

    #Most frequently occuring words
    def get_top_n_words(corpus, n=None):
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in      
                       vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], 
                           reverse=True)
        return words_freq[:n]
    #Convert most freq words to dataframe for plotting bar plot
    top_words = get_top_n_words(corpus, n=10)
    top_df = pandas.DataFrame(top_words)
    top_df.columns=["Word", "Freq"]
    #Barplot of most freq words

    print("\n\n###########################\nCreating MonoWord chart\n###########################\n\n")


    sns.set(rc={'figure.figsize':(13,8)})
    g = sns.barplot(x="Word", y="Freq", data=top_df)
    g.set_xticklabels(g.get_xticklabels(), rotation=30)
    g.figure.savefig(os.path.join(ent_path,f"output_mono_{ent_name}.png"))

    #Most frequently occuring Bi-grams
    def get_top_n2_words(corpus, n=None):
        vec1 = CountVectorizer(ngram_range=(2,2),  
                max_features=2000).fit(corpus)
        bag_of_words = vec1.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in     
                      vec1.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], 
                    reverse=True)
        return words_freq[:n]
    top2_words = get_top_n2_words(corpus, n=10)
    top2_df = pandas.DataFrame(top2_words)
    top2_df.columns=["Bi-gram", "Freq"]
    print(top2_df)
    
    print("\n\n###########################\nCreating Bi-gram chart\n###########################\n\n")
    
    sns.set(rc={'figure.figsize':(13,8)})
    h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
    h.set_xticklabels(h.get_xticklabels(), rotation=45)
    h.figure.set_size_inches(15.7, 14.27)
    h.figure.savefig(os.path.join(ent_path,f"output_bi_{ent_name}.png"))

    #Most frequently occuring Tri-grams
    def get_top_n3_words(corpus, n=None):
        vec1 = CountVectorizer(ngram_range=(3,3), 
               max_features=2000).fit(corpus)
        bag_of_words = vec1.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in     
                      vec1.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], 
                    reverse=True)
        return words_freq[:n]
        
    top3_words = get_top_n3_words(corpus, n=10)
    top3_df = pandas.DataFrame(top3_words)
    top3_df.columns=["Tri-gram", "Freq"]
    print(top3_df)

    print("\n\n###########################\nCreating Tri-gram chart\n###########################\n\n")

    sns.set(rc={'figure.figsize':(13,10)})
    j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
    j.set_xticklabels(j.get_xticklabels(), rotation=45)
    j.figure.set_size_inches(19.7, 17.27)
    j.figure.savefig(os.path.join(ent_path,f"output_tri_{ent_name}.png"))

    print("\n\n###########################\nAnalysis concluded. Check the respective folder\n###########################")

if __name__ == "__main__":
    #Parser for the output html file argument
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="output txt to be analyzed",type=str, nargs='?', default="scrape_output_docusign.txt")

    args = parser.parse_args()

    # transform the parsed argument in the variable for the scraper
    file = args.file

    analysis(file)


