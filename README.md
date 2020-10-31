## Notes on submission

* :point_right: Colab for re-processing and feature engineering: https://colab.research.google.com/drive/1gaybyJtzMh90jdhgXJHxlz46UwSzXOyR?usp=sharing
* :point_right: Colab for modeling: https://colab.research.google.com/drive/1SfoEdKiFA4oha2q_nAvneuxFjW0FIQ29?usp=sharing

### 1. Pre-processing

* Remove all punctuations.
* Remove all numbers.
* Remove repeated characters (such as "Ngon quáaaaa" => "Ngon quá").
* Remove word with single character (IMO, single character words in Vietnamese seem useless).
* Remove some consecutive spaces and some redundant characters.
* Replace emojis and some slang with correct word (e.g, "rùi" => "rồi").
* Strip HTML (this doesn't seem to have so much effect, only a few comments contain HTML).
* Tried but not work:
    * Remove stop-words: the stop words from [here][stopwords] seems to contain a lot of useful information. The accuracy drops dramatically when remove all the stop-words.
    * Remove accents: removing accents alone will drop the accuracy, however when using the accents removed dataset together with the original one, the accuracy is higher.
    * Remove double negative: turn words like "không" in positive sentences into some positive words => this gives the model an easier dataset to train on but when the model meets "hard" dataset that contains double negative, it performs poorly.
    * Turn "chẳng đẹp/ngon" or "không tệ" to "not-positive" and "not-negative".
    * Replace negative/positive words with "negative" or "positive".

### 2. Models and Feature Engineering

Split train set to 2 sub-sets (70:30), one for fitting and one for validation.
* Word2Vec: using pre-trained 400 dimensional embedding [here][word2vec]:
    * Word2Vec + PCA + XGBoost: 0.73 (validation), ensemblers and boosters seem to not work well.
    * Word2Vec + LinearSVM: 0.842 (validation)
    * With accent TF-IDF + without accent TF-IDF + Word2Vec: 0.892 (validation)
    * Word2Vec + GRU: 0.897 (validation) => overfit after 9-10 epochs
    * Word2Vec + Conv1D + LSTM: 0.8881 (validation) => overfit after 9-10 epochs
    
* TF-IDF: bag-of-words with `ngram_range` from 1 to 6:
    * TF-IDF + LinearSVM(C=100.0): 0.90148 (validation), simple but efficient? (1)
    * TF-IDF + LinearSVM(C=100.0) + Remove accents on train set: 0.685 (validation)
    * With accent TF-IDF + without accent TF-IDF: 0.901 (validation), almost as good as (1)
    
* Create extra features:
    * Emoji feature: define a list of negative and positive emoji. Count number of negative and positive emoji in each sentence.
    Also the ratio between negative and positive emoji.
    * Some static features: number of words, number of unique words.
    * Also combine with TF-IDF.
    * **Result**: 
        1. LinearSVC(C=10.0): 0.8518.
        2. TruncatedSVD to reduce dimension down to 500 features + LinearSVC(C=100.0): 0.9009
        
* Models with TF-IDF:
    * LogisticRegression(C=10.0): 0.8962
    * SGDClassifier(loss='log'): 0.877
    * XGBoost(n_estimators=100): 0.8327
    * Hard VotingClassifier (with Logistic, SGD, and RandomForest): 0.896
    * Neural Networks with Embedding: I tried GRU and LSTM, both overfit as 9th-10th epoch, val_accuracy stopped at ~0.89.
 
### 3. Conclusion and problems 
* Best pipeline is **TF-IDF + Linear** (based on the accuracy score on validation set). I then fine-tuned the pipeline using Grid Search cross-validation. 
* However, the model suffers from "double negative" problem. 
Most positive samples with "nhưng", "mỗi tội", "nên không", etc aren't correctly classified.
* Samples with broken font and accent (e.g, "nho ̉ nhă ́ va ̀ co ́ ve ̉ kha ́ châ ̣ nv ba ́ ve ́ thi ̀ niê ̀ mình") is usually misclassified.
* Words with word "không" included (e.g "không gian", "không biết") would easily be given negative weights although the sentence is positive overall => may cause problems.

### 4. Feature works
* Neural Networks are promising (although the fit time for NN with LSTM layers are too large), solving the overfitting problem may open some doors. 
Moreover, as I researched, method like "Weight Ensemble" would work for Vietnamese sentiment analysis. Also should give Bidirectional LSTM a try.
* When I manually check the dataset, there are a lot of mis-labeled example. Re-label dataset + fix samples with "broken accent" would be a good thing to do.
* Some comments are in English: a future solution maybe translate some Vietnamese samples to English, fit the model on them (or using English pre-train embedding?).


[stopwords]: https://github.com/stopwords/vietnamese-stopwords/blob/master/vietnamese-stopwords.txt
[word2vec]: https://github.com/sonvx/word2vecVN