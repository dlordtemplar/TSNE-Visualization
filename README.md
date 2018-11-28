Requirements.txt:

- flaskr
- keras
- coverage
- gensim
- matplotlib
- nltk
- pyenchant
- pytest
- scikit-learn
- seaborn
- spacy (You will need to get the English model: "python -m spacy download en")
- tensorflow-gpu (NOT tensorflow. You need the GPU)


In the resources/embeddings folder, you need to extract the word vectors from here:
	https://github.com/mmihaltz/word2vec-GoogleNews-vectors
	
	Final path to this file: resources/embeddings/GoogleNews-vectors-negative300.bin

Current workflow:
	Run preprocess.py, train.py, and then main.py.
