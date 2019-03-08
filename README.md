### Requirements

- editdistance
- flaskr
- gensim
- keras
- nltk
- pyenchant
- scikit-learn
- spacy (You will need to get the English model: "python -m spacy download en")
- tensorflow-gpu (NOT tensorflow. You need the GPU. [Install 1.12.0 for CUDA 9.0, and 1.13.0 for CUDA 10.0](https://www.tensorflow.org/install/source#tested_build_configurations))

### Setup

In the [resources/embeddings](resources/embeddings) folder, you need to extract the word vectors from here:
	https://github.com/mmihaltz/word2vec-GoogleNews-vectors
	
	Final path to this file: resources/embeddings/GoogleNews-vectors-negative300.bin

Next, run [preprocess.py](flaskr/preprocess.py) and [train.py](flaskr/train.py) to create the model used to run neuron-level visualizations.
Then "flask run" with the values below

    FLASK_APP=flaskr
    FLASK_ENV=development

