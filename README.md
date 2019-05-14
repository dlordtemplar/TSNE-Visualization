### Requirements

Client (This web interface)
- wtforms
- requests
- Flask

[Keras model](https://github.com/dlordtemplar/keras-base-rest) (Neuron, Pair)
- flask
- hunspell (Requires python-dev and libhunspell-dev. Install with 'sudo apt-get install libhunspell-dev')
- numpy
- editdistance
- gensim
- keras
- nltk
- pandas
- scikit-learn
- spacy (You will need to get the English model: "python -m spacy download en")
- tensorflow-gpu (NOT tensorflow. You need the GPU. [Install 1.12.0 for CUDA 9.0, and 1.13.0 for CUDA 10.0](https://www.tensorflow.org/install/source#tested_build_configurations))

[Torch model](https://github.com/dlordtemplar/torch-insuranceQA-rest) (Passage)
- flask
- spacy
- numpy
- torch
- pandas
- nltk
- gensim
- pyenchant (Can be replaced by pyhunspell)
- scikit_learn
- spacy (You will need to get the English model: "python -m spacy download en")



### Setup

This project is the client that fetches information from the two models that serve as REST APIs. Set up the two models:

[Keras model](https://github.com/dlordtemplar/keras-base-rest)

[Torch model](https://github.com/dlordtemplar/torch-insuranceQA-rest)


Then, "flask run --port 5002" with the values below

    FLASK_APP=flaskr
    FLASK_ENV=development

