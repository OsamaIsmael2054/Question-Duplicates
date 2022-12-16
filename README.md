# Question Duplicates

I used the Quora question answer dataset to build a model that could identify similar questions. This is a useful task because you don't want to have several versions of the same question posted.

## Installation

I used docker to containerize a Flask application so user can enter their questions and check if they are similar or not 

```
docker-compose up -d --build
```

## Model

```
We use Siamese networks applied to natural language processing.
The Model consists of LSTM layer followed by linear layer to represent each question as vector
then we calculate similarities between vectors.
```

## Dataset

the dataset used is Quora questions dataset, that consists of two questions and if they are pair or not.
the dataset can be downloaded via :
https://www.kaggle.com/competitions/quora-question-pairs/data

