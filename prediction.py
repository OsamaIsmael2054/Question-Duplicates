from model import Network
import torch
import pickle
import string
from TripletLoss import TripletLoss


with open("Projects\\Question Dublicates\\utils\\vocab_stoi.pkl","rb") as handle:
    Str2Index = pickle.load(handle)

with open("Projects\\Question Dublicates\\utils\\vocab_itos.pkl","rb") as handle:
    IndextoStr = pickle.load(handle)

max_len = 400
vocab_size = len(Str2Index) + 1
loss_fn = TripletLoss()

def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    translator = str.maketrans('', '', string.punctuation)
    return text.lower().translate(translator)

def Question_handler(question:str):
    question_tokenized = []
    for word in question.split():
        if word in Str2Index:
            question_tokenized.append(Str2Index[word])
        else:
            question_tokenized.append(Str2Index["<unk>"])

    if len(question_tokenized) < max_len:
        question_tokenized +=  [Str2Index["<pad>"]] * (max_len - len(question_tokenized))

    elif len(question_tokenized) > max_len:
        question_tokenized[:max_len]

    return question_tokenized


def text_pipeline(question:str):
    clean = remove_punctuation(question)
    question_tokenized = Question_handler(clean)
    question_tokenized = torch.tensor(question_tokenized, dtype=torch.int64)
    return question_tokenized

def predict(Q1:str, Q2:str, thershold=0.7, PATH_TO_MODEL = './utilsWeights.pth'):
    Q1_tokenized = text_pipeline(Q1)
    Q2_tokenized = text_pipeline(Q2)

    Q1_tokenized = Q1_tokenized.unsqueeze(0)
    Q2_tokenized = Q2_tokenized.unsqueeze(0)

    model = Network(vocab_size)
    model.eval()
    model.load_state_dict(torch.load(PATH_TO_MODEL, map_location="cpu"))

    v1 = model(Q1_tokenized)
    v2 = model(Q2_tokenized)

    score = torch.matmul(v1,v2.T)
    prediction = score >= thershold

    return prediction.item()

