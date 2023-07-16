import pickle
import uvicorn 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from fastapi import FastAPI
from pydantic import BaseModel

ps = PorterStemmer()

app = FastAPI(debug=True)

class InputData(BaseModel):
    message = str

def URL_transform(url):
    url = url.lower()
    url= nltk.word_tokenize(url)
    
    y = []
    for i in URL:
        if i.isalnum():
            y.append(i)
            
    URL = y[:]
    y.clear()

    for i in URL:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    URL = y[:]
    y.clear()
    
    for i in URL:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

@app.get('/') 
async def home():
    return {'Title':'Website Phising Solution'}


@app.post("/predict")
async def predict_url(input_data: InputData):
    
    # Preprocess
    transformed_URL = URL_transform(input_data.message)
    
    # Vectorize input
    vector_input = tfidf.transform([transformed_URL])
   
    # Predict
    result = model.predict(vector_input)[0]

    if result == 1:
        return {"prediction": "GOOD"}
    else:
        return {"prediction": "BAD"}

if __name__=="__main__":
   uvicorn.run(app) 