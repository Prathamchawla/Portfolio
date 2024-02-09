#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from scipy.stats import zscore
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# In[2]:


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


# In[3]:


# Statistical Anomaly Detection 
def Z_Score_Value(Value):
    mean = 158668.18606839626
    std = 264941.5785563747
    Z_score = (Value - mean)/std
    Threshold_value = 12.40
    
    if Z_score >= Threshold_value:
        return 1
    else:
        return 0
    
def Tukey_Fences_Values(Value):
    Q1 = 12149.490000000002
    Q3 = 213762.15000000002
    IQR = Q3 - Q1
    tukey_threshold = 16.00 * IQR
    
    if Value < Q1 - tukey_threshold or  Value > Q3 + tukey_threshold:
        return 1
    else:
        return 0
    
    
def modified_zscore_values(value):
    median = 76345.78
    median_absolute_deviation = np.median(np.abs(value - median))
    modified_z_score = np.abs(0.6745 * (value - median) / median_absolute_deviation)
    threshold = 32.30
    
    if modified_z_score > threshold:
        return 1
    else:
        return 0
    
@app.route('/statisticalanomalydetection')
def index1():
    return render_template('statisticalanomaly_index.html')

@app.route('/predict_anomaly', methods=['POST'])
def predict_anomaly():
    try:
        selected_function = request.json['selectedFunction']
        payment_type = request.json['paymentType']
        old_balance = request.json['oldBalance']
        amount = request.json['amount']

        # Choose the selected function
        if selected_function == 'modified_zscore':
            prediction = modified_zscore_values(amount)
        elif selected_function == 'tukey_fences':
            prediction = Tukey_Fences_Values(old_balance)
        elif selected_function == 'z_score':
            prediction = Z_Score_Value(amount)
        else:
            raise ValueError('Invalid function selected')

        # Simulate updating the balance (replace with your actual logic)
        current_balance = old_balance - amount if payment_type == 'debit' else old_balance + amount
        
         

        return jsonify({'prediction': prediction, 'currentBalance': current_balance})

    except Exception as e:
        return jsonify({'error': str(e)})


# In[4]:


# machine learning fraud detection 

with open('model/frauddetection_xgb_model.pkl', 'rb') as model_file:
    fraud_model = pickle.load(model_file)
    
@app.route('/machinelearningfrauddetection')
def machinelearning():
    return render_template('machinelearningfrauddetection.html')

@app.route('/machinelearningpredict', methods=['POST'])
def machinelearningpredict():
    type_of_payment = request.form['type_of_payment']
    old_balance = float(request.form['old_balance'])
    amount = float(request.form['amount'])

    # Map type_of_payment to encoded values
    payment_mapping = {'cash_out': 1, 'payment': 3, 'cash_in': 0, 'transfer': 4, 'debit': 2}
    encoded_payment = payment_mapping.get(type_of_payment.lower(), 0)

    # Create new_balance variable
    new_balance = old_balance + amount

    # Make prediction
    prediction = fraud_model.predict([[encoded_payment, amount, old_balance, new_balance]])

    # Display the result on the HTML page
    if prediction[0] == 0:
        result = 'Not Fraud'
    else:
        result = 'Fraud Transaction'
    
    return render_template('machinelearningfrauddetection.html', result=result)
    


# In[5]:


# Diabetic Prediction  - Logistic Regression

# Load scaler and model
with open('model/diabeticscaler.pkl', 'rb') as scaler_file:
    diabitic_scaler = pickle.load(scaler_file)

with open('model/diabeticmodel.pkl', 'rb') as model_file:
    diabitic_model = pickle.load(model_file)

@app.route('/diabiticmodel')
def home():
    return render_template('diabeticmodel.html')

@app.route('/diabiticmodelpredict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        scaled_features = diabitic_scaler.transform([features])
        prediction = diabitic_model.predict(scaled_features)

        result = None
        if prediction[0] == 1:
            result = 'The patient is diabetic.'
        else:
            result = 'The patient is not diabetic.'

        return render_template('diabeticmodel.html', prediction=result)



# In[6]:


# Algerian Forest Fire Prediction - Linear Regression 

Algerian_ridge_model = pickle.load(open('model/Algerianmodel.pkl','rb'))
Algerian_standard_scaler = pickle.load(open("model/AlgerianScaler.pkl",'rb'))

@app.route('/Algerianpredictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float (request. form. get ('Temperature' ))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain' ))
        FFMC = float(request.form.get('FFMC'))
        DMC = float (request.form.get('DMC' ))
        ISI = float(request.form.get('ISI'))
        region = float (request.form.get('Region'))
        Classes = float (request.form.get('Classes'))
        new_data_scaled = Algerian_standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,region,Classes]])
        result = Algerian_ridge_model.predict(new_data_scaled)
       
        return render_template('Algerian.html',result=result[0])
        
    else:
        return render_template('Algerian.html')


# In[7]:


@app.route('/EDAredwine')
def redwine():
    return render_template('RedWine.html')


# In[8]:


@app.route('/TeslaStock')
def Tesla():
    return render_template('Tesla.html')


# In[9]:


# Adventure Works Power BI
@app.route('/Adventure')
def AW():
    return render_template('AdventureWorks.html')


# In[10]:


@app.route('/SpiceJet')
def SJ():
    return render_template('SpiceJet.html')


# In[11]:


@app.route('/FraudDataset')
def FPB():
    return render_template('FraudPowerBI.html')


# In[12]:


@app.route('/RoadAccident')
def RA():
    return render_template('RoadAccidentExcel.html')


# In[13]:


data=pd.read_csv('ReccomendationPortfolio.csv')
data['artist'] = data['artist'].fillna('')
data['tags'] = data['tags'].fillna('')
data['genre'] = data['genre'].fillna('')

data['text']=data['artist']+ ' ' + data['tags']+ ' ' + data['genre']
data['text']

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index=data['name']).drop_duplicates()

def get_recommendations(name,cosine_sim=cosine_sim):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar songs
    song_indices = [i[0] for i in sim_scores]
    recommended_songs = data['name'].iloc[song_indices]

    print("Recommended Songs:", recommended_songs)
    return recommended_songs.tolist()

@app.route('/ReccomendationSystem')
def get_recommendations_route():
    return render_template('Recommendation.html',recommended_songs=[])
    
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations_routee():
    song_name = request.form['song_name']
    recommended_songs = get_recommendations(song_name)
    return render_template('Recommendation.html', recommended_songs=recommended_songs)


# In[14]:


# sentiment analysis --
with open('model/Sentimentvectorizer.pkl', 'rb') as f:
    Sentimentvectorizer = pickle.load(f)

# Load the logistic regression classifier
with open('model/Sentimentclassifier.pkl', 'rb') as f:
    Sentimentclassifier = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

@app.route('/SentimentAnalysis')
def Sentiment():
    return render_template('Sentiment_Analysis.html')

@app.route('/Sentimentpredict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    text = data['text']

    # Preprocess text
    preprocessed_text = preprocess_text(text)

    # Vectorize the text
    text_vectorized = Sentimentvectorizer.transform([preprocessed_text])

    # Make prediction
    prediction = Sentimentclassifier.predict(text_vectorized)[0]
    sentiment = prediction

    return jsonify({'sentiment': sentiment})


# In[15]:


# LinkedIn Data Analysis
@app.route('/LinkedInDataAnalysis')
def LDA():
    return render_template("LinkedInDataAnalysis.html")


# In[16]:


#PM2.5 Predcition 

with open('model/PM25Scaler.pkl', 'rb') as f:
    PM25scaler = pickle.load(f)

with open('model/PM25Random.pkl', 'rb') as f:
    PM25Random = pickle.load(f)

@app.route('/Pm25Prediction')
def PM25():
    return render_template('PMtemp.html')

# Define the route for prediction
@app.route('/PMpredict', methods=['POST'])
def predictPM():
    # Get the input values from the form
    features = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws']
    user_input = [request.form[feat] for feat in features]

    # Convert input to numpy array and reshape
    user_input = np.array(user_input).reshape(1, -1)

    # Scale the input
    scaled_input = PM25scaler.transform(user_input)

    # Make prediction
    prediction = PM25Random.predict(scaled_input)

    return render_template('PMtemp.html', prediction=prediction[0])


# In[ ]:


if __name__ == '__main__':
    app.run(host='0.0.0.0')

