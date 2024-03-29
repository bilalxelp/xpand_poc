import streamlit as st
import pandas as pd
import base64
from tqdm.auto import tqdm
import requests
import ast
import base64
from transformers import pipeline
from tqdm.auto import tqdm
import re
import langdetect
import stanza
import requests
import ast


def remove_non_english(df, column_name):
    # Create a new column with language codes for each row in the specified column
    df['lang'] = ''
    for x in range(len(df)):
      try:
        df['lang'][x] = langdetect.detect(df[column_name][x])
      except langdetect.lang_detect_exception.LangDetectException:
        df['lang'][x] = None
    
    # Filter the DataFrame to only include rows with English text in the specified column
    df = df[df['lang'] == 'en']
    
    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)
    
    # Drop the 'lang' column
    df = df.drop(columns='lang')
    
    return df

def stanza_tokenizer(text):
    NLP = stanza.Pipeline(lang='en', processors='tokenize')
    
    doc = NLP(str(text))
    
    sent_list = []
    for i, sentence in enumerate(doc.sentences):
        sent_list.append(sentence.text)
        
    clean_sent = []
    for each_sen in sent_list:
        each_sent = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»""'']))''', '', each_sen)
        each_sent = re.sub(r'^{0}'.format(re.escape('. ')), '', each_sent)
        each_sent = re.sub("\n+", "\n",each_sent)
        clean_sent.append(max([each_ele for each_ele in each_sent.split("\n")], key = len))
        
    return clean_sent

def Viewpoint_classifier(df, column_name):
  
  classifier = pipeline("text-classification", model="lighteternal/fact-or-opinion-xlmr-el", tokenizer="lighteternal/fact-or-opinion-xlmr-el")

  df['Viewpoint'] = ''

  for x in range(len(df)):
    result = classifier(str(df[column_name][x]))
    if result[0]['label'] == 'LABEL_0':
      df['Viewpoint'][x] = "Opinion"
    elif result[0]['label'] == 'LABEL_1':
      df['Viewpoint'][x] = "Fact"

  return df

def stance_feminist(df, column_name):

  classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-stance-feminist", tokenizer="cardiffnlp/twitter-roberta-base-stance-feminist")

  df['Label'] = ''
  df['Score'] = ''

  for x in range(len(df)):
    result = classifier(df[column_name][x])
    df['Label'][x] = result[0]['label']
    df['Score'][x] = result[0]['score']

  return df

def Toxic_Detection(df, column_name):

  classifier = pipeline("text-classification", model="martin-ha/toxic-comment-model", tokenizer="martin-ha/toxic-comment-model")

  df['Label'] = ''
  df['Score'] = ''

  for x in range(len(df)):
    result = classifier(df[column_name][x])
    df['Label'][x] = result[0]['label']
    df['Score'][x] = result[0]['score']

  return df

def personality_trait(df, column_name):
    payload = []
    for x in range(len(df)):
        record = {}
        record['id'] = str(x)
        record['language'] = 'en'
        record['text'] = df[column_name][x]
        payload.append(record)
        
    #Creating batches as the API endpoint can take only 32 records at a time
    batch_size=32
    payload_output = []

    for i in range(0, len(payload), batch_size):
        batch = payload[i:i+batch_size]
        # Do something with the batch
        url = "https://personality-traits.p.rapidapi.com/personality"

        headers = {
            "content-type": "application/json",
            "Accept": "application/json",
            "X-RapidAPI-Key": "9303812463msh066f964925ba79cp183e0ajsn6475462a06c2",
            "X-RapidAPI-Host": "personality-traits.p.rapidapi.com"
        }

        response = requests.request("POST", url, json=batch, headers=headers)

        for x in ast.literal_eval(response.text):
            payload_output.append(x)
            
    all_preds = []
    for p in payload_output:
        preds = {}
        preds['prediction'] = p['predictions'][0]['prediction']
        preds['probability'] = p['predictions'][0]['probability']
        all_preds.append(preds)
        
    preds_df = pd.DataFrame(all_preds)
    output_df = pd.concat([df, preds_df], axis=1)
    
    return output_df

def communication_style(df, column_name):
    payload = []
    for x in range(len(df)):
        record = {}
        record['id'] = str(x)
        record['language'] = 'en'
        record['text'] = df[column_name][x]
        payload.append(record)
        
    #Creating batches as the API endpoint can take only 32 records at a time
    batch_size=32
    payload_output = []
    
    for i in range(0, len(payload), batch_size):
        batch = payload[i:i+batch_size]
        # Processing the batch through API
        url = "https://communication-style.p.rapidapi.com/communication"

        headers = {
                "content-type": "application/json",
                "Accept": "application/json",
                "X-RapidAPI-Key": "9303812463msh066f964925ba79cp183e0ajsn6475462a06c2",
                "X-RapidAPI-Host": "communication-style.p.rapidapi.com"
        }

        response = requests.request("POST", url, json=batch, headers=headers)

        for x in ast.literal_eval(response.text):
            payload_output.append(x)
            
    #Mapping model outputs to the original dataframe
    all_preds = []
    for e in payload_output:
        preds = {}
        for p in e['predictions']:
            if p['prediction'] == "self-revealing":
                preds["self-revealing"] = p['probability']

            elif p['prediction'] == "fact-oriented":
                preds["fact-oriented"] = p['probability']

            elif p['prediction'] == "action-seeking":
                preds["action-seeking"] = p['probability']

            elif p['prediction'] == "information-seeking":
                preds["information-seeking"] = p['probability']
        all_preds.append(preds)
        
    preds_df = pd.DataFrame(all_preds)
    output_df = pd.concat([df, preds_df], axis=1)
    
    return output_df

def big_five_personality(df, column_name):
    payload = []
    for x in range(len(df)):
        record = {}
        record['id'] = str(x)
        record['language'] = 'en'
        record['text'] = df[column_name][x]
        payload.append(record)
        
    batch_size = 15
    payload_output = []

    for i in range(0, len(payload), batch_size):
        batch = payload[i:i+batch_size]
        # Do something with the batch
        url = "https://big-five-personality-insights.p.rapidapi.com/api/big5"

        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": "9303812463msh066f964925ba79cp183e0ajsn6475462a06c2",
            "X-RapidAPI-Host": "big-five-personality-insights.p.rapidapi.com"
        }

        response = requests.request("POST", url, json=batch, headers=headers)

        for x in ast.literal_eval(response.text):
            payload_output.append(x)
            
    preds_df = pd.DataFrame(payload_output)
    preds_df.drop('id', axis=1, inplace=True)

    output_df = pd.concat([df, preds_df], axis=1)
    
    return output_df



# Main Streamlit app
def main():
    st.title("Text Classification Playground")
    
    # Create tabs
    tab_names = ['Viewpoint Classifier', 'Stance Feminist', 'Toxicity Detection', 'Personality Trait', 'Communication Style', 'Big-5 Personality']
    tab_contents = [Viewpoint_classifier, stance_feminist, Toxic_Detection, personality_trait, communication_style, big_five_personality]  
    
    tab = st.selectbox("Select Model:", tab_names)
    idx = tab_names.index(tab)
    
    st.subheader(f"{tab} Model")
    
    # File upload and column input
    uploaded_file = st.file_uploader("Upload CSV or Excel file:", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Read the file into a DataFrame
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('xls') or uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Invalid file type. Please upload a CSV or Excel file.")

        st.dataframe(df.head())
        
        column_name = st.text_input("Enter Column Name to Process:")
        
        if st.button("Submit"):
            if column_name:
                if column_name not in df.columns:
                   st.error(f"The column '{column_name}' does not exist in the uploaded file.")

                else:
                    processed_df = tab_contents[idx](df, column_name)
                    st.dataframe(processed_df)
                    
                    # Offer download link for the processed DataFrame
                    csv = processed_df.to_csv(index=False)
                    st.download_button(
                        label="Download Processed CSV",
                        data=csv,
                        file_name=f"{tab}_processed.csv",
                        mime="text/csv",
                    )

if __name__ == "__main__":
    main()
