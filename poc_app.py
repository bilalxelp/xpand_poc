import streamlit as st
import pandas as pd
import base64
from transformers import pipeline
from tqdm.auto import tqdm
import re
import stanza
NLP = stanza.Pipeline(lang='en', processors='tokenize')


def stanza_tokenizer(text):
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


# Function to process the selected tab
def process_data(df, column_name, tab_name):
    if tab_name == 'Viewpoint Classifier':
        # Process data using option 1
        df = df[:10]
        df['Sentences'] = df[column_name].apply(lambda x: stanza_tokenizer(x))
        df = df.explode("Sentences").reset_index(drop=True)
        df = Viewpoint_classifier(df, "Sentences")

    elif tab_name == 'Stance Feminist':
        # Process data using option 2
        df = df[:10]
        df['Sentences'] = df[column_name].apply(lambda x: stanza_tokenizer(x))
        df = df.explode("Sentences").reset_index(drop=True)
        df = stance_feminist(df, "Sentences")

    elif tab_name == 'Toxicity Detection':
        # Process data using option 3
        df = df[:10]
        df['Sentences'] = df[column_name].apply(lambda x: stanza_tokenizer(x))
        df = df.explode("Sentences").reset_index(drop=True)
        df = Toxic_Detection(df, "Sentences")

    # Add more options as needed
    
    return df

# Define the Streamlit app
def main():
    # Set the page title and icon
    # st.set_page_config(page_title='Data Processor', page_icon=':pencil:')
    st.set_page_config(page_title='My Streamlit App', page_icon=':bar_chart:', layout='wide', initial_sidebar_state='auto')


    # Set the sidebar options
    options = ['Viewpoint Classifier', 'Stance Feminist', 'Toxicity Detection']
    # Add more options as needed

    # Set the sidebar inputs
    uploaded_file = st.file_uploader("Upload a CSV or Excel file")
    column_name = st.text_input("Enter the column name to process")

    # Check if file was uploaded and column name is provided
    if uploaded_file is not None and column_name != "":
        # Read the file into a DataFrame
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('xls') or uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Invalid file type. Please upload a CSV or Excel file.")

        # Check if the selected column exists in the DataFrame
        if column_name not in df.columns:
            st.error(f"The column '{column_name}' does not exist in the uploaded file.")
        else:
            # Process the data based on the selected option
            option_selected = st.sidebar.selectbox("Select a model:", options)
            processed_data = process_data(df, column_name, option_selected)

            # Display the processed data
            st.write("Processed Data:")
            st.write(processed_data)

            # Allow the user to download the processed data as a CSV file
            csv = processed_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download processed data</a>'
            st.markdown(href, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
