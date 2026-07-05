import numpy as np
import pandas as pd
import streamlit as st
import pickle
import torch
import torch.nn as nn 
import re
from keras.utils import pad_sequences
from torch.utils.data import TensorDataset, DataLoader
import nltk
from nltk.corpus import stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def Home_Page():
    st.title("Comment Toxicity Detection")
    st.write("Online platforms today rely heavily on user interaction, but this openness also brings a rise in harmful and abusive comments. Toxic behaviour such as hate speech, harassment, and offensive expressions can disrupt conversations and create unsafe digital spaces. Because of this, there is a strong need for systems that can automatically identify problematic comments. This project focuses on building a Python based deep learning model that can examine comment text and determine how likely it is to be toxic, helping platforms maintain healthier communication.")
    st.write("In this project, two models were taken and these two models are trained on massive dataset.\n\n**1.LSTM**\n\n**2.BERT**")
    st.write("The detailed comparision of two models results is given in 'Comparision' page.")
    st.write("To check toxicity in your comments, go to 'Check toxicity in your comments' page. ")


def Comparisions_Page():
    st.title("Comparing LSTM and BERT")

    Metrics_data = {'Models':['LSTM', 'BERT'], 'Train Loss':[0.0711, 0.0217], 'Train Accuracy':[97.63, 99.14], 'Test Loss':[0.0683, 0.0475], 'Test Accuracy':[97.77, 98.44]}
    Metrics_df = pd.DataFrame(Metrics_data).set_index('Models')

    st.subheader("📈Metrics Comparision Charts")
    st.write('\n')
    col1, col2 = st.columns(2, gap='xxlarge')
    col1.subheader("Loss", text_alignment='center')
    col1.bar_chart(Metrics_df[['Train Loss', 'Test Loss']])
    col2. subheader("Accuracy", text_alignment='center')
    col2.bar_chart(Metrics_df[['Train Accuracy', 'Test Accuracy']])

    st.write("By this it is clear that both LSTM and BERT are performing good.")
    st.success("🏅BERT has low error and high accuracy comparitively.")

    st.write('\n')

    st.subheader("🧠Predictions Made By Both Models")
    LSTM_Pred = pd.read_csv('C:/Users/HP/OneDrive/Desktop/GUVI-DS PROJECT REPORTS/FinalTestData.csv')
    BERT_Pred = pd.read_csv('C:/Users/HP/OneDrive/Desktop/GUVI-DS PROJECT REPORTS/Final-BERT-TestData.csv')
    LSTM_Pred.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
    BERT_Pred.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)

    st.write("Few comments were given below, click on the respective buttons to understand how well each model is predicting the toxicity of comments.")
    col3, col4 = st.columns(2, gap='xxlarge')
    col3.write(f"**Comment1:** {BERT_Pred.iloc[81, 0]}")
    first = col4.button('Predict Comment1', type='primary')
    col3.write(f"**Comment2:** {BERT_Pred.iloc[153159, 0]}")
    second = col4.button('Predict Comment2', type='primary')
    col3.write(f"**Comment3:** {LSTM_Pred.iloc[5, 0]}")
    third = col4.button('Predict Comment3', type='primary')
    col3.write(f"**Comment4:** {LSTM_Pred.iloc[153163, 0]}")
    fourth = col4.button('Predict Comment4', type='primary')
    
    col5, col6 = st.columns(2, gap='xxlarge')
    if first:
        col5.subheader("LSTM")
        col5.dataframe(LSTM_Pred.iloc[[81], 1:], hide_index=True)
        col6.subheader("BERT")
        col6.dataframe(BERT_Pred.iloc[[81], 1:], hide_index=True)
    if second:
        col5.subheader("LSTM")
        col5.dataframe(LSTM_Pred.iloc[[153159], 1:], hide_index=True)
        col6.subheader("BERT")
        col6.dataframe(BERT_Pred.iloc[[153159], 1:], hide_index=True)
    if third:
        col5.subheader("LSTM")
        col5.dataframe(LSTM_Pred.iloc[[5], 1:], hide_index=True)
        col6.subheader("BERT")
        col6.dataframe(BERT_Pred.iloc[[5], 1:], hide_index=True)
    if fourth:
        col5.subheader("LSTM")
        col5.dataframe(LSTM_Pred.iloc[[153163], 1:], hide_index=True)
        col6.subheader("BERT")
        col6.dataframe(BERT_Pred.iloc[[153163], 1:], hide_index=True)
    
    st.markdown("### 📊 Model Head-to-Head Comparison")
    st.markdown("""
    | Feature | LSTM Model | BERT Model | Winner |
    | :--- | :--- | :--- | :--- |
    | **Accuracy** | **97.77%**  | **98.48%**  | **BERT (+0.7%)** |
    | **Speed / Latency** | **Near-Instant** (few ms) | **Slow on CPU** (up to 400ms) | 🏆 **LSTM** |
    | **RAM / Hardware** | **Very Lightweight** | **Heavy** (Causes RAM crashes) | 🏆 **LSTM** |
    """)
    
    st.info("💡 **Verdict:** While BERT holds a minor 0.7% accuracy advantage, **LSTM is the chosen model for prediction** due to its instant speed, low memory usage, and better generalization on raw text inputs.")


def Predictions_Page():
    st.subheader("To predict toxicity in your comments, either provide a single comment or upload a csv/excel file containing multiple comments.")
    col7, col8 = st.columns(2, gap='xxlarge')
    single_input = col7.text_input("Enter the comment")
    Single_Comment_Predict = col7.button("Single_Comment_Predict", type='primary')
    file = col8.file_uploader(label="Choose a CSV or Excel file containing comments", type=["csv", "xlsx"])
    text_column = col8.text_input("Enter the comments column name")
    Multiple_Comments_predict = col8.button("Multiple_Comments_predict", type='primary')
    
    
    class PyTorchLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super(PyTorchLSTM, self).__init__()

            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True
            )

            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, text):
            embedded = self.embedding(text)
            lstm_out, (hidden, cell) = self.lstm(embedded)
            last_step_out = lstm_out[:, -1, :]

            out = self.dropout(last_step_out)
            return self.fc(out)
    

    with open('keras_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    vocab_size = 30000 

    device = torch.device("cpu")
    model = PyTorchLSTM(vocab_size=vocab_size, embedding_dim=128, hidden_dim=64, output_dim=6) 
    model.load_state_dict(torch.load("pytorch_toxicity_lstm.pt", map_location=device))
    model.to(device)
    model.eval()

    def predict_probabilities(text):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        words = text.split()
        stop_words = stopwords.words('english')
        clean_words = [word for word in words if word not in stop_words]
        return " ".join(clean_words).strip()
    

    column_names = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]

    if Single_Comment_Predict:
        cleaned_text = predict_probabilities(single_input)
        sequences = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequences, maxlen=128, padding='post', truncating='post')
        input_tensor = torch.tensor(padded_sequence, dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.sigmoid(logits)
        
        scores = [round(float(p) * 100, 2) for p in probabilities[0]]
        for label, score in zip(column_names, scores):
            st.write(f"{label}: {score}%")
    
    if Multiple_Comments_predict:
        
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
            
        df[text_column] = df[text_column].astype(str)
            
        df[text_column] = df[text_column].apply(predict_probabilities)
        sequences = tokenizer.texts_to_sequences(df[text_column].values)
        X_test_data = pad_sequences(sequences, maxlen=128, padding='post', truncating='post')
        batch_size = 64  
        test_tensor = torch.tensor(X_test_data, dtype=torch.long)
        dataset = TensorDataset(test_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                batch_inputs = batch[0].to(device)
                logits = model(batch_inputs)
                probabilities = torch.sigmoid(logits)
        
                percentage_scores = probabilities * 100
                all_probabilities.append(percentage_scores.cpu())
                
        final_scores_matrix = torch.cat(all_probabilities, dim=0).numpy()

        final_scores_matrix = np.round(final_scores_matrix, 2)
        column_names = ["Toxic %", "Severe Toxic %", "Obscene %", "Threat %", "Insult %", "Identity Hate %"]
        pred_df = pd.DataFrame(final_scores_matrix, columns=column_names)

        final_df = pd.concat([df, pred_df], axis=1)
            
        st.write("#### ✨ Finished Batch Processing Results")
        st.dataframe(final_df.head(), use_container_width=True)
        st.write("#### ⬇️ Download Complete Dataset")

        csv_data = final_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_data,
            file_name="toxic_predictions_output.csv",
            mime="text/csv",
            use_container_width=True
        )
            


home = st.Page(Home_Page, title='Home')
comparison = st.Page(Comparisions_Page, title='Comparison')
pred = st.Page(Predictions_Page, title='Check toxicity in your comments')

pg = st.navigation([home, comparison, pred])
pg.run()
