## Comment Toxicity Detection App

An end-to-end Natural Language Processing (NLP) application that evaluates user comments for toxic language.

During development, two distinct architectures were trained and evaluated: **BERT** (Bidirectional Encoder Representations from Transformers) and an **LSTM** (Long Short-Term Memory) recurrent neural network. To optimize the user experience for real-time inference, the lightweight **LSTM model** was selected for prediction due to its speed and lower computational latency.

---

### ✨ Features

*   **Real-time Prediction:** Type any comment into the interface and instantly see the toxicity classification.
*   **Robust Preprocessing:** Built-in text cleaning, NLTK stopword removal, and sequence padding.
  
---

### 🧠 Architectural Trade-offs & Model Selection

Model | Strengths | Trade-offs / Weaknesses | Role in Project
| :--- | :--- | :--- | :--- |
| **BERT** | Exceptionally high semantic accuracy, deep contextual understanding. | High computational overhead, large file size, slow inference speed. | **Evaluation Reference** |
| **LSTM** | **Fast inference times**, highly lightweight, low hardware requirements. | Slightly less contextual nuance than transformers.| **Real-time Prediction** |

---

### 📂 Project Structure

    ├── StreamlitApp.py                # Main Streamlit web application script
    ├── BERT_Model_Training.ipynb      # Dataset preprocessing, training, and evaluation of BERT
    └── LSTM_Model_Training.ipynb      # Dataset preprocessing, training, and evaluation of LSTM
    ├── requirements.txt               # Managed Python dependencies
    ├── keras_tokenizer.pkl            # Trained Keras tokenizer for text-to-sequence conversion
    ├── pytorch_toxicity_lstm.pt       # Trained LSTM model 
    └── README.md                      # Project documentation    

---

### 🛠️ Built With

*   **Deep Learning Frameworks:** PyTorch, TensorFlow / Keras, Hugging Face Transformers (Training stage)
*   **Frontend UI:** Streamlit
*   **NLP Tools:** NLTK, Regular Expressions (`re`)
*   **Data Processing:** NumPy, Pandas
