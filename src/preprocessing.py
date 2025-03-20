import pandas as pd
import re
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, character='Joey'):
    # Фильтруем данные по персонажу
    df = df[df['label'] == character]
    
    # Очистка текста
    df['friend_response'] = df['friend_response'].str.lower().str.strip()
    df['friend_response'] = df['friend_response'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    
    return df.dropna().drop_duplicates()

def split_data(df, test_size=0.2):
    return train_test_split(df, test_size=test_size, random_state=42)