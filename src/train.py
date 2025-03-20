from src.preprocessing import load_data, preprocess_data, split_data
from src.model import ChatBotModel
from datasets import Dataset

def train_model():
    # Загрузка данных
    df = load_data('data/train.csv')
    df = preprocess_data(df, character='Joey')
    
    # Разделение данных
    train_df, eval_df = split_data(df)
    
    # Преобразование в формат Dataset
    train_data = Dataset.from_pandas(train_df)
    eval_data = Dataset.from_pandas(eval_df)
    
    # Инициализация и обучение модели
    model = ChatBotModel()
    tokenized_train_data = model.tokenize_data(train_data)
    tokenized_eval_data = model.tokenize_data(eval_data)
    
    model.train(tokenized_train_data, tokenized_eval_data)

if __name__ == "__main__":
    train_model()