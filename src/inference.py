from src.model import ChatBotModel

def get_response(input_text):
    model = ChatBotModel()
    model.model.from_pretrained('./results')  # Загрузка обученной модели
    return model.generate_response(input_text)