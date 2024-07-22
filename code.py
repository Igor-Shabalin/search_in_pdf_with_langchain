# Здесь мы импортируем необходимые библиотеки и модули

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Задаем ключ API OpenAI. В реальном приложении лучше хранить его в переменных окружения
OPENAI_API_KEY = "sk-rWXNdSFDri8IaLDLMWTkFJyAjfwPGQTVnPOKryNF"

def load_pdfs(directory):
    """
    Функция для загрузки PDF-документов из указанной директории.
    """
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            # Создаем загрузчик для каждого PDF-файла
            loader = PyPDFLoader(os.path.join(directory, filename))
            # Загружаем документ и добавляем его в список
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    """
    Функция для разделения документов на меньшие части (чанки).
    """
    # Создаем разделитель текста, который будет разбивать документы на чанки по 1000 символов
    # с перекрытием в 100 символов между чанками
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Разделяем документы на чанки
    texts = text_splitter.split_documents(documents)
    return texts

def create_vectorstore(texts):
    """
    Функция для создания векторного хранилища из текстовых чанков.
    """
    # Создаем объект для генерации эмбеддингов с помощью OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Создаем векторное хранилище Chroma из текстовых чанков и их эмбеддингов
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    """
    Функция для создания цепочки вопросов и ответов.
    """
    # Инициализируем языковую модель ChatGPT
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,  # Устанавливаем температуру 0 для более детерминированных ответов
        openai_api_key=OPENAI_API_KEY
    )
    # Создаем цепочку для ответов на вопросы
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Используем метод "stuff" для объединения документов
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Ограничиваем поиск 3 наиболее релевантными документами
        return_source_documents=True  # Возвращаем исходные документы вместе с ответом
    )
    return qa_chain



def main():
    """
    Основная функция программы.
    """
    # Укажите путь к директории с PDF-файлами
    pdf_directory = "/"

    # Загружаем PDF-документы
    documents = load_pdfs(pdf_directory)
    # Разделяем документы на чанки
    texts = split_documents(documents)
    # Создаем векторное хранилище
    vectorstore = create_vectorstore(texts)
    # Создаем цепочку вопросов и ответов
    qa_chain = create_qa_chain(vectorstore)

    # Основной цикл программы
    while True:
        query = input("Введите ваш вопрос (или 'выход' для завершения): ")
        if query.lower() == 'выход':
            break

        # Получаем ответ на вопрос
        result = qa_chain.invoke({"query": query})
        answer = result['result']
        source_documents = result['source_documents']

        # Выводим ответ
        print(f"\nОтвет: {answer}\n")

        # Выводим информацию о найденных чанках
        print("Найденные чанки документов:")
        for i, doc in enumerate(source_documents, 1):
            print(f"\nЧанк {i}:")
            print(f"Источник: {doc.metadata['source']}, страница {doc.metadata['page']}")
            print(f"Содержимое: {doc.page_content[:200]}...")  # Выводим первые 200 символов

        print("\n" + "-"*50 + "\n")  # Разделитель для удобства чтения
