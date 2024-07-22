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
