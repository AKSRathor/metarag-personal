import pickle
import faiss
from flask import Flask, request
from flask_cors import CORS
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


from dotenv import load_dotenv
load_dotenv()  
app  =Flask(__name__)
CORS(app)

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain(ret2):
    # vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    # retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=ret2,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT}
                                        )

    return chain

def load_vector_db():
    # Loading the vectordb
    
    instructor_embeddings = HuggingFaceInstructEmbeddings()
    index_path = "faiss_index.bin"
    index = faiss.read_index(index_path)

    docstore_path = "docstore.pkl"
    index_to_docstore_id_path = "index_to_docstore_id.pkl"
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)
    with open(index_to_docstore_id_path, "rb") as f:
        index_to_docstore_id = pickle.load(f)
        
    vectordb2 = FAISS(embedding_function=instructor_embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    retriever2 = vectordb2.as_retriever(score_threshold=0.7)
    return retriever2


@app.route("/metarag", methods =['GET', 'POST'])

def respond():
    if(request.method == 'POST'):
        print("started")
        print(request.json["prompt"])
        myprompt = request.json["prompt"]
        myres = chain(myprompt)["result"]
        myoutputres = {"resp":myres}
        print("got response")
        return myoutputres
    return "I don't know"

if __name__ == "__main__":
    # create_vector_db()
    ret2 = load_vector_db()
    global chain
    chain = get_qa_chain(ret2)
    app.run(debug=True)
    # respond(chain)
    # print(chain)
