import os
import json
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
#from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

llm = Ollama(model="mistral-nemo:latest", base_url='http://localhost:11434/', request_timeout=120.0)
embed_model = OllamaEmbedding(model_name="snowflake-arctic-embed:latest", base_url='http://localhost:11434/')

Settings.llm = llm
Settings.embed_model = embed_model

#llama_debug = LlamaDebugHandler(print_trace_on_end=True)
# creating Callback manager object to show each sub process details \
#callback_manager = CallbackManager([llama_debug])

def load_pdf(file_path):
    reader = SimpleDirectoryReader(input_dir=file_path, recursive=True)
    data = reader.load_data()
    return data


def build_engine(data):
    index = VectorStoreIndex.from_documents(data, embed_model=embed_model)
    # custom prompt template
    template = ("""
        "Imagine you are an advanced Principal Investigator in medical research for neuropathy, with access to 
        "documents in regards to how Ivabradine impacts neuropathy and the autonomic nervous system, "
        "case studies, and expert analyses. Your goal is to provide insightful, accurate, and case study to push the field further.\n\n"
        "Here is some context related to the query:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following inquiry with detailed references to applicable research, "
        "or expert analyses where appropriate:\n\n"
        "Question: {query_str}\n\n"
        "Answer succinctly, starting with the phrase 'According to recent literature,' and ensure your response is understandable to other advanced researchers."
        "Answer in the the following JSON format: "
                
        "Background": "In this literature review of Ivabradine in regards to its affects on diabetic neuropathy and POTS, ...",
        "Results": "The literature review examines...",
        "Conclusion": "Our study shows that ivabradine is effective for patients with POTS as well as small fiberneuropathy...",
        "Introduction": "The postural tachycardia syndrome (POTS) is a heterogeneous group of disorders that result in a disproportionate increase in heart rate upon standing accompanied by symptoms of orthostatic intolerance.",
        "Discussion": "While Ivabradine effectively reduces heart rate in POTS patients, its impact on neuropathic pain..."
        """
    )
    qa_template = PromptTemplate(template)
    query_engine = index.as_query_engine(streaming=True, text_qa_template=qa_template, similarity_top_k=3)
    return query_engine


def create_case_study_report(query_engine):
    response = query_engine.query(
        "Create a rough draft case study report from the given medical research papers."
    )
    return response.response_txt


if __name__ == "__main__":
    # Replace 'path/to/your/pdf1.pdf', 'path/to/your/pdf2.pdf' with your actual PDF paths
    pdf_files = r'C:\Users\Lorin\Downloads\ivabradine'

    loaded_pdfs = load_pdf(pdf_files)
    engine = build_engine(loaded_pdfs)
    case_study_report = create_case_study_report(engine)
    if case_study_report:
        json_dict = json.load(case_study_report)
        print(json_dict)