import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
from chromadb.utils import embedding_functions
import replicate
import os
import time
import gradio as gr

class VectorLoader:
    def __init__(self,tenant_name):
        self.tenant_name = tenant_name
        assert os.path.exists(f"data-{tenant_name}") == True, f"data-{tenant_name} file does not exist"

        self.files = os.listdir(f"data-{tenant_name}")
        self.client = chromadb.PersistentClient(path=f"chromadb/{tenant_name}")

        # using default embedding function (all-MiniLM-L6-v2)
        self.embed_openai = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
        self.collection = self.client.get_or_create_collection(name=f"{tenant_name}_collection",
                            embedding_function=self.embed_openai)

    def load_documents(self,):
        print("loading docs")
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,chunk_overlap=150,length_function=len
        # )
        #TODO: test chunk overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,chunk_overlap=150,length_function=len
        )
        # loop through all files and insert each one of them to the DB
        for f in self.files:
            # support for more document types
            if ".md" in f.lower():

                loader =  UnstructuredMarkdownLoader(f"data-{self.tenant_name}/{f}")
                text =loader.load()
                chunks = text_splitter.split_documents(text)
                # because OPENAI has a rate limit
                time.sleep(20)
                self.collection.add(
                    documents=[c.page_content for c in chunks],
                    metadatas = [c.metadata for c in chunks],
                    ids = [f"{index}_{f.lower()}" for index in range(len(chunks))]
                )

if __name__ == "__main__":
    op = VectorLoader("opeani")
    # op.load_documents()
    template = "You are an AI software engineer at company. Your task is to help new company software engineers understand company codebase and documentation given the context you will be asked, help them only with what is given to you from the context if the context given is not enough, answer with these technologies rest framework, django, nx also by helpful and concise include emojis ,given the context: {context}"
    def get_response(prompt):
        qq = op.collection.query(query_texts=[f"{prompt}"],n_results=3)
        final_qq = []
        for q in range(len(qq["documents"])):
                # we wont take any result that it is not as close.
                if qq["distances"][0][q] < 0.39:
                    final_qq.append(qq["documents"][0][q])
        context = ''.join(final_qq)
        out = replicate.run(
        "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
        input = {
            "max_new_tokens":200,
            "temperature":0.75,
            "top_p":0.95,
            "repetition_penalty":1.1,
            "prompt":prompt,
            "system_prompt": template.replace("{context}",context).replace("{company}",op.tenant_name),
            "stop_sequences":"<end>,<stop>,\\n"
        }
        )
        return out

    examples = ["hey, what can you help me with ?"]
    def random_response(message, history):
        return "".join(get_response(message))

    demo = gr.ChatInterface(random_response,examples=examples)

    demo.launch(share=False)
