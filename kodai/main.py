import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
from chromadb.utils import embedding_functions
import replicate
import os
import time
from dotenv import load_dotenv
import gradio as gr

# init env vars
load_dotenv()
print(os.getenv("OPENAI_API_KEY"),os.getenv("REPLICATE_API_TOKEN"))
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
    op = VectorLoader("openai")
    llama70 = "meta/codellama-13b-python:f7f3a7e9876784f44c970ce0fc0d3aa792ac1570752b9f3b610d6e8ce0bf3220"
    codellamapython13 = "meta/codellama-13b-python:f7f3a7e9876784f44c970ce0fc0d3aa792ac1570752b9f3b610d6e8ce0bf3220"
    llama13 = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
    # op.load_documents()
    template = "You are an AI software engineer at {company}. Your task is to help new {company} software engineers understand codebase and documentation given the context you will be asked, help them only with what is given to you from the context if the context given is not enough, answer with these technologies (rest framework, django, nx) also be helpful and concise include emojis ! ,given the context: {context}"

    examples = ["hey, what can you help me with ?"]
    def get_response(message, history):
        message = f"[INST]{message}[/INST]"
        ff = " "
        for dia in history:
            ff += f"[INST]{dia[0]}[/INST] {dia[1]}"
        ff += message
        qq = op.collection.query(query_texts=[f"{message}"],n_results=5)
        final_qq = []
        for q in range(len(qq["documents"])):
                # we wont take any result that it is not as close.
                if qq["distances"][0][q] < 0.39:
                    final_qq.append(qq["documents"][0][q])
        context = ''.join(final_qq)
        out = replicate.run(
            llama13,
        input = {
            "max_new_tokens":512,
            "temperature":0.7,
            "top_p":0.95,
            "repetition_penalty":1.1,
            "prompt":ff,
            "system_prompt": template.replace("{context}",context).replace("{company}",op.tenant_name),
            "stop_sequences":"<end>,<stop>,\\n"
        }
        ) 
        # queue only accpets generator function which is (get response), then we yeild every chunk of the iterator
        p =""
        for x in out:
            p += x
            yield p

    demo = gr.ChatInterface(get_response,examples=examples)

    demo.queue().launch(share=True)
