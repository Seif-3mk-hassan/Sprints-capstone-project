import os
import shutil
import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
import chromadb
from langchain_chroma import Chroma

support_conversations_PATH = os.path.join('Tasks', 'project-sprints', 'Sprints-capstone-project', 'NLP_Capstone_project', 'data', 'support_conversations.csv')
dotenv.load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')

class IngestChroma:
    def __init__(self):
        self.data_path = support_conversations_PATH
        self.data = None
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        self.collection_name = 'support_tickets'
        self.persist_directory = self.persist_directory = os.path.join('data', 'chroma_db')
        self.texts = []
        self.metadatas = []
        self.ids = []

    def load_data(self):
        import pandas as pd
        self.data = pd.read_csv(self.data_path)

    def ingest_data(self, fresh=False):
        if fresh and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"Cleared existing {self.persist_directory}/")

        for idx, row in enumerate(self.data.itertuples()):
            text = str(row.customer_issue).strip()
            reply = str(row.reference_reply).strip()
            if not text or not reply:
                continue
            self.texts.append(text)
            self.metadatas.append({'reference_reply': reply})
            self.ids.append(f"doc_{idx}")

        print(f"Prepared {len(self.texts)} documents")

        client = chromadb.PersistentClient(path=self.persist_directory)
        collection = client.get_or_create_collection(name=self.collection_name)

        existing_ids = set(collection.get()["ids"])
        print(f"Already in collection: {len(existing_ids)}")

        batch_size = 20
        max_retries = 3
        skipped = 0
        for i in range(0, len(self.texts), batch_size):
            batch_texts = self.texts[i:i + batch_size]
            batch_metas = self.metadatas[i:i + batch_size]
            batch_ids = self.ids[i:i + batch_size]

            new_idx = [j for j, doc_id in enumerate(batch_ids) if doc_id not in existing_ids]
            if not new_idx:
                skipped += len(batch_ids)
                continue

            batch_texts = [batch_texts[j] for j in new_idx]
            batch_metas = [batch_metas[j] for j in new_idx]
            batch_ids = [batch_ids[j] for j in new_idx]

            for attempt in range(max_retries):
                try:
                    batch_embeddings = [
                        self.embeddings.embed_query(t) for t in batch_texts
                    ]
                    collection.add(
                        ids=batch_ids,
                        documents=batch_texts,
                        embeddings=batch_embeddings,
                        metadatas=batch_metas,
                    )
                    print(f"Ingested {min(i + batch_size, len(self.texts))}/{len(self.texts)} (+{len(batch_ids)} new)")
                    break
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                        wait = 60 * (attempt + 1)
                        print(f"Rate limited on batch {i}-{i+batch_size}, waiting {wait}s (retry {attempt+1}/{max_retries})...")
                        time.sleep(wait)
                    else:
                        print(f"Error on batch {i}-{i+batch_size}: {e}")
                        break
            time.sleep(15)

        if skipped:
            print(f"Skipped {skipped} already-ingested documents")
        print(f"Done! Total in collection: {collection.count()}")

    def test_if_working(self):
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        print(f"Documents in Chroma: {db._collection.count()}")

        query = "How can I reset my password?"
        results = db.similarity_search(query, k=3)

        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print("Customer Issue:", result.page_content)
            print("Reference Reply:", result.metadata['reference_reply'])
            print("-" * 50)

        print("=== Chroma DB Summary ===")
        print(f"Total documents: {db._collection.count()}")
        print(f"Embedding model: models/gemini-embedding-2")
        print(f"Persist directory: {self.persist_directory}/")
        print(f"Collection name: {self.collection_name}")
        print(f"Avg issue length: {self.data['customer_issue'].str.len().mean():.0f} chars")
        print(f"Avg reply length: {self.data['reference_reply'].str.len().mean():.0f} chars")
        print(f"Min issue length: {self.data['customer_issue'].str.len().min()} chars")
        print(f"Max issue length: {self.data['customer_issue'].str.len().max()} chars")
