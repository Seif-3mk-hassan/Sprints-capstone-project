from prepare_kaggle_dataset import PrepareKaggleDataset
from ingest_chroma import IngestChroma

if __name__ == "__main__":
    # Step 1: Prepare the dataset
    prepare_dataset = PrepareKaggleDataset()
    prepare_dataset.prepare_data()
    
    # Step 2: Ingest data into Chroma
    ingest_chroma = IngestChroma()
    ingest_chroma.load_data()
    ingest_chroma.ingest_data()
    
    # Step 3: Test if the ingestion worked
    ingest_chroma.test_if_working()