import os
import json
from dotenv import load_dotenv
from text2tabular.extraction.extractor import StructuredExtractor

# Load environment variables
load_dotenv()


def batch_extract(base_dir):
    api_key = os.getenv("OPENAI_API_KEY")
    extractor = StructuredExtractor(
        api_key, model="gpt-4.1-2025-04-14"  # o3-2025-04-16
    )  # gpt-4.1-mini-2025-04-14 and gpt-4.1-2025-04-14

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("10."):
            pdf_path = os.path.join(folder_path, f"{folder}.pdf")
            json_path = os.path.join(folder_path, f"{folder}.json")
            if os.path.exists(pdf_path):  # and pdf_path.endswith(
                # "10.1001_jamanetworkopen.2019.20511.pdf"
                # ):

                print(f"Extracting from {pdf_path}...")
                try:
                    result = extractor.extract(pdf_path)
                    data_dict = result.model_dump(exclude_none=True)
                    with open(json_path, "w") as f:
                        json.dump(data_dict, f, indent=2)
                    print(f"Saved to {json_path}")
                except Exception as e:
                    print(f"Failed to extract {pdf_path}: {e}")


if __name__ == "__main__":
    base_dir = "src/text2tabular/data/real/replication"
    batch_extract(base_dir)
