import json
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load BlueBERT tokenizer and model
# model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
model_name = "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a pipeline for Named Entity Recognition (NER)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

import json

# Load your medical records JSON file
with open("corpus/medical_records_corpus.json", "r") as file:
    records = json.load(file)



# Process each record with BlueBERT
processed_records = []
k=0
for record in records:
    text = f"Medical history: {', '.join(record['medical_history'])}. " \
           f"Diagnoses: {', '.join(record['current_diagnoses'])}. " \
           f"Treatments: {', '.join(record['treatments'])}."
    print(k)
    k+=1
    # Apply BlueBERT NER
    entities = ner_pipeline(text)

    # Store the results in the record
    processed_record = {
        "record_id": record["record_id"],
        "original_text": text,
        "entities": entities
    }
    processed_records.append(processed_record)

import numpy as np

# Helper function to recursively convert numpy.float32 to native Python float
def convert_floats(obj):
    if isinstance(obj, dict):
        return {key: convert_floats(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(item) for item in obj]
    elif isinstance(obj, np.float32):
        return float(obj)  # Convert np.float32 to native float
    else:
        return obj

processed_records_converted = convert_floats(processed_records)

# Save to JSON
with open("processed_medical_records.json", "w") as outfile:
    json.dump(processed_records_converted, outfile, indent=4)

print("Processing complete. Results saved to processed_medical_records.json.")


# Load the SciBERT model
# nlp = spacy.load("en_core_sci_scibert")

# Load the medical corpus
input_file = "corpus/medical_records_corpus.json"  # Replace with your file name
output_file = "processed_medical_records_spacy.json"


processed_records = []
k=0
# Process each record in the corpus
for record in records:
    text = f"Medical history: {', '.join(record['medical_history'])}. " \
           f"Diagnoses: {', '.join(record['current_diagnoses'])}. " \
           f"Treatments: {', '.join(record['treatments'])}."
    print(k)
    k+=1
    # Use spaCy's NLP pipeline
    doc = ner_pipeline(text)

    # Extract entities recognized by the model
    entities = [
        {
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "label": ent.label_
        }
        for ent in doc.ents
    ]

    # Add processed data to the record
    processed_record = {
        "record_id": record["record_id"],
        "original_text": text,
        "entities": entities
    }
    processed_records.append(processed_record)

# Save the processed data
with open(output_file, "w") as file:
    json.dump(processed_records, file, indent=4)

print(f"Processed records saved to {output_file}")
