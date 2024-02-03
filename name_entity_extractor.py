import pandas as pd
import spacy

#using nlp library for the name entities extraction
nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text):
    doc = nlp(text)
    named_entities = [i.text for i in doc.ents]
    return "; ".join(named_entities)

# Load data from CSV file, but using 1000 data as it will be too big to extract all
data = pd.read_csv('wiki_movie_plots_deduped.csv',nrows=1000)

# Apply extract_named_entities function to each row in the 'plot' column
data['Plot_ner'] = data['Plot'].apply(extract_named_entities)

# Save data to CSV file with the new column
data.to_csv('output_file.csv', index=False)
