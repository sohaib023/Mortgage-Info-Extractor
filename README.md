# Mortgage-Info-Extractor
This program takes as input a PDF file of a mortgage agreement and extracts 4 fields from the agreement:
  - Name of Borrower
  - Address of property to be mortgaged
  - Account Number
  - Interest Rate
  
The python script will require you to set-up Google Cloud Vision API and Stanford NER.

Different pre-processing techniques have been applied to minimize the noise and increase the OCR accuracy. Since there were multiple similar fields to each of the desired ones, thus heuristics were applied to classify the text according to the other text in its locality. 

![image not found](resources/0org.png?raw=true "Original")
