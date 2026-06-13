import pdfplumber

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


pdf_path = "D:/Vishal_Files/Coding/Python Projects/stock-RAG/data/CCL-Final_Report_2024_9-07pm.pdf"
print(extract_text(pdf_path))

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split() #break the input into a list of words
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
