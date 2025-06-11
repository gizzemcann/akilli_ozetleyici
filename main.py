from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import PyPDF2

def read_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def main():
    # Hugging Face erişim token'ı
    hf_token = "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"

    # Model bilgisi
    model_name = "csebuetnlp/mT5_multilingual_XLSum"

    # Model ve tokenizer yükle
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=hf_token)

    # Özetleme pipeline'ı
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    # 📄 PDF dosyasının yolu
    pdf_path = "ornek.pdf"  # Kendi dosya adınla değiştir
    full_text = read_pdf(pdf_path)

    # Uzun metni parçalara ayır
    chunk_size = 1000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    all_summaries = []

    print("📢 Özetler:")
    for i, chunk in enumerate(chunks):
        summary = summarizer(chunk, max_length=60, min_length=25, do_sample=False)[0]['summary_text']
        print(f"\n👉 Bölüm {i+1} Özeti:\n{summary}")
        all_summaries.append(summary)

    # 🔄 Tüm özetleri birleştir ve son özetle
    combined_text = " ".join(all_summaries)
    final_summary = summarizer(combined_text, max_length=243, min_length=40, do_sample=False)[0]['summary_text']

    print("\n🟩 BİRLEŞTİRİLMİŞ TEK ÖZET 🟩\n")
    print(final_summary)
if __name__ == "__main__":
    main() 