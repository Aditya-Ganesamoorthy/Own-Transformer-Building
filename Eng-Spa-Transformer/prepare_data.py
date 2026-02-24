# prepare_data.py
# Prepare English-Spanish dataset from spa.txt

def prepare_dataset(file_path, max_lines=20000):

    english_sentences = []
    spanish_sentences = []

    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[:max_lines]:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            english_sentences.append(parts[0].lower())
            spanish_sentences.append(parts[1].lower())

    with open("english.txt", "w", encoding="utf-8") as f_en:
        for line in english_sentences:
            f_en.write(line + "\n")

    with open("spanish.txt", "w", encoding="utf-8") as f_es:
        for line in spanish_sentences:
            f_es.write(line + "\n")

    print(f"Saved {len(english_sentences)} sentence pairs.")


if __name__ == "__main__":
    prepare_dataset("spa.txt", max_lines=20000)