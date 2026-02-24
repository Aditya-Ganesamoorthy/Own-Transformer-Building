def reduce_kftt(en_path, ja_path, max_lines=20000):
    with open(en_path, encoding="utf-8") as f_en:
        en_lines = f_en.readlines()[:max_lines]

    with open(ja_path, encoding="utf-8") as f_ja:
        ja_lines = f_ja.readlines()[:max_lines]

    with open("english.txt", "w", encoding="utf-8") as f_en_out:
        f_en_out.writelines(en_lines)

    with open("japanese.txt", "w", encoding="utf-8") as f_ja_out:
        f_ja_out.writelines(ja_lines)

    print(f"Saved {len(en_lines)} aligned sentence pairs.")

if __name__ == "__main__":
    reduce_kftt(
        "kyoto-train.en",
        "kyoto-train.ja",
        max_lines=20000
    )