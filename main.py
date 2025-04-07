from pythainlp.tokenize import word_tokenize
from difflib import SequenceMatcher
from collections import Counter

# ตัดคำภาษาไทย พร้อมลบช่องว่าง
def tokenize_thai(text):
    tokens = word_tokenize(text, engine='newmm')
    return [t for t in tokens if t.strip() != '']

# ROUGE-1 (Unigram Overlap) เหมือนกัน 1 คำ
def rouge_1_score(ref_tokens, cand_tokens):
    """
    คำนวณคะแนน ROUGE-1 ระหว่างประโยคอ้างอิงและประโยคที่ต้องการตรวจสอบ
    โดยดูจากจำนวนคำ (unigram) ที่ตรงกัน

    Parameters:
        ref_tokens (list): รายการคำจากประโยคอ้างอิง
        cand_tokens (list): รายการคำจากประโยคที่ต้องการเปรียบเทียบ

    Returns:
        tuple: (precision, recall, f1-score) ของ ROUGE-1
    """
    ref_counter = Counter(ref_tokens)
    cand_counter = Counter(cand_tokens)

    overlap = sum((ref_counter & cand_counter).values())
    precision = overlap / len(cand_tokens) if cand_tokens else 0
    recall = overlap / len(ref_tokens) if ref_tokens else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    return precision, recall, f1


# ฟังก์ชันช่วยสร้าง bigrams จากรายการคำ
def get_bigrams(tokens):
    """
    สร้าง bigrams (กลุ่มคำ 2 คำติดกัน) จากรายการคำ

    Parameters:
        tokens (list): รายการคำ (tokens)

    Returns:
        list: รายการ bigram ในรูปแบบของ tuple (word1, word2)
    """
    return list(zip(tokens, tokens[1:]))


# ROUGE-2 (Bigram Overlap) เหมือนกัน 2 คำ (ติดกัน)
def rouge_2_score(ref_tokens, cand_tokens):
    """
    คำนวณคะแนน ROUGE-2 ระหว่างประโยคอ้างอิงและประโยคที่ต้องการตรวจสอบ
    โดยดูจากจำนวน bigram (คำ 2 คำติดกัน) ที่ตรงกัน

    Parameters:
        ref_tokens (list): รายการคำจากประโยคอ้างอิง
        cand_tokens (list): รายการคำจากประโยคที่ต้องการเปรียบเทียบ

    Returns:
        tuple: (precision, recall, f1-score) ของ ROUGE-2
    """
    ref_bigrams = get_bigrams(ref_tokens)
    cand_bigrams = get_bigrams(cand_tokens)

    ref_counter = Counter(ref_bigrams)
    cand_counter = Counter(cand_bigrams)

    overlap = sum((ref_counter & cand_counter).values())
    precision = overlap / len(cand_bigrams) if cand_bigrams else 0
    recall = overlap / len(ref_bigrams) if ref_bigrams else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    return precision, recall, f1


# ROUGE-L (Longest Common Subsequence) คำเหมือนกันยาวที่สุด (ไม่จำเป็นต้องติดกัน)
def rouge_l_score(ref_tokens, cand_tokens):
    """
    คำนวณคะแนน ROUGE-L โดยใช้ความยาวของลำดับคำที่เหมือนกันและเรียงลำดับตรงกัน
    (Longest Common Subsequence: LCS) ระหว่างประโยคอ้างอิงและประโยคเปรียบเทียบ

    Parameters:
        ref_tokens (list): รายการคำจากประโยคอ้างอิง
        cand_tokens (list): รายการคำจากประโยคที่ต้องการเปรียบเทียบ

    Returns:
        tuple: (precision, recall, f1-score) ของ ROUGE-L
    """
    matcher = SequenceMatcher(None, ref_tokens, cand_tokens)
    lcs_len = sum(triple.size for triple in matcher.get_matching_blocks())

    precision = lcs_len / len(cand_tokens) if cand_tokens else 0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    return precision, recall, f1

# รวมเขียนเป็นฟังก์ชั่นเดียว
def validate_thai_sentence(reference, candidate):
    """
    คำนวณค่า ROUGE-1, ROUGE-2 และ ROUGE-L ระหว่างประโยคอ้างอิง (reference)
    และประโยคที่ต้องการตรวจสอบ (candidate) โดยใช้การตัดคำภาษาไทยจาก PyThaiNLP

    Parameters:
        reference (str): ประโยคอ้างอิงภาษาไทย (เช่น ประโยคต้นฉบับหรือคำตอบที่ถูกต้อง)
        candidate (str): ประโยคที่ต้องการเปรียบเทียบ (เช่น คำตอบจากโมเดลหรือผู้เรียน)

    Returns:
        dict: ค่าคะแนนของแต่ละ ROUGE metric ในรูปแบบ:
              {
                "rouge1": (precision, recall, f1),
                "rouge2": (precision, recall, f1),
                "rougeL": (precision, recall, f1)
              }
              โดยที่แต่ละค่าอยู่ในช่วง 0.0 ถึง 1.0
    """
    ref_tokens = tokenize_thai(reference)
    cand_tokens = tokenize_thai(candidate)

    scores = {
        "rouge1": rouge_1_score(ref_tokens, cand_tokens),
        "rouge2": rouge_2_score(ref_tokens, cand_tokens),
        "rougeL": rouge_l_score(ref_tokens, cand_tokens)
    }

    return scores

if __name__ == "__main__":
    text = "เผ่าภูมิ ย้ำวิกฤตเศรษฐกิจ เร่งกระตุ้นผ่าน ดิจิทัลวอลเล็ต"
    tokens = tokenize_thai(text)
    print(tokens)

    # Example usage
    reference = "เผ่าภูมิ ย้ำวิกฤตเศรษฐกิจ เร่งกระตุ้นผ่าน ดิจิทัลวอลเล็ต"
    candidate = "เผ่าภูมิ ยาวิกฤตเศรษฐกิ เร่งกระตุ้นผ่าน ดิจิทัลวอลเล็ต"

    scores = validate_thai_sentence(reference, candidate)

    for metric, (precision, recall, f1) in scores.items():
        print(f"{metric.upper()}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
