# ==============================================================================
# text_transforms.py — Xử lý văn bản cho TransVG
# ==============================================================================
# Tất cả tác vụ liên quan đến text:
#   - Tokenize câu mô tả bằng BERT tokenizer
#   - Tạo input_ids, attention_mask cho BERT
#   - Padding/truncation câu về max_query_len
#
# TransVG khác SeqTR ở chỗ:
#   - SeqTR: clean text → build vocab → GloVe embedding → BiGRU
#   - TransVG: text → BERT tokenizer → BERT encoder (end-to-end)
#
# BERT tokenizer tự xử lý:
#   - Lowercase (bert-base-uncased)
#   - Wordpiece: "playing" → ["play", "##ing"]
#   - Thêm special tokens: [CLS] + tokens + [SEP]
#   - Padding + truncation
# ==============================================================================

from transformers import BertTokenizer


class TextTransform:
    """
    Tokenize câu mô tả thành input cho BERT.

    BERT cần 2 tensor:
        - input_ids:      [max_len] — chỉ số token trong vocab BERT
        - attention_mask:  [max_len] — 1 = token thật, 0 = padding

    Format output:
        [CLS] token1 token2 ... tokenN [SEP] [PAD] [PAD] ...
        ← ———————— max_len ——————————————————————————————— →

    Ví dụ (max_query_len=15, max_len=17):
        Input:  "the man in red shirt"
        Output: [CLS] the man in red shirt [SEP] [PAD] [PAD] ...
        IDs:    [101, 1996, 2158, 1999, 2417, 6399, 102, 0, 0, ...]
        Mask:   [1,   1,    1,    1,    1,    1,    1,   0, 0, ...]
    """

    def __init__(self, bert_model="bert-base-uncased", max_query_len=15):
        """
        Args:
            bert_model (str): Tên model BERT trên HuggingFace
            max_query_len (int): Số token text tối đa (KHÔNG tính [CLS], [SEP])
                Tổng chiều dài = max_query_len + 2 (cho [CLS] và [SEP])
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

        # max_query_len = 15 token text thực
        # + 2 special tokens ([CLS] + [SEP]) = 17 tokens tổng
        self.max_len = max_query_len + 2

    def __call__(self, text):
        """
        Tokenize 1 câu text.

        Args:
            text (str): Câu mô tả, ví dụ "the man in red shirt"

        Returns:
            input_ids (list[int]): [max_len] — token IDs
            attention_mask (list[int]): [max_len] — 1 cho token thật, 0 cho PAD
        """
        # encode_plus tự động:
        #   1. Lowercase (vì bert-base-uncased)
        #   2. Tokenize bằng WordPiece
        #   3. Thêm [CLS] ở đầu, [SEP] ở cuối
        #   4. Pad hoặc cắt về max_length
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",      # Pad đến max_length
            truncation=True,           # Cắt nếu dài quá
            return_attention_mask=True,
            return_token_type_ids=False,  # Không cần (chỉ có 1 câu)
        )

        input_ids = encoded["input_ids"]           # list[int], len = max_len
        attention_mask = encoded["attention_mask"]  # list[int], len = max_len

        return input_ids, attention_mask

    def decode(self, input_ids):
        """
        Chuyển input_ids ngược lại thành text (debug/visualization).

        Args:
            input_ids (list[int]): Token IDs

        Returns:
            str: Text đã decode
        """
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    print("=== Test TextTransform ===\n")

    transform = TextTransform(
        bert_model="bert-base-uncased",
        max_query_len=15
    )

    # Test 1: Câu bình thường
    text = "the man in red shirt"
    input_ids, attention_mask = transform(text)
    print(f"Input text:     '{text}'")
    print(f"Input IDs:      {input_ids}")
    print(f"Attention mask: {attention_mask}")
    print(f"Length:          {len(input_ids)} (= 15 + 2)")
    print(f"Decoded:        '{transform.decode(input_ids)}'")

    # Test 2: Câu dài (sẽ bị cắt)
    long_text = "the person wearing a blue hat standing near the large wooden table on the left side of room"
    input_ids_long, mask_long = transform(long_text)
    print(f"\nLong text:      '{long_text}'")
    print(f"Decoded (cut):  '{transform.decode(input_ids_long)}'")
    print(f"Mask sum:       {sum(mask_long)} tokens thật (max {len(mask_long)})")

    # Test 3: Câu ngắn (sẽ bị pad)
    short_text = "left dog"
    input_ids_short, mask_short = transform(short_text)
    print(f"\nShort text:     '{short_text}'")
    print(f"Input IDs:      {input_ids_short}")
    print(f"Attention mask: {mask_short}")
    print(f"Mask sum:       {sum(mask_short)} tokens thật (rest is PAD)")

    print("\n✅ TextTransform test passed!")
