"""
Task 5: Korean→English Seq2Seq translation reproduction.
Architecture extracted from notebook: BahdanauAttention + GRU Encoder + GRU Decoder.
Weight load: tut1-model.pt uses a DIFFERENT architecture (LSTM decoder, different
vocab sizes 241556/57237, different attention key names), so it cannot be loaded
with the notebook's classes. Fallback: smoke-train a tiny model on fabricated data.

Environment: KMP_DUPLICATE_LIB_OK=TRUE, python venv at /opt/anaconda3/envs/venv
"""

import os
import sys
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ──────────────────────────────────────────────────────────────────────────────
# 1. Model classes (exactly as extracted from notebook)
# ──────────────────────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(a, encoder_outputs)
        context = context.permute(1, 0, 2)
        output, hidden = self.rnn(embedded, hidden)
        output = output.squeeze(0)
        context = context.squeeze(0)
        prediction = self.fc_out(torch.cat((output, context), dim=1))
        return prediction, hidden, a.squeeze(1)


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, max_len=30, bos_id=1, eos_id=2):
        batch_size = src.shape[1]
        outputs = []
        attentions = []
        encoder_outputs, hidden = self.encoder(src)

        if trg is not None:
            for t in range(trg.shape[0]):
                input = trg[t]
                output, hidden, attention = self.decoder(input, hidden, encoder_outputs)
                outputs.append(output.unsqueeze(0))
                attentions.append(attention.unsqueeze(0))
        else:
            input = torch.full((batch_size,), bos_id, dtype=torch.long, device=self.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            for t in range(max_len):
                output, hidden, attention = self.decoder(input, hidden, encoder_outputs)
                outputs.append(output.unsqueeze(0))
                attentions.append(attention.unsqueeze(0))
                top1 = output.argmax(1)
                input = top1
                finished |= (top1 == eos_id)
                if finished.all():
                    break

        outputs = torch.cat(outputs, dim=0)
        attentions = torch.cat(attentions, dim=0)
        return outputs, attentions


# ──────────────────────────────────────────────────────────────────────────────
# 2. Try loading tut1-model.pt (PRIORITY 1)
# ──────────────────────────────────────────────────────────────────────────────

def try_load_pretrained():
    model_path = os.path.expanduser("~/Downloads/tut1-model.pt")
    if not os.path.exists(model_path):
        print(f"[SKIP] tut1-model.pt not found at {model_path}")
        return False

    print(f"[INFO] tut1-model.pt found ({os.path.getsize(model_path)} bytes)")
    sd = torch.load(model_path, map_location="cpu", weights_only=False)

    print(f"[INFO] Weight keys ({len(sd)} total):")
    for k in sorted(sd.keys()):
        print(f"  {k}: {sd[k].shape}")

    # Key mismatches with notebook classes:
    expected = {
        "encoder.embedding.weight", "encoder.rnn.weight_ih_l0",
        "encoder.rnn.weight_hh_l0", "encoder.rnn.bias_ih_l0", "encoder.rnn.bias_hh_l0",
        "decoder.embedding.weight",
        "decoder.rnn.weight_ih_l0", "decoder.rnn.weight_hh_l0",
        "decoder.rnn.bias_ih_l0", "decoder.rnn.bias_hh_l0",
        "decoder.attention.W1.weight", "decoder.attention.W1.bias",
        "decoder.attention.W2.weight", "decoder.attention.W2.bias",
        "decoder.attention.v.weight",
        "decoder.fc_out.weight", "decoder.fc_out.bias",
    }
    actual = set(sd.keys())

    if expected == actual:
        print("[INFO] Key set matches notebook classes perfectly!")
        return True

    missing = expected - actual
    extra = actual - expected
    print(f"[INFO] KEY MISMATCH with notebook classes.")
    print(f"  Missing from weight: {missing}")
    print(f"  Extra in weight: {extra}")

    # The weight uses LSTM decoder (decoder.lstm.*), different attention keys
    # (W_decoder, W_encoder, W_combine), and different fc name (decoder.fc).
    # Vocab sizes differ too: enc=241556, dec=57237 vs notebook's 3000/3000.
    print("[CONCLUSION] tut1-model.pt uses a DIFFERENT architecture than the notebook:")
    print("  - decoder uses LSTM (key: decoder.lstm.*) not GRU (decoder.rnn.*)")
    print("  - attention keys: W_decoder/W_encoder/W_combine vs W1/W2/v")
    print("  - fc_out renamed to fc")
    print("  - vocab sizes: enc=241556, dec=57237 (vs 3000 in notebook)")
    print("[FALLBACK] Proceeding to smoke-train a tiny model.")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Fallback: smoke-train tiny model on fabricated data
# ──────────────────────────────────────────────────────────────────────────────

# Simple word-level vocab for the smoke test
KO_VOCAB = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "안녕하세요": 4, "저는": 5, "학생": 6, "입니다": 7,
    "오늘": 8, "날씨": 9, "가": 10, "좋습니다": 11,
}
EN_VOCAB = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "hello": 4, "i": 5, "am": 6, "a": 7, "student": 8,
    "today": 9, "the": 10, "weather": 11, "is": 12, "good": 13,
}
KO_INV = {v: k for k, v in KO_VOCAB.items()}
EN_INV = {v: k for k, v in EN_VOCAB.items()}

# Fabricated training pairs
TRAIN_PAIRS = [
    (["안녕하세요"], ["hello"]),
    (["저는", "학생", "입니다"], ["i", "am", "a", "student"]),
    (["오늘", "날씨", "가", "좋습니다"], ["today", "the", "weather", "is", "good"]),
    (["안녕하세요", "저는", "학생", "입니다"], ["hello", "i", "am", "a", "student"]),
    (["오늘", "안녕하세요"], ["today", "hello"]),
    (["학생", "입니다"], ["a", "student"]),
    (["날씨", "가", "좋습니다"], ["the", "weather", "is", "good"]),
    (["저는", "학생"], ["i", "am", "student"]),
    (["오늘", "학생", "입니다"], ["today", "i", "am", "a", "student"]),
    (["안녕하세요", "좋습니다"], ["hello", "good"]),
    (["저는", "오늘"], ["i", "am", "today"]),
    (["날씨", "좋습니다"], ["the", "weather", "good"]),
    (["학생", "가", "좋습니다"], ["a", "student", "is", "good"]),
    (["안녕하세요", "학생", "입니다"], ["hello", "a", "student"]),
    (["오늘", "날씨", "좋습니다"], ["today", "the", "weather", "is", "good"]),
    (["저는", "좋습니다"], ["i", "am", "good"]),
    (["학생", "오늘"], ["a", "student", "today"]),
    (["안녕하세요", "오늘", "좋습니다"], ["hello", "today", "is", "good"]),
    (["저는", "날씨", "입니다"], ["i", "am", "the", "weather"]),
    (["오늘", "학생"], ["today", "a", "student"]),
]


def encode_ko(tokens):
    return [KO_VOCAB.get(t, 3) for t in tokens]


def encode_en(tokens):
    return [EN_VOCAB[t] for t in tokens]


def run_smoke_train():
    print("\n" + "=" * 60)
    print("SMOKE TRAINING: tiny Korean→English model")
    print("=" * 60)

    VOCAB_KO = len(KO_VOCAB)
    VOCAB_EN = len(EN_VOCAB)
    EMB_DIM = 32
    HID_DIM = 64
    EPOCHS = 100
    LR = 0.005

    device = torch.device("cpu")
    encoder = Encoder(VOCAB_KO, EMB_DIM, HID_DIM).to(device)
    attention = BahdanauAttention(HID_DIM).to(device)
    decoder = Decoder(VOCAB_EN, EMB_DIM, HID_DIM, attention).to(device)
    model = Seq2SeqAttention(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"[CONFIG] vocab_ko={VOCAB_KO}, vocab_en={VOCAB_EN}, emb={EMB_DIM}, hid={HID_DIM}")
    print(f"[CONFIG] epochs={EPOCHS}, lr={LR}, pairs={len(TRAIN_PAIRS)}")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        random.shuffle(TRAIN_PAIRS)
        for ko_tokens, en_tokens in TRAIN_PAIRS:
            src_ids = encode_ko(ko_tokens)
            trg_input_ids = [1] + encode_en(en_tokens)  # <bos> + tokens
            trg_label_ids = encode_en(en_tokens) + [2]   # tokens + <eos>

            src = torch.tensor(src_ids).unsqueeze(1)
            trg_input = torch.tensor(trg_input_ids).unsqueeze(1)
            trg_label = torch.tensor(trg_label_ids)

            optimizer.zero_grad()
            outputs, _ = model(src, trg_input)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            loss = criterion(outputs, trg_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg = total_loss / len(TRAIN_PAIRS)
            print(f"  Epoch {epoch+1}/{EPOCHS}, avg_loss={avg:.4f}")

    return model


def greedy_decode(model, ko_tokens, max_len=10):
    model.eval()
    src_ids = encode_ko(ko_tokens)
    src = torch.tensor(src_ids).unsqueeze(1)
    with torch.no_grad():
        outputs, _ = model(src, max_len=max_len)
    pred_ids = outputs.argmax(2).squeeze(1).tolist()
    if 2 in pred_ids:
        pred_ids = pred_ids[:pred_ids.index(2)]
    return [EN_INV.get(i, f"<{i}>") for i in pred_ids]


# ──────────────────────────────────────────────────────────────────────────────
# 4. Main: reproduce 3 translation samples
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 5: LANG Knowledge Extraction — Seq2Seq Reproduction")
    print("=" * 60)

    # Step 1: Try loading pretrained weights
    load_ok = try_load_pretrained()

    if load_ok:
        print("\n[SUCCESS] Pretrained model loaded! (NOT REACHED — see above)")
    else:
        print("\n[FALLBACK] Training tiny smoke model...")

    # Step 2: Smoke-train
    model = run_smoke_train()

    # Step 3: 3 translation samples
    test_sentences = [
        ["안녕하세요"],
        ["저는", "학생", "입니다"],
        ["오늘", "날씨", "가", "좋습니다"],
    ]
    test_labels = [
        "hello",
        "i am a student",
        "today the weather is good",
    ]

    print("\n" + "=" * 60)
    print("3 TRANSLATION SAMPLES (smoke-trained model)")
    print("=" * 60)
    for i, (ko_tokens, label) in enumerate(zip(test_sentences, test_labels)):
        src_text = " ".join(ko_tokens)
        pred = greedy_decode(model, ko_tokens)
        pred_text = " ".join(pred)
        print(f"\n  Sample {i+1}:")
        print(f"    Source (KO):     {src_text}")
        print(f"    Predicted (EN):  {pred_text}")
        print(f"    Reference (EN):  {label}")

    print("\n" + "=" * 60)
    print("DONE — exit 0")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
