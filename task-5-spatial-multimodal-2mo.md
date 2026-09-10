# Task 5: LANG Knowledge Extraction — 한-영 Seq2Seq + Bahdanau Attention

## (1) 모델 구조

Notebook extracts a **Bahdanau-attention Seq2Seq** with single GRU layers:

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v  = nn.Linear(hidden_dim, 1, bias=False)

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
    # Wraps Encoder+Decoder; forward(src, trg) for teacher-forcing,
    # forward(src) for greedy inference with early stopping on <eos>.
```

**Key hyperparams from notebook:**
- `input_dim = output_dim = 3000` (SentencePiece vocab)
- `emb_dim = 256`, `hid_dim = 512`
- Encoder GRU: Embedding(3000→256) → GRU(256→512)
- Decoder GRU: Embedding(3000→256) → GRU(256→512) → Linear(1024→3000)
- BahdanauAttention: W1(512→512), W2(512→512), v(512→1, no bias)

**tut1-model.pt architecture (different from notebook):**
- Encoder: GRU(256→512), vocab 241556
- Decoder: **LSTM**(768→512), vocab 57237
- Attention: W_decoder(512→512), W_encoder(512→512), W_combine(512→1)
- Output: fc(512→57237)
- **Conclusion:** weight file is from a different training run; cannot load with notebook classes.

## (2) 데이터 준비

**Notebook EN→ES pipeline (cells 5-17):**
```python
# Download spa-eng corpus (118,964 pairs)
df = pd.read_csv(path_to_file, sep="\t", names=["eng", "spa"])

# Preprocessing: lowercase, split punctuation, strip special chars
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[ " "]+', " ", sentence)
    sentence = re.sub(r"[^a-záéíóúüñ?.!,]+", " ", sentence)
    return sentence.strip()

# SentencePiece tokenizers (vocab=3000, pad=0, bos=1, eos=2, unk=3)
spm.SentencePieceTrainer.train(input="eng_corpus.txt", model_prefix="encoder_spm",
    vocab_size=3000, pad_id=0, bos_id=1, eos_id=2, unk_id=3)
spm.SentencePieceTrainer.train(input="spa_corpus.txt", model_prefix="decoder_spm",
    vocab_size=3000, pad_id=0, bos_id=1, eos_id=2, unk_id=3)

# Dataset: MAX_LEN=30, BATCH_SIZE=64, 80/20 split
class TranslationDataset(Dataset):
    # src_ids padded to max_len
    # trg_input = [bos] + token_ids + [eos], padded
    # trg_label = token_ids + [eos], padded
```

**Korean→English project pipeline (cells 38-41):**
```python
# Korean-English Park corpus (94,123 pairs → 78,968 after dedup)
# preprocessing() redefined to preserve Hangul: [가-힣ㄱ-ㅎㅏ-ㅣ0-9?.!,]
# Korean tokenization: KoNLPy Mecab morphological analysis
# English tokenization: split() + <start>/<end> tokens
# Max token length ≤ 40 filter
```

## (3) 메트릭 로그 (from notebook outputs)

**EN→ES training (10 epochs, no validation):**
| Epoch | Train Loss |
|-------|-----------|
| 1 | 2.6346 |
| 2 | 1.4320 |
| 3 | 1.0444 |
| 4 | 0.8013 |
| 5 | 0.6356 |
| 6 | 0.5172 |
| 7 | 0.4390 |
| 8 | 0.3828 |
| 9 | 0.3432 |
| 10 | 0.3209 |

**EN→ES training (20 epochs, with validation):**
| Epoch | Train Loss | Validation Loss |
|-------|-----------|----------------|
| 1 | 0.3034 | 1.8925 |
| 5 | 0.2736 | 2.0793 |
| 10 | 0.2662 | 2.2481 |
| 15 | 0.2759 | 2.3740 |
| 20 | 0.2930 | 2.4699 |

Observation: Train loss converges fast but validation loss diverges (overfitting).

**Notebook translation examples (EN→ES):**
- "the most powerful man all over the world." → `el más sabr.oso es igual al mundo todo el mundo .`
- "may i help you ?" → `puedo ayudarle ?`
- "can i have some coffee ?" → `me puede alguno de café ?`

## (4) 재현 Smoke 로그

**tut1-model.pt load result:**
```
[INFO] tut1-model.pt found (442333429 bytes)
[CONCLUSION] tut1-model.pt uses a DIFFERENT architecture than the notebook:
  - decoder uses LSTM (key: decoder.lstm.*) not GRU (decoder.rnn.*)
  - attention keys: W_decoder/W_encoder/W_combine vs W1/W2/v
  - fc_out renamed to fc
  - vocab sizes: enc=241556, dec=57237 (vs 3000 in notebook)
[FALLBACK] Proceeding to smoke-train a tiny model.
```

**Smoke training config:**
- vocab_ko=12, vocab_en=14, emb_dim=32, hid_dim=64
- 20 fabricated Korean→English pairs
- 100 epochs, Adam lr=0.005
- Final avg_loss=0.0001

**3 Translation Samples:**
```
Sample 1:
  Source (KO):     안녕하세요
  Predicted (EN):  hello
  Reference (EN):  hello  ✓

Sample 2:
  Source (KO):     저는 학생 입니다
  Predicted (EN):  i am a student
  Reference (EN):  i am a student  ✓

Sample 3:
  Source (KO):     오늘 날씨 가 좋습니다
  Predicted (EN):  today the weather is good
  Reference (EN):  today the weather is good  ✓
```

All 3 predictions match references exactly. Exit code: 0.
