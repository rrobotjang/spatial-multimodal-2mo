# Task 5 Evidence: LANG Knowledge Extraction

## 1. Weight Load Status

```
[INFO] tut1-model.pt found (442333429 bytes)
[INFO] Weight keys (18 total):
  decoder.attention.W_combine.bias: torch.Size([1])
  decoder.attention.W_combine.weight: torch.Size([1, 512])
  decoder.attention.W_decoder.bias: torch.Size([512])
  decoder.attention.W_decoder.weight: torch.Size([512, 512])
  decoder.attention.W_encoder.bias: torch.Size([512])
  decoder.attention.W_encoder.weight: torch.Size([512, 512])
  decoder.embedding.weight: torch.Size([57237, 256])
  decoder.fc.bias: torch.Size([57237])
  decoder.fc.weight: torch.Size([57237, 512])
  decoder.lstm.bias_hh_l0: torch.Size([2048])
  decoder.lstm.bias_ih_l0: torch.Size([2048])
  decoder.lstm.weight_hh_l0: torch.Size([2048, 512])
  decoder.lstm.weight_ih_l0: torch.Size([2048, 768])
  encoder.embedding.weight: torch.Size([241556, 256])
  encoder.rnn.bias_hh_l0: torch.Size([2048])
  encoder.rnn.bias_ih_l0: torch.Size([2048])
  encoder.rnn.weight_hh_l0: torch.Size([2048, 512])
  encoder.rnn.weight_ih_l0: torch.Size([2048, 256])
```

**Result: tut1-model.pt CANNOT be loaded with notebook classes.**
- Decoder uses LSTM (`decoder.lstm.*`) not GRU (`decoder.rnn.*`)
- Attention keys differ: `W_decoder/W_encoder/W_combine` vs `W1/W2/v`
- Output layer renamed: `decoder.fc` vs `decoder.fc_out`
- Vocab sizes differ: enc=241556, dec=57237 vs 3000/3000 in notebook

Fallback used: smoke-trained tiny model.

## 2. Translation Samples

```
  Sample 1:
    Source (KO):     안녕하세요
    Predicted (EN):  hello
    Reference (EN):  hello

  Sample 2:
    Source (KO):     저는 학생 입니다
    Predicted (EN):  i am a student
    Reference (EN):  i am a student

  Sample 3:
    Source (KO):     오늘 날씨 가 좋습니다
    Predicted (EN):  today the weather is good
    Reference (EN):  today the weather is good
```

All 3 predictions match references exactly.

## 3. Commands and Results

```
$ KMP_DUPLICATE_LIB_OK=TRUE python scripts/seq2seq_ko_en.py
  # Exit code: 0
  # Smoke-trained: 100 epochs, 20 fake KO→EN pairs, final avg_loss=0.0001
  # Architecture: GRU encoder(12,32,64) + BahdanauAttention(64) + GRU decoder(14,32,64)
```

## 4. File Manifest

- `scripts/seq2seq_ko_en.py` — model classes + weight loader + smoke trainer + 3 translations
- `task-5-spatial-multimodal-2mo.md` — knowledge extraction report (in repo root)
- `.omo/evidence/task-5-spatial-multimodal-2mo.md` — this evidence file
