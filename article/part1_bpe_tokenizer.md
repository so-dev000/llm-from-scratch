# はじめに

こんにちは。
本シリーズ「ゼロから作るLLM」では、大学生の筆者がLLMをゼロから実装していきます。

# レギュレーション

PyTorchを用いて実装を進めていきますが、LLMの仕組みを理解することが目的なので次のような制約を設けることにしました。

- `nn.Linear`, `nn.Dropout` 等の基本レイヤーは使用可
- `torch.nn.functional` の基本関数（softmax, relu等）は使用可
- `DataLoader` は使用可
- `nn.Transformer`, `nn.MultiheadAttention` 等の高レベルモジュールは使用せず, 自作する

# 前提知識

- ニューラルネットワークと自然言語処理の基本概念
  - ゼロから作るディープラーニングの1と2を読むのがおすすめです
- PyTorch の基本的な使い方

# トークナイザーとは

LLMはテキストを直接扱えず, 内部では数値（トークンID）の列として処理しています。
この「テキスト ↔ トークンID列」の変換を行うのがトークナイザーです。
例を見てみるのが早いので, ブラウザ上でトークナイザーの挙動を確認できるサイト
[tiktokenizer](https://tiktokenizer.vercel.app/?model=gpt2) で実際の動作を見てみましょう。

例（GPT-2トークナイザー）:

- `the` → `1169`（1トークン）
- `Hello` → `15496`（1トークン）
- `hello` → `31373`（1トークン）
- `DeepSeek` → `29744, 4653, 988`（3トークン、`Deep` + `Se` + `ek`）
- `こんにちは` → `46036, 22174, 28618, 2515, 94, 31676`（6トークン）

観察できること:

- 頻出語は1トークンにまとめられる
- 大文字小文字で別トークンになる（`Hello` ≠ `hello`）
- GPT2登場時に存在しなかった未知語（`DeepSeek`）も既存のサブワードに分割して表現できる
- 学習データに少ない言語（日本語など）は細かく分割される

# トークナイザーの種類

トークナイザーには主に3つのアプローチがあります。

- **文字単位**
  - 例: `Hello` → `H`, `e`, `l`, `l`, `o`
- **単語単位**
  - 例: `Hello world` → `Hello`, `world`
- **サブワード単位**
  - 例: `baseball` → `base`, `ball`

本記事ではGPT系列で採用され、サブワード単位のトークナイザーであるBPEを実装します。

# BPE (Byte Pair Encoding)

BPEは頻出するバイト列のペアを繰り返しマージして語彙を構築するアルゴリズムです。

## アルゴリズム

学習データとして `abababcb` があるとします。

**Step 0: 初期状態**

トークン列: `a b a b a b c b`

語彙: `a`, `b`, `c`

ペア出現回数: `(a, b) = 3`, `(b, a) = 2`, `(b, c) = 1`, `(c, b) = 1`

**Step 1: 最頻ペア (a, b) をマージ → 新トークン `ab`**

トークン列: `ab ab ab c b`

語彙: `a`, `b`, `c`, `ab`

ペア出現回数: `(ab, ab) = 2`, `(ab, c) = 1`, `(c, b) = 1`

マージルール: `(a, b) → ab`

**Step 2: 最頻ペア (ab, ab) をマージ → 新トークン `abab`**

トークン列: `abab ab c b`

語彙: `a`, `b`, `c`, `ab`, `abab`

ペア出現回数: `(ab, c) = 1`, `(c, b) = 1`

マージルール: `(a, b) → ab`, `(ab, ab) → abab`

**Step 3: 最頻ペア (c, b) をマージ → 新トークン `cb`**

トークン列: `abab ab cb`

語彙: `a`, `b`, `c`, `ab`, `abab`, `cb`

マージルール: `(a, b) → ab`, `(ab, ab) → abab`, `(c, b) → cb`

このように、頻出するパターンが段階的に1つのトークンにまとまっていきます。これを目標の語彙サイズに達するまで繰り返します。

## 処理フロー

```
学習時: テキスト → バイト列 → 最頻ペアをマージ（繰り返し）→ 語彙完成
推論時: テキスト → バイト列 → 学習済みマージを適用 → トークンID列
```

# 完成形

```python
from tokenizer.bpe import BPE

# 学習
texts = ["ab", "abc", "abcd"]
tokenizer = BPE()
tokenizer.train(texts, vocab_size=300)

# エンコード（学習データ）
tokens = tokenizer.encode("ab")
print(tokens)  # [260]

# エンコード（学習データにない文字列）
tokens = tokenizer.encode("abcde")
print(tokens)  # [262, 105] → abcd + e

# デコード
text = tokenizer.decode(tokens)
print(text)  # abcde
```

コード全体: [GitHub](https://github.com/so-dev000/llm-from-scratch/blob/main/tokenizer/bpe.py)

# 実装

## 特殊トークン

学習データには現れないが、モデルが必要とする特殊なトークンを定義します。

```python
DEFAULT_SPECIAL_TOKENS = {
    "<PAD>": 0,  # パディング（系列長を揃えるため）
    "<UNK>": 1,  # 未知語（語彙外のトークン）
    "<BOS>": 2,  # 文頭（Begin Of Sentence）
    "<EOS>": 3,  # 文末（End Of Sentence）
}
```

## Pre-tokenize: 正規表現による分割

BPEを適用する前に、正規表現を用いてテキストを単語単位で分割します。
ここではGPT-2と同じパターンを使用します。

```python
GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
```

これによって以下のように分割されます:

- 入力: `"Hello, world! I'm here."`
- 出力: `["Hello", ",", " world", "!", " I", "'m", " here", "."]`

## 学習

```python
def train(self, texts, vocab_size):
    num_special = len(self.special_tokens)  # 4個: <PAD>, <UNK>, <BOS>, <EOS>

    # 語彙の初期化
    # ID 0-3: 特殊トークン
    for token, idx in self.special_tokens.items():
        self.vocab[idx] = token.encode("utf-8")

    # ID 4-259: バイトトークン（256種類）
    for i in range(256):
        self.vocab[num_special + i] = bytes([i])

    # Pre-tokenize: 正規表現による分割 → バイト列に変換
    all_ids = []
    for text in texts:
        chunks = self.compiled_pattern.findall(text)
        for chunk in chunks:
            byte_vals = list(chunk.encode("utf-8"))
            # バイト値を語彙IDに変換（特殊トークン分をオフセット）
            chunk_ids = [num_special + b for b in byte_vals]
            all_ids.append(chunk_ids)

    # 隣接ペアをカウント
    pair_counts = Counter()
    for chunk in all_ids:
        pair_counts.update(zip(chunk, chunk[1:]))

    # マージループ
    num_merges = vocab_size - 256 - num_special
    for i in tqdm(range(num_merges), desc="Training BPE"):
        if not pair_counts:
            break
        pair = max(pair_counts, key=pair_counts.get)  # 最頻ペアを取得
        new_idx = num_special + 256 + i  # ID 260以降が新しいマージトークン

        self.merges[pair] = new_idx
        self.vocab[new_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        # マージを適用して、ペアのカウントを更新
        self._apply_merge_and_update_pair_counts(
            all_ids, pair, new_idx, pair_counts
        )
```

処理の流れ:

1. 語彙を初期化（特殊トークン4個 + 256バイト = 260個）
2. テキストをPre-tokenize → バイト列に変換
3. 隣接ペアをカウント
4. マージループ(vocab_sizeに達するまで繰り返し)
   - 最頻ペアを取得
   - 新しいトークンとして語彙に追加
   - マージを適用してペアカウントを更新

### マージ適用

ナイーブな実装では、マージのたびに全ペアを再カウントするため、学習データが大きいと訓練時間がとんでもないことになります(100万件のデータセットで約100時間)。そこでマージの影響を受けるペアのみを更新することで、計算量を削減しています(これでも4-5時間かかるので、第3回で2万件のデータセットを使って動作検証した後は普通にHugging Faceのtokenizer使います...)。

```python
def _apply_merge_and_update_pair_counts(self, chunks, pair, new_idx, pair_counts):
    """
    マージの影響を受けるペアのみを更新する

    例: (A, B) → X にマージする場合

    マージ前: ...P A B Q...
    マージ後: ...P X Q...

    影響を受けるペア:
    - 減少: (P, A), (A, B), (B, Q)
    - 増加: (P, X), (X, Q)
    """
    p0, p1 = pair
    del pair_counts[pair]  # マージ済みペアを削除

    for chunk in chunks:
        i = 0
        while i < len(chunk) - 1:
            if chunk[i] == p0 and chunk[i + 1] == p1:
                # 左隣のペアを更新
                if i > 0:
                    pair_counts[(chunk[i - 1], p0)] -= 1
                # 右隣のペアを更新
                if i + 2 < len(chunk):
                    pair_counts[(p1, chunk[i + 2])] -= 1

                # マージ実行
                chunk[i : i + 2] = [new_idx]

                # 新しいペアを追加
                if i > 0:
                    pair_counts[(chunk[i - 1], new_idx)] += 1
                if i + 1 < len(chunk):
                    pair_counts[(new_idx, chunk[i + 1])] += 1
            else:
                i += 1

    # カウント0以下のペアを削除
    pair_counts += Counter()
```

## エンコード

```python
def encode(self, text, add_special_tokens=False):
    num_special = len(self.special_tokens)

    # Pre-tokenize: 正規表現による分割
    chunks = self.compiled_pattern.findall(text)
    tokens = []

    if add_special_tokens:
        tokens.append(self.special_tokens["<BOS>"])

    for chunk in chunks:
        # バイト列に変換
        byte_vals = list(chunk.encode("utf-8"))
        chunk_tokens = [num_special + b for b in byte_vals]

        # 学習済みマージを優先度順に適用
        while len(chunk_tokens) >= 2:
            # 現在のトークン列に存在する全ペアを取得
            stats = set(zip(chunk_tokens, chunk_tokens[1:]))
            # 最も優先度の高い（早くマージされた）ペアを選択
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # マージ可能なペアがない
            idx = self.merges[pair]
            chunk_tokens = self._merge(chunk_tokens, pair, idx)
        tokens.extend(chunk_tokens)

    if add_special_tokens:
        tokens.append(self.special_tokens["<EOS>"])

    return tokens
```

学習時と同様にバイト列に変換し、学習済みのマージを優先度順に適用します。

## デコード

```python
def decode(self, ids, skip_special_tokens=True):
    # 特殊トークンを除外
    if skip_special_tokens:
        special_ids = set(self.special_tokens.values())
        ids = [i for i in ids if i not in special_ids]

    # 語彙からバイト列を取得して結合
    tokens = b"".join(self.vocab[i] for i in ids)

    # 不正なUTF-8バイト列を � (U+FFFD) に置換してデコード
    text = tokens.decode("utf-8", errors="replace")
    return text
```

vocabからバイト列を取得して結合し、UTF-8としてデコードします。

# まとめ

本記事ではGPT系列で採用されているBPEトークナイザーを実装しました。
Part2ではTransformerの実装について解説しています。

https://qiita.com/so_dev000/items/d587d3641c7977c6e3e7

# 参考

https://huggingface.co/learn/llm-course/chapter6/4

https://huggingface.co/learn/llm-course/chapter6/5

https://www.youtube.com/watch?v=zduSFxRajkE

https://tiktokenizer.vercel.app/?model=gpt2
