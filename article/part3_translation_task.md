# はじめに

Part2ではTransformerモデルを実装しました。
今回は実装したTransformerを使って英日翻訳タスクを解いていきます。

https://qiita.com/so_dev000/items/d587d3641c7977c6e3e7

# 今回やること

- データセット準備とトークナイザー訓練
- 訓練パイプラインの構築
- モデルの訓練
- 推論（Beam Search、Greedy Decoding）

# PyTorch Lightning について

本プロジェクトでは訓練ループの実装に PyTorch Lightning を使用しています。

## PyTorch Lightning とは

PyTorchで書くのがやや億劫な訓練ループの抽象化をいい感じに行ってくれるライブラリです。ここでは詳しい説明は割愛しますが、以下のページを見ると雰囲気は掴めると思います。最初は訓練ループを手書きしてましたが、途中でこのライブラリの存在を知ったのでClaude Codeに書き換えてもらいました。若干変更コストは高いのでできれば最初から導入したかったです。

https://lightning.ai/docs/pytorch/stable/starter/introduction.html

# プロジェクト構成

```
scripts/
  config.py          # 設定管理
  prepare.py         # トークナイザー訓練
  train.py           # モデル訓練
  eval.py            # モデル評価
  inference.py       # 推論
  lightning_module.py # PyTorch Lightningモジュール

model/
  transformer.py     # Transformerモデル

utils/
  training_pipeline.py  # 訓練パイプライン
  inference_pipeline.py # 推論パイプライン
  decoding_strategy.py  # デコード戦略
  collate.py # collate関数
```

コード全体: [GitHub](https://github.com/so-dev000/llm-from-scratch)

# 1. データセット準備とトークナイザー訓練

## データセット

計算資源に限りがあったため、小規模な英日対訳データセットを使用して学習を行いました。

https://huggingface.co/datasets/ryo0634/bsd_ja_en

## トークナイザー訓練

```python
def prepare_transformer_tokenizers(
    dataset_name,
    vocab_size,
    src_lang,
    tgt_lang,
    src_column,
    tgt_column,
):
    from datasets import load_dataset
    from tokenizer.bpe import BPE

    dataset_dir = dataset_name.replace("/", "_")
    tokenizer_dir = f"data/tokenizers/{dataset_dir}"

    src_tokenizer_path = f"{tokenizer_dir}/{src_lang}_bpe.pkl"
    tgt_tokenizer_path = f"{tokenizer_dir}/{tgt_lang}_bpe.pkl"

    dataset = load_dataset(dataset_name, split="train")

    # 英語トークナイザーを訓練
    src_texts = [ex[src_column] for ex in dataset]
    src_tokenizer = BPE(pattern=GPT2_PATTERN if src_lang == "en" else None)
    src_tokenizer.train(src_texts, vocab_size=vocab_size)

    # 日本語トークナイザーを訓練
    tgt_texts = [ex[tgt_column] for ex in dataset]
    tgt_tokenizer = BPE(pattern=None)
    tgt_tokenizer.train(tgt_texts, vocab_size=vocab_size)

    # 保存
    os.makedirs(tokenizer_dir, exist_ok=True)
    src_tokenizer.save(src_tokenizer_path)
    tgt_tokenizer.save(tgt_tokenizer_path)

    return tokenizer_dir
```

# 2. 訓練パイプラインの構築

## DataModule

データセットの事前処理を行なうモジュールです。

```python
class TransformerDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenized_datasets = None
        self.src_tokenizer = None
        self.tgt_tokenizer = None

    def setup(self, stage: Optional[str] = None):
        # トークナイザーをロード
        dataset_dir = self.config.data.dataset_name.replace("/", "_")
        tokenizer_dir = f"{self.config.tokenizer_dir}/{dataset_dir}"

        self.src_tokenizer = BPE.load(f"{tokenizer_dir}/{self.config.data.src_lang}_bpe.pkl")
        self.tgt_tokenizer = BPE.load(f"{tokenizer_dir}/{self.config.data.tgt_lang}_bpe.pkl")

        # 語彙サイズをconfigに設定
        self.config.model.src_vocab_size = self.src_tokenizer.get_vocab_size()
        self.config.model.tgt_vocab_size = self.tgt_tokenizer.get_vocab_size()

        # データセットをロードして分割
        dataset = load_dataset(self.config.data.dataset_name, split="train")
        train_rest = dataset.train_test_split(test_size=0.15, seed=42)
        val_test = train_rest["test"].train_test_split(test_size=0.33, seed=42)
        train_val = DatasetDict({
            "train": train_rest["train"],
            "val": val_test["train"]
        })

        # トークン化
        def preprocess_batch(batch):
            src_ids = []
            for text in batch[self.config.data.src_column]:
                ids = self.src_tokenizer.encode(text, add_special_tokens=True)
                if len(ids) > self.config.data.max_length:
                    ids = ids[:self.config.data.max_length - 1] + [
                        self.src_tokenizer.special_tokens["<EOS>"]
                    ]
                src_ids.append(ids)

            tgt_ids = []
            for text in batch[self.config.data.tgt_column]:
                ids = self.tgt_tokenizer.encode(text, add_special_tokens=True)
                if len(ids) > self.config.data.max_length:
                    ids = ids[:self.config.data.max_length - 1] + [
                        self.tgt_tokenizer.special_tokens["<EOS>"]
                    ]
                tgt_ids.append(ids)

            return {"src": src_ids, "tgt": tgt_ids}

        self.tokenized_datasets = train_val.map(
            preprocess_batch,
            batched=True,
            batch_size=1000,
            remove_columns=train_val["train"].column_names,
            desc="Tokenizing dataset",
        )

        self.tokenized_datasets.set_format(type="torch", columns=["src", "tgt"])

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["train"],
            batch_size=self.config.data.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.config.data.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_datasets["val"],
            batch_size=self.config.data.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=self.config.data.num_workers,
        )
```

## Collate関数

可変長の文字列をバッチ化する際、短い文字列に `<PAD>` を追加して長さを揃えます。

```python
from torch.nn.utils.rnn import pad_sequence

def collate(batch, pad_id=0):
    src_tensors = [item["src"] for item in batch]
    tgt_tensors = [item["tgt"] for item in batch]

    # パディング
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_id)

    # マスクを生成
    src_mask = src_padded != pad_id
    tgt_mask = tgt_padded != pad_id

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }
```

collate関数の処理の例:

```
入力:
  src: [[1, 2, 3], [4, 5]]
  tgt: [[6, 7], [8, 9, 10]]

出力:
  src: [[1, 2, 3], [4, 5, 0]]  # 0: <PAD>
  tgt: [[6, 7, 0], [8, 9, 10]]
  src_mask: [[True, True, True], [True, True, False]]
  tgt_mask: [[True, True, False], [True, True, True]]
```

## LightningModule

訓練ループをPyTorch Lightningで抽象化したモジュールです。

```python
class TransformerLightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        self.model = Transformer(config.model)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.data.pad_idx,
            label_smoothing=config.training.label_smoothing,
        )

    def _shared_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        src_mask = batch["src_mask"]
        tgt_padding_mask = batch["tgt_mask"]

        # Encoder用のマスク（padding mask）
        src_mask_expanded = src_mask.unsqueeze(1) & src_mask.unsqueeze(2)

        # 入力と出力を1トークンずらす
        tgt_input = tgt[:, :-1]   # <BOS> I like sushi
        tgt_output = tgt[:, 1:]   # I like sushi <EOS>

        # Decoder用のマスク（causal mask + padding mask）
        tgt_len = tgt_input.size(1)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        tgt_padding = tgt_padding_mask[:, :-1]
        tgt_input_mask = (
            causal_mask.unsqueeze(0)
            & tgt_padding.unsqueeze(1)
            & tgt_padding.unsqueeze(2)
        )

        # Forward
        output = self.model(
            src,
            tgt_input,
            encoder_src_mask=src_mask_expanded,
            decoder_src_mask=src_mask,
            tgt_mask=tgt_input_mask,
        )

        # Loss計算
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        loss = self.criterion(output, tgt_output)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.optimizer.initial_lr,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )

        # 学習率スケジューラ
        def lr_lambda(step):
            step = max(step, 1)
            model_dim = self.config.model.model_dim
            warmup_steps = self.config.optimizer.warmup_steps
            return (model_dim ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
```

### Optimizer と Scheduler

**Attention Is All You Need**の論文で提案されている学習率スケジューラを実装しています。

$$\text{lr} = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

```python
def lr_lambda(step):
    step = max(step, 1)
    model_dim = config.model.model_dim
    warmup_steps = config.optimizer.warmup_steps
    return (model_dim ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

# 3. モデルの訓練

## 設定

Configクラスでハイパーパラメータを一元管理しています。

### Config構造

```python
@dataclass
class Config:
    data: DataConfig           # データセットの設定
    optimizer: OptimizerConfig # Optimizerの設定
    model: TransformerModelConfig # Transformerモデルの構造設定
    training: TrainingConfig   # 訓練設定
    inference: InferenceConfig # 推論設定
    modal: ModalConfig         # Modal実行環境の設定
    wandb: WandbConfig         # Weights & Biases の設定

    run_name: str = None
    checkpoint_dir: str = "/vol/runs"
    tokenizer_dir: str = "/vol/tokenizers"
```

### 各設定の詳細

#### DataConfig - データセットの設定

```python
@dataclass
class DataConfig:
    dataset_name: str       # データセット名
    batch_size: int         # バッチサイズ
    max_length: int         # 最大系列長
    num_workers: int
    pad_idx: int
    vocab_size: int = 8000  # 語彙サイズ
    src_lang: str = "en"    # 翻訳元言語
    tgt_lang: str = "ja"    # 翻訳先言語
    src_column: str = "en_sentence"  # データセットの入力カラム名
    tgt_column: str = "ja_sentence"  # データセットの出力カラム名
```

#### OptimizerConfig - Optimizerの設定

```python
@dataclass
class OptimizerConfig:
    warmup_steps: int
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    initial_lr: float = 1.0  # 学習率の初期値
    optimizer_type: str = "Adam"
    scheduler_type: str = "inverse_sqrt"  # 学習率スケジューラのタイプ
```

#### TransformerModelConfig - Transformerモデルの構造設定

```python
@dataclass
class TransformerModelConfig:
    model_type: Literal["transformer"] = "transformer"
    src_vocab_size: int = None  # 入力語彙サイズ（訓練時に自動設定）
    tgt_vocab_size: int = None  # 出力語彙サイズ（訓練時に自動設定）
    model_dim: int = 256        # モデルの次元数
    encoder_layers: int = 4     # Encoderの層数
    decoder_layers: int = 4     # Decoderの層数
    num_heads: int = 8          # AttentionのHead数
    feedforward_dim: int = None # FFNの中間層サイズ
    dropout: float = 0.1        # Dropout率
    activation: str = "relu"    # 活性化関数
    max_seq_len: int = 5000
    padding_idx: int = 0
```

論文よりもパラメータ数を減らしています。

#### TrainingConfig - 訓練設定

```python
@dataclass
class TrainingConfig:
    num_epochs: int
    label_smoothing: float
    early_stopping_patience: int
    gradient_clip_val: float = 1.0
    precision: str = "32-true"
    accumulate_grad_batches: int = 1
    val_check_interval: float = 1.0
```

#### InferenceConfig - 推論設定

```python
@dataclass
class InferenceConfig:
    # 特殊トークン
    pad_idx: int = 0
    unk_idx: int = 1
    bos_idx: int = 2
    eos_idx: int = 3

    # Beam Search設定
    beam_size: int = 5
    length_penalty: float = 0.6
    max_output_offset: int = 10

    max_gen_len: int = 100
```

### ModalConfig - Modal実行環境の設定

```python
@dataclass
class ModalConfig:
    gpu_type: str = "L40S"
    timeout_hours: int = 12
    volume_name: str = "XXX"
    secret_name: str = "XXX"
```

訓練にはModal.comのGPUを使用しています。

https://modal.com/

### WandbConfig - Weights & Biases の設定

```python
@dataclass
class WandbConfig:
    project: str = "XXX"
    entity: str = None
    enabled: bool = True
    log_model: bool = False
```

学習曲線のモニタリングにWeights & Biasesを使用しています。
GPUのメトリクスもモニタリングでき、スマホからでも進捗を見ることができるのでとても便利です。学生だと無料でライセンス認証できます。

https://wandb.ai/

## 訓練実行

まず、CPUインスタンス上でデータセットを事前処理してキャッシュに保存します。
料金が高いGPUインスタンスの使用時間を少しでも減らすための工夫です。

```python
@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=1800,
)
def prepare_dataset(config: Config):
    dataset_dir = config.data.dataset_name.replace("/", "_")
    cache_path = f"/vol/processed/{dataset_dir}"

    if os.path.exists(cache_path):
        print(f"Data already prepared at {cache_path}")
        return

    data_module = get_data_module(config)
    data_module.prepare_data()
    data_module.setup(stage="fit")
    volume.commit()
```

次に、訓練を実行します:

```python
@app.function(
    image=image,
    gpu="L40S",
    volumes={"/vol": volume},
    timeout=3600 * 12,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(config: Config):
    torch.set_float32_matmul_precision("high")

    # キャッシュからデータを読み込み
    data_module = get_data_module(config)
    data_module.load_from_cache()

    pl_module = TransformerLightningModule(config)

    # Callbacks設定
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.training.early_stopping_patience,
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=f"{config.checkpoint_dir}/{config.run_name}",
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]

    # Logger設定
    logger = WandbLogger(
        project=config.wandb.project,
        name=config.run_name,
    )

    # Trainer設定
    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.training.gradient_clip_val,
        precision=config.training.precision,
    )

    # 訓練開始
    trainer.fit(pl_module, datamodule=data_module)

    volume.commit()
```

以下のような学習曲線を描きました。検証データに対する損失が5.0あたりまで下がった後再び上昇し、過学習を起こしてしまっているのがわかります。

![スクリーンショット 2025-12-21 22.40.47.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4069100/5f71824d-e11b-4e84-afea-580ef942be66.png)

# 4. 推論

## デコード戦略

翻訳時に出力文を生成するロジックを書いていきます。以下に挙げる記事を参考に実装しています。

https://huggingface.co/blog/mlabonne/decoding-strategies

https://huggingface.co/blog/how-to-generate

### Greedy Decoding

各ステップで最も確率が高いトークンを選択します。

```python
class GreedyDecoding(DecodingStrategy):
    def decode(self, model, src_tokens, src_mask, max_len):
        device = src_tokens.device
        batch_size = src_tokens.size(0)
        context = model.prepare_context(src_tokens, src_mask)
        bos_idx = self.config.bos_idx
        eos_idx = self.config.eos_idx
        results = []

        for batch_idx in range(batch_size):
            output_tokens = [bos_idx]
            for _ in range(max_len):
                tgt_input = torch.tensor([output_tokens], device=device)
                context_subset = {
                    k: v[batch_idx:batch_idx + 1] for k, v in context.items()
                }
                logits = model.generate_next_token(tgt_input, context_subset)
                # 確率が最大のトークン
                next_token = logits.argmax(dim=-1).item()
                output_tokens.append(next_token)
                if next_token == eos_idx:
                    break
            results.append(torch.tensor(output_tokens, device=device))
        return results
```

### Beam Search

複数の候補（ビーム）を保持しながら探索します。

```python
class BeamSearch(DecodingStrategy):
    def decode(self, model, src_tokens, src_mask, max_len):
        device = src_tokens.device
        batch_size = src_tokens.size(0)
        beam_size = self.config.beam_size
        context = model.prepare_context(src_tokens, src_mask)
        bos_idx = self.config.bos_idx
        eos_idx = self.config.eos_idx

        # 各バッチに対してビームを初期化
        beams = [
            [(torch.tensor([bos_idx], device=device), 0.0)]
            for _ in range(batch_size)
        ]

        for _ in range(max_len):
            all_candidates = []
            for batch_idx in range(batch_size):
                candidates = []
                for seq, score in beams[batch_idx]:
                    if seq[-1].item() == eos_idx:
                        candidates.append((seq, score))
                        continue

                    tgt_input = seq.unsqueeze(0)
                    context_subset = {
                        k: v[batch_idx:batch_idx + 1] for k, v in context.items()
                    }
                    logits = model.generate_next_token(tgt_input, context_subset)
                    log_probs = F.log_softmax(logits, dim=-1)
                    top_log_probs, top_indices = log_probs.topk(beam_size)

                    for k in range(beam_size):
                        new_seq = torch.cat([seq, top_indices[0, k].unsqueeze(0)])
                        new_score = score + top_log_probs[0, k].item()
                        candidates.append((new_seq, new_score))

                # 長さペナルティを適用してソート
                candidates = sorted(
                    candidates,
                    key=lambda x: x[1] / ((5 + len(x[0])) / 6) ** self.config.length_penalty,
                    reverse=True,
                )[:beam_size]
                all_candidates.append(candidates)

            beams = all_candidates

            # 全ビームが終了したらループを抜ける
            if all(all(seq[-1].item() == eos_idx for seq, _ in beam) for beam in beams):
                break

        # 各バッチで最も高評価のシーケンスを返す
        results = []
        for beam in beams:
            best_seq, _ = beam[0]
            results.append(best_seq)
        return results
```

## 推論パイプライン

```python
def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, config, strategy="beam"):
    results = translate_batch(
        model, [sentence], src_tokenizer, tgt_tokenizer, config, strategy
    )
    return results[0]

def translate_batch(model, sentences, src_tokenizer, tgt_tokenizer, config, strategy="beam"):
    model.eval()
    device = next(model.parameters()).device

    # 入力をトークン化
    src_ids = []
    for sentence in sentences:
        ids = src_tokenizer.encode(sentence, add_special_tokens=True)
        if len(ids) > config.data.max_length:
            ids = ids[:config.data.max_length - 1] + [
                src_tokenizer.special_tokens["<EOS>"]
            ]
        src_ids.append(ids)

    max_src_len = max(len(ids) for ids in src_ids)
    src_tokens = torch.zeros(len(sentences), max_src_len, dtype=torch.long)
    src_mask = torch.zeros(len(sentences), max_src_len, dtype=torch.bool)

    for i, ids in enumerate(src_ids):
        src_tokens[i, :len(ids)] = torch.tensor(ids)
        src_mask[i, :len(ids)] = True

    src_tokens = src_tokens.to(device)
    src_mask = src_mask.to(device)

    max_output_len = max_src_len + config.inference.max_output_offset

    # デコード戦略を選択
    if strategy == "beam":
        decoder = BeamSearch(config.inference)
    elif strategy == "greedy":
        decoder = GreedyDecoding(config.inference)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 推論実行
    with torch.no_grad():
        output_seqs = decoder.decode(model, src_tokens, src_mask, max_output_len)

    # トークンIDをテキストに変換
    translations = []
    for seq in output_seqs:
        tokens = seq.tolist()
        if config.inference.bos_idx in tokens:
            tokens = tokens[1:]
        if config.inference.eos_idx in tokens:
            eos_pos = tokens.index(config.inference.eos_idx)
            tokens = tokens[:eos_pos]

        translation = tgt_tokenizer.decode(tokens, skip_special_tokens=True)
        translations.append(translation)

    return translations
```

## 翻訳結果

以下のCSVにテストデータ（訓練にも検証にも使用していない初見のデータ）に対して翻訳を行った結果をまとめています。デコード戦略にはBeam Searchを使用しています。`ref`が正解ラベル、`hyp`が翻訳結果です。
短文は割と頑張っていますが、少し長い文になると支離滅裂な翻訳になってしまいました。そもそもデータセットの規模が2万件と非常に小さく、モデルのパラメータ数を削ったりしているので妥当な結果ではありますが、Transformerの雰囲気を掴むにはいいタスクだったと思います。

https://github.com/so-dev000/llm-from-scratch/blob/main/public/bsd_translation_eval_results.csv

# まとめ

本記事では、実装したTransformerモデルを使って英日翻訳タスクを解きました。
Part4ではいよいよGPT-2の実装をしていきます。

# ゼロから作るLLM - 目次

https://qiita.com/so_dev000/items/8082a34582e97ebe0815
