# YorubaLiteSent: Lightweight Sentiment Analysis for Low-Resource Yoruba

## Research motivation

YOSM has only 800 labelled Yoruba training reviews. The existing TextCNN and
CNN-BiLSTM-Attention models are compact, but they learn most word information
from this small labelled set. The initial MENYO experiment enlarged the
vocabulary but left the added embedding rows randomly initialized; it reduced
OOV rate but did not transfer semantic knowledge or improve performance.

Large pretrained transformers can improve low-resource NLP, but they are costly
to deploy. AfroXLM-R base has roughly 300 million parameters, while the pure
YOSM TextCNN has 548 thousand parameters. LoRA reduces the parameters updated
during training, but the complete transformer still has to be loaded at
inference. Therefore, a LoRA transformer alone is not a lightweight deployment
solution.

## Research question

> Can a compact Yoruba CNN-BiLSTM-Attention model, initialized with MENYO
> subword embeddings and trained using optional transformer knowledge
> distillation, achieve a competitive accuracy-efficiency trade-off for movie
> sentiment classification?

## Proposed system

```text
MENYO Yoruba training text
        ↓
FastText subword embedding pretraining
        ↓
Pure YOSM training vocabulary
        ↓
Compact CNN-BiLSTM-Attention student
        ↑
Optional AfroXLM-R teacher probabilities during training only
        ↓
Small student model for deployment
```

### Why this is genuine transfer learning

FastText learns vectors from character subwords across MENYO Yoruba text. It can
provide useful initial vectors for YOSM words and better handle word variants,
without adding random embedding rows for unseen vocabulary. The student keeps a
pure YOSM vocabulary, so its embedding table remains small.

The transformer, if used, acts only as an offline teacher. Its soft confidence
scores guide the student during training. The deployed model is still the small
student, not AfroXLM-R.

## Controlled experiments

| System | MENYO knowledge | Transformer at deployment | Purpose |
|---|---|---:|---|
| TextCNN | None | No | Existing baseline |
| CNN-BiLSTM-Attention | None | No | Existing strong baseline |
| FastText student | MENYO subword embeddings | No | Tests real embedding transfer |
| YorubaLiteSent | MENYO embeddings + teacher soft labels | No | Tests embedding transfer and distillation |
| AfroXLM-R teacher | Pretrained African-language transformer | Yes, teacher only | Accuracy upper reference |

Run each student system with seeds `13, 21, 42, 87, 101`. Report mean and
standard deviation for accuracy, precision, recall, and F1. Also report model
parameters, saved checkpoint size, and CPU batch-1 latency.

## Runbook

```bash
pip install -r requirements-research.txt

# A. Compact student with MENYO FastText vectors, no transformer.
python research/lightweight_student.py --seed 42 --benchmark

# B. Fine-tune an offline AfroXLM-R teacher. This is not deployed.
python research/yoruba_transformer.py finetune \
  --checkpoint Davlan/afro-xlmr-base \
  --seed 42

# C. Distilled compact student. Use the train probabilities exported in step B.
python research/lightweight_student.py \
  --teacher_probs research_outputs/yosm_transformer/lora_seed_42/teacher_train_probabilities.npz \
  --seed 42 --benchmark

# D. Repeat student training with five seeds, then aggregate the metrics.
for seed in 13 21 42 87 101; do
  python research/lightweight_student.py \
    --teacher_probs research_outputs/yosm_transformer/lora_seed_42/teacher_train_probabilities.npz \
    --seed "$seed" --benchmark
done
python research/aggregate_runs.py --root research_outputs/yorubalitesent
```

The teacher may require a 16 GB GPU. The final student is much smaller and can
be trained and deployed on substantially lighter hardware. Never use YOSM test
text to pretrain FastText, choose checkpoints, or generate student training
targets.

## Project contribution statement

> YorubaLiteSent investigates whether subword-aware auxiliary Yoruba embeddings
> and transformer knowledge distillation can improve a compact sentiment student
> without deploying a large transformer. It evaluates the accuracy-efficiency
> trade-off on the official YOSM benchmark through controlled ablations and
> repeated-seed reporting.

## References

- Shode, Adelani, and Feldman (2022). [YOSM: A New Yoruba Sentiment Corpus for
  Movie Reviews](https://arxiv.org/abs/2204.09711).
- Shode et al. (2023). [NollySenti: Leveraging Transfer Learning and Machine
  Translation for Nigerian Movie Sentiment Classification](https://aclanthology.org/2023.acl-short.85/).
- Bojanowski et al. (2017). [Enriching Word Vectors with Subword
  Information](https://aclanthology.org/Q17-1010/).
- Hinton, Vinyals, and Dean (2015). [Distilling the Knowledge in a Neural
  Network](https://arxiv.org/abs/1503.02531).
- Alabi et al. (2022). [Adapting Pre-trained Language Models to African
  Languages via Multilingual Adaptive Fine-Tuning](https://huggingface.co/Davlan/afro-xlmr-base).
