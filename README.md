# Single-Bit Quantization of Transformers/BERT

Eight progressively refined experiments in single-bit and ternary weight quantization for sentiment
classification on SST-2, inspired by BitNet b1.58. Scratch-built 1-bit transformers top out at
76.38% accuracy; replacing the dense layers of a pre-trained `bert-base-uncased` with sign-only
weights trained through a straight-through estimator (STE) reaches **85.67% accuracy and 85.65% F1**.

## Why

BitNet b1.58 showed that language models with ternary weights can match full-precision quality at
scale. This project asks the small-scale version of that question: on a standard classification
task, which training tricks keep a 1-bit model useful, and does pre-training survive binarization.

## How it works

Every linear layer is replaced by a `BitLinear` layer whose weights are reduced to their sign (or to
{-1, 0, +1} in the ternary variant). Each approach keeps the previous setup and changes one thing,
so the table reads as an ablation:

- Approaches 1 to 3: a scratch transformer encoder with 1-bit weights, then positional encoding and
  mixed precision, then dropout and quantization-aware training.
- Approaches 4 and 5: median scaling of activations plus STE, so gradients flow through the sign
  function, and a tuned variant.
- Approach 6: full multi-head attention with progressive precision lowering (8, 4, then 1 bit).
- Approaches 7 and 8: pre-trained BERT with 1-bit weights and STE, then ternary weights with
  activation quantization.

```mermaid
flowchart LR
    A[Scratch 1-bit transformer] --> B[+ positional encoding, AMP]
    B --> C[+ dropout, QAT]
    C --> D[+ median scaling, STE]
    D --> E[+ multi-head attention, progressive MoQ]
    E --> F[Pre-trained BERT, 1-bit + STE]
    F --> G[Ternary BERT + activation quant]
```

## Evaluation and results

Models are trained on the GLUE SST-2 train split and scored on its validation split. Every approach
was trained for 3 epochs.

| # | Model / technique | Accuracy | F1 |
|:-:|---|:-:|:-:|
| 1 | Scratch transformer, 1-bit weights | 76.38 | 76.38 |
| 2 | + positional encoding and AMP | 62.27 | 61.26 |
| 3 | + dropout and QAT | 63.88 | 63.65 |
| 4 | + median scaling and STE | 69.84 | 69.65 |
| 5 | Tuned variant of 4 | 70.76 | 70.66 |
| 6 | + multi-head attention and progressive MoQ | 70.18 | 70.15 |
| 7 | **BERT-base, 1-bit weights + STE** | **85.67** | **85.65** |
| 8 | Ternary BERT + activation quantization | 50.92 | 34.36 |

Pre-training is the largest single factor: the same 1-bit treatment applied to BERT beats every
scratch model by about 15 points. Mixed precision destabilized sign-only training (approach 2), and
quantizing activations as well as weights collapsed the model to near chance (approach 8).

## Limitations

- Each approach was trained for only 3 epochs; longer schedules may change the ranking.
- One run per approach, so differences of a point or two are within run-to-run noise.
- Weights are binarized in simulation; the project measures accuracy, not memory or speed on real
  low-bit kernels.
- A single task (SST-2 sentiment); results may not transfer to generation or harder benchmarks.

## License

Code and results are released under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).

## Citation

```bibtex
@misc{chawla2025singlebit,
  author       = {Naman Chawla},
  title        = {Single-Bit Quantization of Transformers/BERT},
  year         = {2025},
  howpublished = {\url{https://github.com/palindromeRice/SingleBitLLMs}}
}
```

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).
