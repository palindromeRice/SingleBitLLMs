# 🧠 Single-Bit Quantization in NLP: Experiments & Insights

## 📌 Overview
Microsoft’s **BitNet b1.58** — a *1.58-bit* large-language model (LLM) capable of running on commodity CPUs — reignited interest in ultra-low-precision inference.  
Inspired by this work, I explored **single-bit (and ternary) quantization** on the SST-2 sentiment-analysis task.  
This repo walks through **eight progressively refined approaches**, starting from scratch-built transformers and culminating in a quantized, fine-tuned BERT.

---

## 🧪 Experimentation Strategy
1. **Baseline** – scratch-built transformer + 1-bit weights.  
2. **Incremental tricks** – add positional encoding, dropout, mixed precision, QAT.  
3. **Advanced tricks** – median-scaling, Straight-Through Estimator (STE), progressive Mixed-precision Quantization (MoQ).  
4. **Pre-trained models** – swap in BERT, then apply STE / ternary + activation quantization.

At each stage I addressed shortcomings of the previous approach while monitoring accuracy/F1, model size, and training stability.

---

## 🔍 Detailed Approaches

<details>
<summary><strong>Approach 1 – Simple Quantized Transformer (Classifier)</strong></summary>

* **Goal** – prove 1-bit feasibility.  
* **Key steps**
  * Scratch implementation of a miniature Transformer encoder.
  * Replaced all linear layers with custom <code>BitLinear</code> (sign-only weights).
  * Adam + CE loss; no fancy schedulers.
* **Results** – <br>Accuracy 76.38 % | F1 76.38 %  
* **Takeaway** – works, but capacity is tiny and no positional clues → limited ceiling.
</details>

<details>
<summary><strong>Approach 2 – + Positional Encoding & Mixed Precision</strong></summary>

* Added sinusoidal PE, automatic mixed precision (AMP), scheduler + grad-clip.  
* **Results** – 62.27 % / 61.26 %.  
* **Why worse?** AMP introduced instability with sign-only weights; capacity still low.
</details>

<details>
<summary><strong>Approach 3 – + Dropout & Quantization-Aware Training (QAT)</strong></summary>

* Injected dropout; trained with fake-quant ops (PyTorch QAT).  
* **Results** – 63.88 % / 63.65 %.  
* **Takeaway** – tiny bump; still under-fits.
</details>

<details>
<summary><strong>Approach 4 – Median Scaling + Straight-Through Estimator (STE)</strong></summary>

* Normalised activations via median scaling; back-prop with STE.  
* **Results** – 69.84 % / 69.65 %.  
* **Takeaway** – big jump → scaling + STE help gradients flow in 1-bit nets.
</details>

<details>
<summary><strong>Approach 5 – Variant of (4)</strong></summary>

* Tweaked scaling factor & clipping range.  
* **Results** – 70.76 % / 70.66 %.  
* **Takeaway** – careful hyper-tuning matters even in low-bit land.
</details>

<details>
<summary><strong>Approach 6 – Multi-Head Attention & Progressive MoQ</strong></summary>

* Upgraded to full MH-Attention encoder; progressively lowered precision (8→4→1-bit) during fine-tuning.  
* **Results** – 70.18 % / 70.15 %.  
* **Takeaway** – capacity ↑, but extra heads partly cancelled by quantization loss.
</details>

<details>
<summary><strong>Approach 7 – Pre-trained BERT (+ STE)</strong></summary>

* Started from <code>bert-base-uncased</code>; swapped every dense/attn projection to BitLinear; STE for back-prop.  
* **Results** – **85.67 % / 85.65 %** (best).  
* **Takeaway** – pre-training supplies strong linguistic priors; 1-bit layers fine-tune well with STE.
</details>

<details>
<summary><strong>Approach 8 – Ternary BERT (+ Activation Quant)</strong></summary>

* Pushed further: ternary weights {-1,0,+1} + per-layer activation quant + sub-layer norm.  
* **Results** – 50.92 % / 34.36 %.  
* **Takeaway** – too aggressive; activation quant hurt expressive power.
</details>

---

> ⏱️ **Note**: Each approach in this repository was trained for only **3 epochs** due to time and resource constraints. Despite this, the results already reveal promising trends in low-bit training. I believe the community can build on these implementations — running longer training schedules, tuning hyperparameters, and applying these ideas to larger tasks — to unlock even better performance and deeper insights into ultra-low precision NLP.


## 📊 Result Table

| # | Model / Technique                                                                    | Acc. | F1  |
|:-:|---------------------------------------------------------------------------------------|:---:|:---:|
| 1 | Scratch Transformer + 1-bit weights                                                  | 76.38 | 76.38 |
| 2 | + PosEnc & AMP                                                                        | 62.27 | 61.26 |
| 3 | + Dropout & QAT                                                                       | 63.88 | 63.65 |
| 4 | + Median Scaling & STE                                                                | 69.84 | 69.65 |
| 5 | Variant of 4                                                                          | 70.76 | 70.66 |
| 6 | + MH-Attention & Progressive MoQ                                                     | 70.18 | 70.15 |
| 7 | **BERT-base + STE-quantized**                                                         | **85.67** | **85.65** |
| 8 | Ternary BERT (+ Activation Quant)                                                     | 50.92 | 34.36 |


---

## ⚠️ Notebook Rendering Issue on GitHub

Due to a known compatibility issue with Jupyter widgets metadata (`metadata.widgets.state` missing), GitHub is currently **unable to render the notebook properly on the web interface**.

📌 **Workaround**:  
To view and run the notebook without errors, please **clone the repository locally** and open the notebook in **VS Code**, **JupyterLab**, or another local IDE.



