
---

# **Self-Supervised Representation Learning: A Comparative Study on Hyper-parameters, Model Selection, Evaluation, and Data Imbalance**

## **1. Introduction**

Self-supervised learning (SSL) has become a powerful approach for learning visual representations without human annotations.
In this project, we adopt a SimCLR-style contrastive learning framework and investigate how different design choices influence downstream classification performance on the STL-10 dataset.

The goal of this report is to analyze our system through six required machine-learning themes:

1. **Hyper-parameter tuning**
2. **Model selection**
3. **Model evaluation**
4. **Under-fitting**
5. **Over-fitting**
6. **Learning with unbalanced data**

Instead of discussing these topics independently, we connect them through a single experimental pipeline and analyze how each aspect emerges naturally during our study.

---

## **2. Hyper-parameter Tuning**

Hyper-parameter tuning is essential in SSL because the contrastive objective is sensitive to batch size, temperature, and the duration of pre-training.
We tuned two key hyper-parameters:

* **Temperature (τ ∈ {0.1, 0.2})**
* **Batch size (128 vs. 256)**
* **SSL epochs (20 vs. 50)**

Observations:

* **Temperature strongly affected alignment vs. uniformity:**

  * τ = 0.1 → stronger attraction between positives → more stable but sometimes collapsed representations
  * τ = 0.2 → more separation but also more noisy gradients
* **Batch size improved contrastive learning stability:**

  * Larger batch = more negative samples per step
  * But also **slower GPU throughput** on a laptop GPU
* **Training duration (20 vs. 50 epochs) directly affected convergence**

Through empirical tuning, we found:

* **Best performing configuration:**
  **τ = 0.1, batch = 128, 50 epochs**
* **Worst performing configuration:**
  **τ = 0.1, batch = 256, 20 epochs** (clear under-fitting)

These observations become crucial when we perform model selection.

---

## **3. Model Selection**

Model selection aims to choose the best representation among multiple pre-trained encoders.

We compared **four distinct models**:

| Model                 | τ   | Batch | Epochs | Acc.       |
| --------------------- | --- | ----- | ------ | ---------- |
| temp0.1_batch128      | 0.1 | 128   | 50     | **0.5300** |
| temp0.2_batch128      | 0.2 | 128   | 50     | 0.3769     |
| temp0.1_batch256_ep20 | 0.1 | 256   | 20     | 0.2024     |
| temp0.2_batch256_ep20 | 0.2 | 256   | 20     | 0.3533     |

Selection rationale:

* The **τ = 0.1 + batch128 + long training** achieved the most discriminative features.
* Very low accuracy in batch256_ep20 models indicated insufficient representation learning due to reduced training time.

Thus, model selection naturally followed from our hyper-parameter tuning results.

---

## **4. Model Evaluation**

To evaluate the learned representations, we used:

* **A linear probe** trained for 20 epochs on STL-10 labels
* **Test accuracy**
* **Per-class performance trends**
* **Confusion matrices** to inspect class-level behavior

Key evaluation findings:

* τ = 0.1 consistently produced more balanced decision boundaries
* τ = 0.2 improved a few classes (e.g., truck, cat) but harmed others (bird, car)
* The worst models collapsed into predicting only a few classes, indicating severe under-fitting

These outcomes gave us a practical understanding of how SSL representations transfer to downstream tasks.

---

## **5. Under-fitting**

Under-fitting occurs when a model fails to learn enough structure from the data.

We observed under-fitting in:

### **• Small-epoch models (20 epochs)**

The models with only 20 SSL epochs, especially **batch256_ep20**, showed:

* Extremely low accuracy (≈20%)
* Almost uniform or degenerate predictions
* High confusion between unrelated classes

Reason:

* SSL requires many iterations before the representation space becomes meaningful
* Larger batch sizes reduce gradient frequency → further slowing convergence

Thus, insufficient training time produced clear under-fitting.

---

## **6. Over-fitting**

Over-fitting is less common in SSL pre-training but can appear in **linear evaluation**.

Signs of mild over-fitting:

* Training accuracy > Validation/Test accuracy
* Some classes became over-represented in predictions
  (e.g., the model repeatedly predicting “ship” or “truck”)

reason：

* The STL-10 labelled set is small (5k samples)
* Linear probe is trained using full supervision
* Some SSL representations cause class imbalance in the feature space

Although over-fitting was not severe, it still appeared as a performance gap between training and testing.

---

## **7. Learning with Unbalanced Data**

STL-10 has **highly imbalanced intra-class difficulty**:

* Some classes (e.g., airplane, ship, truck) are visually distinctive
* Others (cat, dog, monkey) overlap heavily and are harder even with supervised training

SSL magnifies this imbalance:

* High-variance classes suffer more from contrastive transformations
* Confusion matrices show that “cat”, “dog”, “monkey” remain consistently difficult across all models
* The worst-performing models collapse onto majority-looking patterns (predicting “airplane” or “ship”)

This confirms that **data imbalance affects SSL representations**, not only supervised models.

---

## **8. Conclusion**

This project demonstrates how the required ML topics—hyper-parameter tuning, model selection, evaluation, under-/over-fitting, and data imbalance—naturally arise within a single self-supervised learning pipeline.

Key takeaways:

* Hyper-parameters such as temperature, batch size, and training duration critically influence SSL quality
* Model selection must be based on consistent evaluation metrics, not training loss
* Under-fitting is the dominant failure mode when training is too short
* Over-fitting appears mildly during supervised linear probing
* Data imbalance continues to challenge SSL models, especially for visually similar classes

Overall, τ = 0.1 with batch size 128 and longer training produced the best general-purpose representation.

---

# **Appendix: Code Structure (for TA understanding)**

```
project/
│
├── simclr_runAll.py           # Full hyper-parameter sweep (128 batch)
├── simclr_runAll_256_20.py    # Reduced training version (256 batch, 20 epochs)
├── evaluate_models.py         # Linear evaluation, confusion matrices, metrics
│
├── results/
│   ├── temp0.1_batch128_encoder.pt
│   ├── temp0.2_batch128_encoder.pt
│   ├── temp0.1_batch256_ep20_encoder.pt
│   ├── temp0.2_batch256_ep20_encoder.pt
│   └── summary.json
│
└── data/                      # STL10 dataset
```

---
