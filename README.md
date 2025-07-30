# Data Augmentation with Large Language Models  

## 1 · Datasets  
* **IMDB** – binary (positive/negative) movie-review classification  
* **AG News** – four-class (world, sports, business, sci/tech) news-headline classification  

Each dataset is loaded, trimmed, and label-balanced in `dataset.ipynb`, then saved to `data/`.

## 2 · Augmentation Pipeline  
1. For every sentence, generate **3–6** paraphrases with three LLMs:  
   • `tuner007/pegasus_paraphrase` • `humarin/chatgpt_paraphraser_on_T5_base` • `mrm8488/t5-small-finetuned-quora-for-paraphrasing`  
2. Merge original and generated sentences.  
3. Embed all texts with `sentence-transformers/all-MiniLM-L6-v2` into 384-dimensional vectors (`embedding.ipynb`).  
4. Train an `sklearn` `MLPClassifier`; compare different training sizes (×2, ×3, ×5) and test sizes (×4, ×6) in `train.ipynb`.  
5. Combine predictions of original and augmented test instances via majority vote (ensemble) for the final decision.

## 3 · Results (Highlights)  
| Dataset  | Baseline Acc. | Best Train-Aug | Best Test-Aug (Ensemble) | Gain |
|----------|---------------|---------------|--------------------------|------|
| IMDB     | 82 %          | 88 % (×3 train) | 89 % (×4 test)          | +7 pp |
| AG News  | 81 %          | 91 % (×5 train) | 92 % (×6 test)          | +11 pp |

Augmentation yields steady accuracy improvements on AG News and notable gains on IMDB; the ensemble of original + augmented views consistently outperforms single-view inference. For full tables and plots see `Rapor.pdf`.
