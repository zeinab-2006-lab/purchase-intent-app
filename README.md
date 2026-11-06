 # 🛒 Will This Visitor Buy? – Purchase Intent Prediction

Most website visitors leave without buying. This project predicts whether a website visitor is likely to make a purchase based on browsing session data. It also suggests whether the website should do nothing, show a coupon, or offer a chat assistant.

🔗 **Live App:** [Will This Visitor Buy?](https://purchase-intent-app-mvneznprlredwp4ytz7dtd.streamlit.app/)

🔗 **GitHub Repository:** [Purchase Intent App](https://github.com/zeinab-2006-lab/purchase-intent-app)

## Results

Measured on a held-out test set (2,441 sessions) that was never used for training, tuning, or threshold selection.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline (always "no purchase") | 0.500 | 0.156 | 0.000 | 0.000 | 0.000 |
| Logistic Regression (tuned) | 0.926 | 0.687 | 0.657 | 0.678 | 0.668 |
| Random Forest (tuned) | 0.935 | 0.742 | 0.716 | 0.639 | 0.675 |
| Gradient Boosting (tuned) | 0.939 | 0.757 | 0.681 | 0.709 | 0.695 |

### Key Takeaways

- The best model catches 71% of real buyers (271 of 382 in the test set), and 68% of the visitors it flags actually buy.
- Accuracy is misleading here: the baseline is 84.4% accurate by never predicting a purchase, yet it finds zero buyers.
- The strongest signal by far is PageValues (buyers: median 16.8; non-buyers: 0.0), followed by the month of the visit.

## What the App Does

Users enter a browsing session, including:

- Pages visited.
- Time spent on different page types.
- Bounce rate and exit rate.
- Page value.
- Month and visitor type.

The model returns a purchase score and one of three suggested actions:

- **No action:** Very likely to buy anyway (about 9% of sessions; 80% of them buy).
- **Show coupon:** Undecided visitor (about 30% of sessions; 25% of them buy).
- **Offer chat assistant:** Low intent (about 60% of sessions; only 1.3% buy).

A sidebar allows a manager to adjust the decision thresholds. Invalid inputs are caught and displayed with clear messages.

## Approach

| Step | What Was Done |
|---|---|
| Data | UCI Online Shoppers Purchasing Intention — 12,330 sessions (12,205 after dropping 125 duplicates), 15.6% purchase rate |
| EDA | Univariate, bivariate (purchase rate by month / visitor type / page value), and multivariate analysis |
| Pipeline | Split first → feature engineering → scaling + one-hot encoding → model, all inside one scikit-learn Pipeline (no leakage) |
| Models | Majority-class baseline, Logistic Regression, Random Forest, Gradient Boosting |
| Tuning | 5-fold stratified CV + RandomizedSearchCV on the training set only |
| Threshold | Chosen from out-of-fold training predictions (0.77); action zones learned the same way |
| Evaluation | ROC-AUC, PR-AUC, precision/recall/F1, confusion matrix, error analysis, permutation importance |
| Simulation | What-if coupon campaign with an assumed coupon effect (clearly labelled — real uplift needs an A/B test) |

## Limitations

- One website and one year of data; the dataset has no January or April sessions.
- PageValues dominates the model but only exists if your analytics tool tracks it.
- Buyers who never reach high-value pages (PageValues = 0) are mostly missed.
- The model predicts who is likely to buy, not who a coupon will convince.
- Class weights make the scores a ranking signal rather than exact probabilities.

## Run It Yourself

Clone the repository:

```bash
git clone https://github.com/zeinab-2006-lab/purchase-intent-app.git
cd purchase-intent-app
```

Install the dependencies:

```bash
pip install -r requirements.txt
pip install jupyter matplotlib seaborn tabulate
```

Download the dataset from the UCI link below and put the CSV file in the `data/` folder.

Run `purchase_intent_notebook.ipynb` (Restart & Run All) to recreate the models folder.

Start the Streamlit app:

```bash
streamlit run app.py
```

**Note:** The dataset is not included in this repository. Download it from the UCI Machine Learning Repository.

## Files

| File | Purpose |
|---|---|
| `purchase_intent_notebook.ipynb` | Full analysis: framing → EDA → pipeline → models → tuning → evaluation → export |
| `features.py` | Feature engineering shared by the notebook and the app |
| `app.py` | Streamlit web application |
| `models/` | Saved pipeline and metadata used by the app |
| `requirements.txt` | Library versions used for training |
| `Will This Visitor Buy Final Project..pptx` | Final project presentation |
## Dataset & Credits

Dataset: C. Okan Sakar & Yomi Kastro, **Online Shoppers Purchasing Intention Dataset**, UCI Machine Learning Repository.

https://archive.ics.uci.edu/dataset/468

Built by **Zainab Reda** as an individual final project for the Machine Learning course at NextGen Institute.
