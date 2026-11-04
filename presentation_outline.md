# Presentation outline — 11 slides, about 10 minutes

Every [X] must come from your own notebook output. Take the figures straight from the notebook.

| # | Slide | Content | Figure to use | Time |
|---|---|---|---|---|
| 1 | Title | "Will This Visitor Buy?" · team names · NextGen Institute | — | 0:15 |
| 2 | The problem | Most visitors leave without buying. Showing a discount to everyone wastes margin. Goal: act only on the right visitors. | Simple funnel sketch | 0:45 |
| 3 | Decision & metric | Three actions: no action / coupon / chat. Why accuracy is misleading (baseline accuracy [X]%); ROC-AUC + recall/precision matter. | Class-balance bar chart (§2) | 0:45 |
| 4 | The data | UCI Online Shoppers: [N] sessions, 17 features, [X]% purchase rate. Cleaning: duplicates dropped, negative durations clipped. | Column groups table | 0:45 |
| 5 | EDA insight 1–2 | Your two strongest findings (e.g. PageValues of buyers vs non-buyers) | Boxplots (§3.2) | 1:00 |
| 6 | EDA insight 3–4 | Visitor type / month / exit rate findings | Purchase rate by group (§3.2) | 1:00 |
| 7 | Pipeline | Split first → feature engineering → scaling/encoding → model, all in one `Pipeline`. How leakage was avoided. | Pipeline diagram | 1:00 |
| 8 | Models & tuning | Baseline + 3 algorithms, 5-fold CV, RandomizedSearchCV, selected on CV not on test | CV vs tuned table (§6–7) | 1:00 |
| 9 | Results | Test ROC-AUC / PR-AUC / recall / precision; confusion matrix; what the model misses (error analysis); top features | Confusion matrix + importance chart (§9) | 1:30 |
| 10 | Live demo | Open the deployed app, run two examples, show the suggested action and the policy sliders | The app | 1:00 |
| 11 | Recommendation & limitations | Use the three-zone policy; A/B test the coupon (simulation uses an *assumed* effect); one site, one year; PageValues must be tracked | Zone table (§8) | 1:00 |

**Speaker tips**
- Say the number and what it means in plain words ("it catches X out of every 10 real buyers").
- Be upfront that the campaign simulation is a what-if and that real uplift needs an A/B test — it shows maturity.
- Be ready to explain: why you split before preprocessing, why recall/PR-AUC instead of accuracy, why the threshold was chosen on training data.
