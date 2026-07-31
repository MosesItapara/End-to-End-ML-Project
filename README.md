# End-to-End ML Project — California Housing Price Prediction

A full regression pipeline that predicts median house value for California districts, built while working through Chapter 2 of Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn and PyTorch*. Covers the complete ML workflow from raw data to a tuned, saved model.

## Dataset

The [California housing dataset](https://github.com/ageron/data) (1990 census, ~20,000 district-level records) with features like median income, location, room counts, and ocean proximity.

## Workflow

- **Data exploration & visualization** — geographical scatter plots, distribution histograms
- **Stratified train/test split** — sampled by income category to avoid sampling bias
- **Correlation analysis** — identifying which features drive `median_house_value`
- **Data cleaning** — median imputation for missing numerical values
- **Categorical encoding** — one-hot encoding for `ocean_proximity`
- **Feature engineering** — custom transformers for ratio features (bedrooms/rooms, rooms/house, people/house) and cluster-similarity features
- **Feature scaling** — log-transforms for long-tailed features, standardization for the rest
- **Transformation pipelines** — `sklearn` `Pipeline` / `ColumnTransformer` tying preprocessing together
- **Model training & evaluation** — Linear Regression and Decision Tree baselines, evaluated via cross-validation
- **Hyperparameter tuning** — `GridSearchCV` and `RandomizedSearchCV` over a Random Forest ensemble
- **Error analysis** — inspecting feature importances of the best model
- **Final evaluation** — scored once on the held-out test set
- **Model persistence** — best model serialized with `joblib`

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (pipelines, transformers, model selection)
- Matplotlib

## Status

Currently a single Colab notebook (`End_to_End_ML_Project.ipynb`) covering training end-to-end. Not yet served — next steps are to modularize the pipeline into scripts, wrap the trained model behind an API, and containerize/deploy it, following the structure used in this author's other production ML projects.

## Getting Started

```bash
pip install pandas numpy scikit-learn matplotlib joblib
jupyter notebook End_to_End_ML_Project.ipynb
```

Run all cells to reproduce data prep, training, tuning, and the final `California_Housing_Model.pkl` artifact.
