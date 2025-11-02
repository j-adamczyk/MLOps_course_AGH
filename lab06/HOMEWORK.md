# Laboratory 6 homework

Homework is applying all techniques learned during the lab on the
[Banking77 dataset](https://huggingface.co/datasets/PolyAI/banking77).
It consists of online banking queries, annotated with their corresponding intents, resulting in 77 classes.
It's already split into training and testing parts, with F1 score typically used for evaluation.
It is also [known to have label errors](https://aclanthology.org/2022.insights-1.19/),
great for data-centric AI approach.
Label descriptions are [available here](https://huggingface.co/datasets/PolyAI/banking77#data-fields).

Instructions:
- do tasks in order, they have been designed to build upon prior steps
- write clean code, descriptive variable and function names, use comments where necessary etc.
- send your resulting application code as .zip file

1. Download and load the data. Perform the initial exploration and cleaning.
   - it is already split into training and testing parts
   - labels go from 0 to 76
   - checks should include e.g.:
     - checking data types, number of texts,
     - plotting the class distribution
     - plotting the histogram of text lengths
2. Use CleanLab to detect and fix data quality issues:
   - use `all-MiniLM-L6-v2` Sentence Transformer embeddings + logistic regression as the model
   - note that `LogisticRegressionCV` works out-of-the-box for multiclass classification, and
     you should use `class_weight="balanced"` in case of class imbalance
   - detect and fix: label issues, near duplicates, outliers
   - when analyzing duplicates, you can print rows at given indices with `.iloc[]`
   - comment:
     - do your results agree with [the paper](https://aclanthology.org/2022.insights-1.19/)
       pointing out the label quality issues?
     - are the problems severe and worth fixing in your opinion?
3. Train a small text classifier:
   - you can use `distilbert/distilbert-base-uncased` model, or any other that you want
   - make 80-20% train-valid split
   - use F1-score to select the best model
   - perform hyperparameter tuning, or use larger learning rate and train for longer than in
     the lab, since the dataset is small and task is harder
   - modify how `y_pred_proba` and `y_pred` are returned to accommodate multiclass classification,
     `np.argmax()` may be useful
4. Test the resulting model:
   - check a few quality metrics
   - for multiclass metrics, use `average="macro"` (non-AUROC) or `multi_class="ovo"` (AUROC)
   - plot the F1-score for each class
   - comment:
     - is the overall F1-score comparable with the papers?
     - is it high enough from your perspective?
     - are there significant differences between classes?
5. Use Giskard for behavioral testing:
   - modify `prediction_function()` appropriately
   - HuggingFace pipeline has `top_k=None` option to get scores for all labels, not just the label
     with the highest probability
   - analyze the results, are there any additional problems with the resulting classifier?
6. Use Captum for local explainability:
   - modify the `explain_text` function from the lab to accommodate multiclass classification
   - explain predictions for a few samples from a test set
   - comment, is the model focusing on the right parts of texts?

Grading:
- section 1: 1 point
- section 2: 2 points
- section 3: 2 points
- section 4: 2 points
- section 5: 1.5 points
- section 6: 1.5 points
