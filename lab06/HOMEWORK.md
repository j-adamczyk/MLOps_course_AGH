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
3. Use the basal decision model from the lab as a zero-shot intent classifier:
   - use the 77 intents as options, with readable names, e.g. "card arrival" instead of `card_arrival`
     (names are in `dataset.features["label"].names`)
   - a question with k options uses the first k option codes from section 3.6 of the
     [prompt format](https://github.com/rkinas/basal/blob/main/docs/FORMAT2.md); check that each code
     is a single token at the answer position, like in the lab
   - use the temperature for 77 options (`"27-255"` bucket of `"choice"` in `CALIBRATION.json`)
   - prompts are much longer than in the lab, so use a sample of the test set, e.g. 500 queries
4. Test the resulting model:
   - check a few quality metrics
   - for multiclass metrics, use `average="macro"` (non-AUROC) or `multi_class="ovo"` (AUROC)
   - plot the F1-score for each class
   - check the calibration with the multiclass version of ECE (confidence of the predicted class vs
     accuracy)
   - comment:
     - is the overall F1-score comparable with fine-tuned models from the papers?
     - is it high enough from your perspective?
     - are there significant differences between classes? Do the worst classes have unclear names?
5. Use LangTest for behavioral testing:
   - make the model wrapper return intent names, and save the test data with intent names instead of
     numbers, like in the lab
   - use the robustness tests from the lab, and add at least 2 more that make sense for this task
   - analyze the results, are there any additional problems with the classifier?
6. Use Captum for local explainability:
   - modify the `explain_text()` function from the lab to accommodate multiclass classification,
     e.g. use the predicted class as the `target`
   - explain predictions for a few samples from a test set
   - comment, is the model focusing on the right parts of texts?

Grading:
- section 1: 1 point
- section 2: 2 points
- section 3: 2 points
- section 4: 2 points
- section 5: 1.5 points
- section 6: 1.5 points
