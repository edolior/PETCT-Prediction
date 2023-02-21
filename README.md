# PETCT-Prediction
It is interesting to see that advancements in medical technology have led to an increase in the number of clinical records being gathered from hospitalized patients. The use of medical instruments, such as PET, CT, MRI, and ultrasound, has become essential in detecting diseases, especially in their early stages, to provide appropriate treatment options to patients. In this context, this paper proposes a new method called Text Test Time Augmentation (TTTA) to improve the classification accuracy of cancer severity levels in 10 different body sectors using PET-CT pathology reports of patients written in Hebrew. TTTA uses Test-Time Augmentation (TTA) to enlarge essential text information retrieval and robustness to the model while maintaining inference time. The method generates and evaluates augmentation sets during training and test time to magnify and deepen the extraction of essential text from each report, which is the main novelty of this study. The study uses a large institutional report repository from Ziv hospital to extract the semantics of the most contributing factors indicating the patients' physical health condition impacted by their disease. Results indicate that the TTTA method outperforms the Baseline models without TTA integration based on the AUC evaluation criteria. This method has the potential to assist doctors in accurately classifying cancer severity levels, providing deeper insights into the characteristics of the disease, and suggesting appropriate treatment options to patients.

MVC Architecture:

<img width="492" alt="PETCT Architecture_V2" src="https://user-images.githubusercontent.com/44165771/212911771-70781778-9880-4f09-81b7-0cd9df5c9a04.png">

Model Pipepline:

<img width="546" alt="petct_workflow_cap4" src="https://user-images.githubusercontent.com/44165771/212910375-9eb023ed-5a0c-4717-9e27-fdf75d547330.png">

a - Both training and testing text report inputs are first transformed from PDF format to text and then parsed into paragraphs. b - The parser identifies and adds different data depending on the different sectors of diagnosis that were written in the report. Reports contain different titles for each patient (e.g., previous medical history, sectors examined, or any unexpected additional information for a certain patient). If a new feature is detected, it will be added dynamically to the overall set of features. If an existing feature is missing for a certain patient, it is set to null. c - Pre-processing stage includes the normalization of characters to lowercase, punctuation and stop-words removal, correction of invalid alpha-numeric characters, and removal of delimiter tokens specified to the hospital's format. d - One hot encoding transforms categorical features that are under 16 possible outcomes (e.g., Health Care, Service History). Imbalanced and sparse discovered features are removed if more than 90% of the data has missing values. e - Imputation is applied when features contain more than 50% of missing data (e.g., Age, Glucose Level). Multiple Imputation was applied since it outperformed imputation results over KNN and by Median. f - Training and testing samples are transformed from raw text to TF-IDF embedded representation by three sub-set groups: demographics, examination settings, and sector diagnoses. g - The training embedding TF-IDF output is used as features for the model. However, as part of the TTA algorithm, testing samples are also transformed into different augmentations. These augmentation sets are converted to TF-IDF embedded representation as well. h - An XGBoost model performs training and testing-time augmentation. It outputs a multi-class and multi-label examination result for each patient to achieve a more accurate diagnosis over models not applying TTA.

Pipeline Functions:

<img width="128" alt="model_functions_pycharm" src="https://user-images.githubusercontent.com/44165771/212913656-e3262e4d-0b44-462c-a52f-b461dfbfbe17.png">

How to Run:

1. Before running, choose your pipeline functions in Model.py (Package: Model) 
2. Run Main.py
3. That's it!


