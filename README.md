# PETCT-Prediction
Clinical records gathered from hospitalized patients have been in the rise during recent years due to continuous improvements in medical technology and since it has become more accessible to more hospitals around the world. Medical results reported by doctors and from medical instruments are highly important to use for detection, preferably in early stages, for the presence or suspicion of severe disease. Many diseases are still being studied today by doctors to find a proper medical solution to treat patients efficiently and to understand better characteristics. Patients undergo examinations to detect illnesses. These examinations include scans such as Positron Emission Tomography (PET), Computed Tomography (CT), Magnetic Resonance Imaging (MRI), and Ultrasound. Doctors evaluate the scans, explain the diagnosis results, and suggest types of treatment options to the patients. Recent studies in cancer classification have yet to propose a proper practical solution for helping doctors achieve accurate diagnoses with additional and profound insights from text, especially those written in Hebrew.

We present in this paper a classification system for identifying severity levels of cancer in 10 different sectors of the body. Test-Time Augmentation (TTA) is applied for enlarging essential text information retrieval and robustness to the model while maintaining inference time. This method automatically analyzes PET-CT pathology reports of patients. It indicates the patients current medical state with very high confidence. A large institutional report repository is used from a hospital to extract the semantics of the most contributing factors indicating the patients’ physical health condition impacted by their disease. The goal of this paper is to significantly increase the improvement of Hebrew PET-CT reports classification accuracy. Augmentation sets are generated and evaluated by the model during test-time to enhance, magnify and deepen the extraction of essential text from each report. This method is the novel of this study and results demonstrate higher performance by AUC evaluation criteria of the TTA model outperforming the Baseline model without TTA integration.

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


