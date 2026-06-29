# Romanized Bengali Emotion Analysis using Transformer Models via Zero-Shot learning approcah 

## Project Overview
This project explores the challenging task of emotion analysis in Romanized Bengali (also known as Banglish), a language variant commonly used in digital communication. We investigate the effectiveness of various Transformer-based models, including a purpose-built BanglaBERT, Multilingual BERT (mBERT), and DistilBERT, under different training and zero-shot evaluation scenarios. The primary goal is to understand how these models perform when confronted with cross-script and cross-lingual challenges, specifically focusing on Romanized Bengali text.

## Dataset
The dataset used for this project is loaded from `Emotion.csv`. It contains text samples labeled with various emotions and categorized by language ('EN' for English, 'BN' for native Bengali script, 'RN' for Romanized Bengali).

**Key Characteristics:**
*   **Languages**: English (EN), Native Bengali (BN), Romanized Bengali (RN)
*   **Emotions**: joy, sadness, anger, disgust, fear, surprise, none, others
*   **Class Imbalance**: The dataset exhibits significant class imbalance, addressed by calculating and applying class weights during model training.

## Methodology

### 1. Data Preprocessing
*   **Bengali Text Cleaning**: For native Bengali (BN) text, we applied cleaning steps including removal of punctuation (ASCII and Bengali) and numbers (Western and Bengali digits), and standardization of whitespaces.
*   **Class Weight Calculation**: `sklearn.utils.class_weight.compute_class_weight` was used to calculate inverse-frequency class weights, which are then passed to `torch.nn.CrossEntropyLoss` during training to mitigate the impact of class imbalance.

### 2. Transformer Models
We employed three different Transformer models:

*   **BanglaBERT (`csebuetnlp/banglabert`)**: A BERT-like model pre-trained specifically on a large Bengali corpus, making it highly suitable for native Bengali text.
*   **Multilingual BERT (mBERT) (`bert-base-multilingual-cased`)**: Pre-trained on 104 languages, offering strong multilingual capabilities.
*   **DistilBERT (`distilbert-base-uncased`)**: A smaller, faster, and lighter version of BERT, efficient for resource-constrained environments.

### 3. Cascaded Banglish Emotion Pipeline
An initial pipeline was established to demonstrate Banglish-to-Bangla transliteration. This pipeline consists of two stages:
*   **Stage 1**: Transliteration of Romanized Bengali (Banglish) to native Bengali script using `fms-byte/banglish_to_bangla` model.
*   **Stage 2**: Emotion classification of the native Bengali script using a pre-trained `csebuetnlp/banglabert` model.

### 4. Training and Evaluation Scenarios
Models were fine-tuned and evaluated under different conditions:

#### A. Training on Native Bengali (BN) Data
Each model (BanglaBERT, mBERT, DistilBERT) was fine-tuned on the native Bengali (BN) subset of the dataset. Training involved PyTorch `Dataset` and `DataLoader` for batching, `AdamW` optimizer, and `get_linear_schedule_with_warmup` scheduler. Evaluation was performed on a validation set (80/20 train/validation split).

#### B. Training on English (EN) Data
Each model was subsequently fine-tuned exclusively on the English (EN) subset of the dataset. The training setup was similar to the BN training scenario.

#### C. Zero-Shot Evaluation on Romanized Bengali (RN) Data (from EN-trained models)
Models trained *exclusively on English data* were then used to perform zero-shot emotion classification on the Romanized Bengali (RN) data under three scenarios:
1.  **Direct RN**: Predicting directly on the raw Romanized Bengali text.
2.  **RN-BN Transliterated**: Predicting on Romanized Bengali text after it was transliterated to native Bengali script (RN → BN).
3.  **RN-BN-EN Translated**: Predicting on Romanized Bengali text after a two-step translation process (RN → BN → EN).

## Key Findings and Results

### Validation Accuracy (Models Trained on BN Data):
*   **BanglaBERT**: Achieved the highest validation accuracy on BN data, demonstrating its effectiveness for native Bengali text.
*   **mBERT**: Performed reasonably well on BN data.
*   **DistilBERT**: Showed the lowest accuracy on BN data, likely due to its English-centric pre-training and smaller architecture.

### F1-Score Comparison Across Emotions (Models Trained on BN Data):
The F1-score plots revealed that some emotions (e.g., 'joy', 'none') were consistently better predicted across models, while minority classes (e.g., 'fear', 'others', 'sad', 'surprise') had significantly lower F1-scores, indicating the persistent challenge of class imbalance even with weights.

### Zero-shot Accuracy (Models Trained on EN Data, Evaluated on RN Data):

| Model                   | Scenario             | Accuracy |
| :---------------------- | :------------------- | :------- |
| BanglaBERT (EN-trained) | Direct RN            | 0.5022   |
| BanglaBERT (EN-trained) | RN-BN Transliterated | 0.4362   |
| BanglaBERT (EN-trained) | RN-BN-EN Translated  | 0.4204   |
| mBERT (EN-trained)      | Direct RN            | 0.4222   |
| mBERT (EN-trained)      | RN-BN Transliterated | 0.3149   |
| mBERT (EN-trained)      | RN-BN-EN Translated  | 0.4116   |
| DistilBERT (EN-trained) | Direct RN            | 0.3694   |
| DistilBERT (EN-trained) | RN-BN Transliterated | 0.2832   |
| DistilBERT (EN-trained) | RN-BN-EN Translated  | 0.4415   |

**Observations:**
*   **BanglaBERT (EN-trained)**: Surprisingly performed best on **Direct RN** text when trained on English. This suggests some robustness to Romanized script despite its Bengali pre-training and English fine-tuning.
*   **DistilBERT (EN-trained)**: Achieved its highest zero-shot accuracy when the RN text was **RN-BN-EN Translated**. This is expected, as DistilBERT is English-centric, and translating the input to English before classification aligns with its training data.
*   **mBERT (EN-trained)**: Showed varied performance, with **Direct RN** performing slightly better than RN-BN Transliterated, but **RN-BN-EN Translated** performing comparably to Direct RN.
*   **Transliteration vs. Direct**: For BanglaBERT, direct Romanized input was better than transliterated, while for DistilBERT, an extra English translation step was beneficial. This highlights the complex interplay between model architecture, pre-training, fine-tuning language, and input representation.

## Conclusion
This project demonstrates the complexities of cross-script and cross-lingual emotion analysis, particularly for low-resource language variants like Romanized Bengali. While purpose-built models like BanglaBERT show strong performance on native script, their generalization to Romanized forms or English-trained scenarios varies. Multilingual models like mBERT offer flexibility, and English-centric models like DistilBERT benefit significantly from translation to their native language. The choice of strategy (direct, transliterated, or translated) depends heavily on the specific model and its training background.

## Future Work
*   **Larger RN Datasets**: Fine-tuning models directly on larger Romanized Bengali datasets could significantly improve performance.
*   **Multitask Learning**: Explore training models on both translation/transliteration and emotion classification simultaneously.
*   **Advanced Cross-lingual Techniques**: Investigate more sophisticated cross-lingual transfer learning methods.
*   **Error Analysis**: Conduct a deeper error analysis for misclassified emotions, especially in minority classes.
