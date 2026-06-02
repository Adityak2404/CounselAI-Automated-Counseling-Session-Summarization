## 📝 Project Summary

**CounselAI: Automated Counseling Session Summarization using Emotion-Aware T5 and Multi-Dataset Training**

CounselAI is a **four-stage automated pipeline** developed as my **B.Tech Major Project** (2025-2026) at Delhi Technological University. It addresses the significant documentation burden on mental health professionals by automatically generating structured clinical summaries from lengthy, unstructured counseling conversations.

### Key Highlights

- **Emotion-Aware Feature Extraction**: Fine-tuned **BERT-base** model to classify emotional intensity into **five tiers**.
- **Abstractive Summarization**: Fine-tuned **T5-base** model using multi-dataset training on **MEMO**, **CounselChat**, and **Mental Health Counseling Conversations** datasets.
- **Thematic Analysis**: Applied **LDA (Latent Dirichlet Allocation)** for topic modeling to improve summary structure.
- **Structured Clinical Reports**: Generates professional, section-wise summaries suitable for EHR integration.

### Methodology Pipeline
<img width="758" height="330" alt="{1E5C398D-BD56-4E02-B36A-1B4B59E7BCAA}" src="https://github.com/user-attachments/assets/862fcd3b-990a-435a-9000-7608275228bc" />


### Results & Performance

Evaluated on the MEMO dataset with strong performance:

- **ROUGE-1**: 44.8  
- **ROUGE-2**: 28.5  
- **ROUGE-L**: 36.3  
- **BERTScore F1**: 0.883

<img width="766" height="450" alt="{A25E52DC-0501-4CDC-A97C-CBF5A83703E4}" src="https://github.com/user-attachments/assets/983f2e5b-8f5d-4972-b1e7-4e34c61a0900" />

<img width="824" height="463" alt="{3F175E5A-AC2D-4657-A4F9-8C6AA6ACA7B9}" src="https://github.com/user-attachments/assets/3b612fda-290e-42a0-bc49-3e51dfa4c911" />

<img width="812" height="417" alt="{B638C48D-17F6-4B1A-8836-614CB1F18F32}" src="https://github.com/user-attachments/assets/9dffad49-eb5a-4995-b56b-222f4a70a29c" />

<img width="756" height="360" alt="{1854DC03-05A0-4EEB-8DA2-B62E31EC4554}" src="https://github.com/user-attachments/assets/4b6aa99b-0cff-4bfb-ac75-957579dccd5f" />


### Tech Stack
**Python** • **PyTorch** • **Hugging Face Transformers** • **BERT** • **T5** • **NLTK** • **scikit-learn** • **LDA**
