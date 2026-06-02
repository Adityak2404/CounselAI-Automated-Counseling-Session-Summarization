## 📝 Project Summary

**CounselAI: Automated Counseling Session Summarization using Emotion-Aware T5 and Multi-Dataset Training**

CounselAI is a **four-stage automated pipeline** developed as my **B.Tech Major Project** (2025-2026) at Delhi Technological University. It addresses the significant documentation burden on mental health professionals by automatically generating structured clinical summaries from lengthy, unstructured counseling conversations.

### Key Highlights

- **Emotion-Aware Feature Extraction**: Fine-tuned **BERT-base** model to classify emotional intensity into **five tiers**.
- **Abstractive Summarization**: Fine-tuned **T5-base** model using multi-dataset training on **MEMO**, **CounselChat**, and **Mental Health Counseling Conversations** datasets.
- **Thematic Analysis**: Applied **LDA (Latent Dirichlet Allocation)** for topic modeling to improve summary structure.
- **Structured Clinical Reports**: Generates professional, section-wise summaries suitable for EHR integration.

### Methodology Pipeline

### Results & Performance

Evaluated on the MEMO dataset with strong performance:

- **ROUGE-1**: 44.8  
- **ROUGE-2**: 28.5  
- **ROUGE-L**: 36.3  
- **BERTScore F1**: 0.883

ROUGE Score Comparison Across Models  
BERTScore F1 Comparison Across Models  
Emotional Intensity Tier Classification Performance  
T5 Training and Validation Loss Over Epochs

### Tech Stack
**Python** • **PyTorch** • **Hugging Face Transformers** • **BERT** • **T5** • **NLTK** • **scikit-learn** • **LDA**

---

**This project showcases the practical application of Generative AI and NLP in the mental health domain**, combining emotional intelligence with advanced summarization techniques.
