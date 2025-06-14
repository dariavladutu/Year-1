
# Barman’s Helper: Image Classification App  
### Year 1 – Block C | Applied Data Science & AI | Breda University of Applied Sciences  
**Author:** Daria-Elena Vlăduțu

---

## Project Overview  
This project focused on designing an interpretable and user-centered image classification solution. In response to a business challenge from the **Innovation Square**, I developed **Barman’s Helper**—a smart image classifier that helps bars and restaurants manage inventory in real time by identifying key bar items through image recognition.

From ideation to prototype, the solution applies responsible AI principles, user testing, and deep learning to deliver practical value to the **HoReCa industry**.

---

## Problem Statement  
Inventory loss and disorganization are among the top operational challenges in the hospitality sector. Barman’s Helper addresses this with a classification model that detects common inventory items—like beverage bottles, fruit, and coffee packages—directly from images, then connects to a stock management platform via API.

---

## Innovation Goals
- Classify visual inventory items (6 product classes)
- Integrate model into a user-friendly interface prototype
- Prioritize interpretability and ethical AI in deployment

---

## Model Development

### Dataset  
A **custom dataset** of 712 images was scraped and preprocessed. Six classes included:
- `alcoholic_beverages_bottle`
- `non_alcoholic_beverages`
- `non_alcoholic_beverages_bottle`
- `coffee_grounds_package`
- `cocktail_fruits`
- `lemon_lime`

### Models Built
1. **Simple CNN**  
   - 8-layer architecture  
   - Accuracy: **65%**

2. **VGG16 (Transfer Learning)**  
   - Fine-tuned using feature extraction  
   - Accuracy: **86%**, Val Accuracy: **87%**

### Technical Stack
- `Python`, `Keras`, `TensorFlow`, `NumPy`, `OpenCV`
- Data augmentation & transfer learning
- Pretrained architectures (VGG16)

---

## Responsible AI

Model transparency and fairness were crucial:
- Used **Integrated Gradients** and **SHAP** to visualize feature influence
- Evaluated tradeoffs between interpretability vs. accuracy
- Performed **error analysis** and Human-Level Performance (HLP) estimation

---

## Human-Centered Design

### Wireframe Prototype  
A working app prototype was built in **Figma** and tested via a **think-aloud study**:
- A/B Testing measured usability improvements
- UI refined based on participant feedback
- Demo video [available here](https://edubuas-my.sharepoint.com/:v:/g/personal/236578_buas_nl/EcYTQt9cpN9Hq10ubsP-sSwBIXsndYW1OUlHJ6BdK7Sz9w?e=odtkND)

---

## Project Structure
```
barmans-helper-ml/
├── creative_brief/             # Proposal, DataLab responses
├── notebooks/                  # Final deep learning notebooks
├── explainability/             # XAI visuals (SHAP, IG)
├── prototype/                  # Final wireframe (Figma), user study
├── reports/                    # Learning log, final presentation
├── media/                      # Demo video & wireframe screenshots
└── README.md
```

---

## Key Learnings
- Developed and tuned multiple neural networks
- Improved model explainability through XAI tools
- Designed an AI prototype guided by real user needs
- Balanced accuracy and interpretability in model lifecycle
