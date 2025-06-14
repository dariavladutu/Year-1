
# ANWB Vehicle Trajectory Prediction – Team 18  
### Year 1 – Block D | Applied Data Science & AI | Breda University of Applied Sciences  
**Author:** Daria-Elena Vlăduțu (Team 18)

---

## Project Overview  
In this final team-based project, we collaborated to design a predictive solution for the **ANWB (Royal Dutch Touring Club)**. The goal was to explore how historical trip data and geospatial information could be used to predict driver destinations in real time, supporting smarter mobility and infrastructure planning.

---

## Problem Statement  
The ANWB collects GPS trip data from vehicle drivers across the Netherlands. Our task: create a system that predicts the **final destination** of a trip in progress, using only data collected until a given moment—enabling applications like early traffic warnings, route suggestions, or service alerts.

---

## Team Project, Individual Grading  
While the solution was developed collaboratively, each student maintained their own notebook, logs, and deliverables. This README reflects my **individual contributions**, particularly in data preprocessing, model design, and evaluation.

---

## Data Highlights  
- Raw GPS logs: ~57,000 trajectories  
- Attributes: Latitude, longitude, timestamp, anonymized driver ID  
- Processed features: distance, speed, stop detection, bearing, trip ID, normalized path encoding

---

## Preprocessing & Feature Engineering  
- Outlier removal using z-scores on distance and time deltas  
- Stop detection via velocity thresholds  
- Polyline simplification (Douglas-Peucker)  
- Feature engineering:
  - Cumulative trip distance
  - Segment speed and heading direction
  - Time since trip start
  - Latitude/longitude normalization

---

## Model Design  
### Final Model: **XGBoost Classifier**
- Predicts final grid cell based on partial trajectory
- Trained on split trajectory windows (~30%–70%)
- Tuned with GridSearchCV

### Benchmarked with:
- k-Nearest Neighbors
- Random Forest
- Support Vector Machines
- Naive Baseline (last known point)

---

## Evaluation
- Accuracy: **83.6%** (XGBoost)
- Visual validation on Folium maps and grid overlays
- Error radius used as a second metric for real-world viability

---

## Responsible AI & Ethics  
- Ensured GPS anonymization throughout the pipeline  
- Complied with GDPR principles on geospatial tracking  
- Considered bias from trip frequency and geographic imbalance  

---

## Tools & Technologies
- `Python`, `scikit-learn`, `XGBoost`, `Pandas`, `NumPy`  
- `Folium`, `Geopandas`, `Matplotlib`  
- Trello (team management)  
- Jupyter Notebook (final delivery)

---

## Repository Structure
```
anwb-trajectory-prediction/
├── 📂 notebooks/                 
│   ├── Final notebook.ipynb
│   ├── Jupyter_Notebook_template_Y1D_*.ipynb
│   └── Preprocessing steps.ipynb

├── 📂 data/                      
│   ├── openmeteo_breda.csv
│   ├── knmi_preprocessing_documentation.odt
│   ├── Preprocessing steps combined file.pdf
│   ├── ANWB preprocessing.docx
│   └── Additional processed CSVs

├── 📂 reports/                  
│   ├── Project Proposal Team 18.pdf
│   ├── Project Proposal - Legal & ML Doc.pdf
│   ├── Proposal - Deployment & Interface Design.pdf
│   ├── Improve the Safety in Breda - Plan.pdf
│   ├── Feedback & Final Report files
│   └── DS&AI - Y1D - Trello.pdf

├── 📂 logs/
│   ├── Learning Log - Y1D_2023-24_ADSAI.pptx
│   ├── Worklog - Y1D_2023-24_ADSAI.xlsx
│   └── Section C - ILO slides.pptx

├── 📂 deployment/
│   ├── Deployment planning docs
│   ├── Interface screenshots / design files
│   └── User testing reports

├── 📂 legal/
│   └── GDPR considerations, anonymization steps

├── 📂 code/
│   ├── py_files/
│   ├── test scripts
│   └── model export/validation tools

├── 📂 documentation/
│   ├── Backlog_Evidence.png
│   ├── proposal.pdf
│   ├── task_docs/
│   └── Week_4_group_discussion_summary.pdf

├── README.md

```

---

## Key Learnings
- Learned to build scalable ML pipelines for geospatial data  
- Practiced team collaboration with clear versioning & review  
- Balanced performance with privacy in a sensitive application domain  
- Gained deeper understanding of real-time prediction constraints

---

