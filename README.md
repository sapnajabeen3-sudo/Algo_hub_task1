# 🏠 Bengaluru House Price Prediction

A simple, beginner-friendly end-to-end Machine Learning project that predicts house
prices in Bengaluru using **Linear Regression**. Built with Python and deployed
using **Streamlit**.

---

## 📌 Project Overview

This project walks through the complete Machine Learning workflow:

1. **Data Preprocessing** – Cleaning the raw Bengaluru housing dataset.
2. **Exploratory Data Analysis (EDA)** – Understanding patterns in the data.
3. **Model Training** – Training a Linear Regression model.
4. **Deployment** – A Streamlit web app where users can input house details
   and get an instant price prediction.

The goal of this project is to keep the code **simple, clean, and easy to
understand** — ideal for internship submissions and learning purposes.

---

## ✨ Features

- Clean and well-commented Python code (no complex classes or pipelines).
- Full data cleaning: handles missing values, ranges in square footage,
  BHK extraction, rare location grouping, and outlier removal.
- Exploratory Data Analysis notebook with visualizations.
- Linear Regression model trained with Scikit-learn.
- Model evaluation using MAE, MSE, RMSE, and R² Score.
- Interactive Streamlit web app for real-time predictions.
- Ready to deploy on **Streamlit Community Cloud**.

---

## 📊 Dataset Information

- **Source:** `Bengaluru_House_Data.csv`
- **Rows:** ~13,320 (before cleaning)
- **Columns:**
  - `area_type` – Type of area (Super built-up, Plot, Built-up, Carpet)
  - `availability` – Availability status (dropped during preprocessing)
  - `location` – Location of the property in Bengaluru
  - `size` – Number of bedrooms (converted to `bhk`)
  - `society` – Housing society name (dropped, too many missing values)
  - `total_sqft` – Total area in square feet (cleaned to numeric)
  - `bath` – Number of bathrooms
  - `balcony` – Number of balconies
  - `price` – Price of the house in Lakhs (INR) — **target variable**

After cleaning, outlier removal, and encoding, the dataset is used to train
the model.

---

## 🤖 Machine Learning Model

- **Algorithm:** Linear Regression (Scikit-learn)
- **Why Linear Regression?** It's simple, interpretable, and a great
  starting point for understanding regression-based price prediction.
- **Encoding:** One-Hot Encoding applied to `area_type` and `location`.
- **Evaluation Metrics:**
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² Score

Run `train_model.py` to see the exact metrics printed in your terminal.

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── preprocessing.py        # Data cleaning functions
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── model.pkl               # Trained Linear Regression model
├── model_columns.pkl       # Column order used during training
├── cleaned_data.csv        # Cleaned dataset (output of preprocessing.py)
├── Bengaluru_House_Data.csv  # Raw dataset
├── notebooks/
│   └── EDA.ipynb            # Exploratory Data Analysis notebook
├── images/                  # Screenshots for this README
└── .gitignore
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/House-Price-Prediction.git
   cd House-Price-Prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the required libraries**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

### Step 1: Preprocess the data (optional — already done, output saved as `cleaned_data.csv`)
```bash
python preprocessing.py
```

### Step 2: Train the model (optional — already done, output saved as `model.pkl`)
```bash
python train_model.py
```

### Step 3: Run the Streamlit app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Deployment

This app is ready to deploy on **Streamlit Community Cloud**:

1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"** and connect your GitHub repository.
4. Set the main file path to `app.py`.
5. Click **Deploy** — that's it!

Make sure `model.pkl`, `model_columns.pkl`, and `cleaned_data.csv` are
included in the repository so the app can load them.

---

## 🖼️ Screenshots

_Add screenshots of your running app here._

```
images/
  ├── home_page.png
  └── prediction_result.png
```

Example:

![App Screenshot](images/home_page.png)

---

## 🚀 Future Improvements

- Try other regression models (Ridge, Lasso, Random Forest) and compare performance.
- Add more features like proximity to schools, hospitals, or public transport.
- Improve outlier detection using more advanced statistical techniques.
- Add data visualizations directly inside the Streamlit app.
- Deploy with a custom domain and add authentication for saved searches.

---

## 👤 Author

**Your Name**
Machine Learning Intern
Feel free to connect with me on [LinkedIn](https://linkedin.com) or check out
my other projects on [GitHub](https://github.com).

---

⭐ If you found this project helpful, consider giving it a star on GitHub!
