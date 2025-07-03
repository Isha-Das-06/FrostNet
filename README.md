# ❄️ FrostNet – Deep Learning for Cryogenic Temperature Anomaly Detection

**FrostNet** is a time-series anomaly detection project focused on monitoring and analyzing **cryogenic temperatures**—typically temperatures below -150°C. This system is particularly relevant for applications in scientific research, industrial storage, space engineering, and superconductor technologies, where maintaining temperature stability is crucial.

Using a **Long Short-Term Memory (LSTM)** deep learning model, FrostNet is capable of learning patterns from historical data and identifying subtle anomalies that could signal faults, system failures, or unexpected behavior in a cryogenic environment.

---

## 🧠 Project Goals

- Generate synthetic cryogenic temperature data that simulates real-world sensor behavior.
- Preprocess and normalize the data for deep learning workflows.
- Train an LSTM-based neural network for time-series modeling and anomaly detection.
- Evaluate model performance using precision, recall, MAE, and RMSE.
- Visualize temperature fluctuations and detected anomalies to gain actionable insights.

---

## 📁 Folder & File Structure

```

FrostNet/
├── data/                               # Contains raw and preprocessed datasets
│   └── cryogenic\_synthetic.csv
├── outputs/                            # Trained LSTM model, evaluation logs, saved metrics
│   └── lstm\_model.pth
├── visualization/                      # Graphs, anomaly plots, prediction vs actual plots
│   └── anomaly\_visualization.png
├── 01\_data\_generation\_and\_loading.ipynb
├── 02\_data\_preprocessing\_and\_cleaning.ipynb
├── 03\_model\_building\_and\_training.ipynb
├── 04\_model\_evaluation\_and\_anomaly\_detection.ipynb
├── 05\_visualization\_and\_insights.ipynb
└── README.md

````

---

## ⚙️ Environment Setup

- **Python version**: 3.8  
- **Libraries used**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `torch` (PyTorch)
  - `datetime`
  - `os`, `random` (for synthetic data)

> Note: You can install dependencies using the following command:

```bash
pip install -r requirements.txt
````

---

## 📓 Notebook Descriptions

### 1. 📄 `01_data_generation_and_loading.ipynb`

* Generates synthetic cryogenic temperature data using a **random walk** model with occasional spikes to simulate anomalies.
* Timestamps are generated at 1-minute intervals.
* Anomaly labels (`1` for anomaly, `0` for normal) are inserted at random points.
* Output is saved as `data/cryogenic_synthetic.csv`.

![Category Distribution](visualizations/plotting%20raw%20data.jpg)

### 2. 🧼 `02_data_preprocessing_and_cleaning.ipynb`

* Loads the generated data and inspects for missing values.
* Applies normalization to temperature readings using MinMaxScaler.
* Handles any data gaps or duplicates.
* Produces histograms and correlation plots to understand data distribution.

![Category Distribution](visualizations/Comparison%20Graph%20-%20Before%20vs%20After%20Preprocessing.jpg)

### 3. 🏗️ `03_model_building_and_training.ipynb`

* Builds an **LSTM model** using PyTorch for sequence modeling.
* The model accepts a sliding window of temperature readings and learns to reconstruct sequences.
* Trained for **30 epochs** using Mean Squared Error (MSE) loss.
* Saves the trained model to `outputs/lstm_model.pth`.

![Category Distribution](visualizations/Plot%20Training%20and%20Testing%20Loss.jpg)

### 4. 📊 `04_model_evaluation_and_anomaly_detection.ipynb`

* Uses the trained LSTM model to generate predicted temperature sequences.
* Computes reconstruction error for each time point.
* Anomalies are detected using a fixed threshold on the reconstruction error.
* **Evaluation Metrics**:

  * **Precision**: `0.0120`
  * **Recall**: `0.0600`
  * **MAE (Mean Absolute Error)**: `0.0587`
  * **RMSE (Root Mean Square Error)**: `0.2423`

  ![Category Distribution](visualizations/Model%20Evaluation%20Metrics.jpg)

### 5. 🖼️ `05_visualization_and_insights.ipynb`

* Plots original vs predicted temperature values.
* Highlights detected anomalies on time-series plots.
* Provides insights into model performance and failure cases.
* Charts are saved to the `visualization/` folder for reporting and presentations.

![Category Distribution](visualizations/Detected%20Anomalies%20in%20Cryogenic%20Temperature.jpg)
![Category Distribution](visualizations/Cryogenic%20Temperature%20Over%20Time.jpg)

---

## 📈 Key Features

* ✅ Fully self-contained and modular Jupyter notebooks
* ✅ End-to-end anomaly detection pipeline with synthetic data
* ✅ Deep learning using **LSTM networks**
* ✅ Supports visualization of anomalies for interpretability
* ✅ Logs and model outputs saved for reproducibility

---

## 📂 Dataset Details

**File**: `data/cryogenic_synthetic.csv`

| Column Name        | Description                                      |
| ------------------ | ------------------------------------------------ |
| `Timestamp`        | 1-minute interval datetime values                |
| `Temperature (°C)` | Simulated cryogenic sensor readings              |
| `Anomaly`          | Binary label (1 = anomaly, 0 = normal condition) |

> The data mimics the behavior of real cryogenic systems with periodic anomalies and realistic noise patterns.

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/FrostNet.git
cd FrostNet
```

2. Open the notebooks in Jupyter Notebook or JupyterLab.

3. Run notebooks **in the following order**:

   * `01_data_generation_and_loading.ipynb`
   * `02_data_preprocessing_and_cleaning.ipynb`
   * `03_model_building_and_training.ipynb`
   * `04_model_evaluation_and_anomaly_detection.ipynb`
   * `05_visualization_and_insights.ipynb`

4. View graphs in the `visualization/` folder and model metrics in `outputs/`.

---

## 💡 Future Enhancements

* Incorporate **real-world cryogenic datasets**.
* Use **Transformer**-based time-series models for comparison.
* Integrate with a **dashboard** (e.g., Streamlit) for live anomaly monitoring.
* Enable **online learning** for adapting to incoming data streams.

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## 👨‍💻 Author

- **Name:** [Isha Das]
- **Email:** [ishadas2006@gmail.com]

---