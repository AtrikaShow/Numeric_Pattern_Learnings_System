# Numeric Pattern Learning System (PyTorch)

## 🚀 Introduction
Welcome! This project is a foundational Machine Learning system built from scratch using Python and PyTorch.  
Unlike complex demos (like face recognition) that hide the details, this project peels back the layers to show exactly how a machine *learns*.

**The Goal:** Teach a computer to figure out the relationship `y = 3x + 2` just by looking at examples.

---

<!-- ## 🧠 Mental Model for Developers (MERN Analogy)

If you come from a web development background, here is how to translate ML concepts:

| ML Concept | Backend/Web Analogy |
| :--- | :--- |
| **Model (`model.py`)** | The **Business Logic / Service**. It handles the data flow but starts "stupid" (empty logic). |
| **Training (`train.py`)** | A **Database Migration** script that populates your Service with logic. |
| **Weights/Parameters** | The **Database State**. These are the numbers the model "saves" to remember what it learned. |
| **Loss Function** | **Unit Tests**. It checks "Did the model get the right answer?" and returns a score. |
| **Optimizer** | An **Auto-Fix Linter**. It looks at the failed Unit Test (High Loss) and tweaks the code to fix it. |
| **Predict (`predict.py`)** | The **REST API Endpoint**. It takes user input, runs the logic, and returns a JSON response. |

--- -->

## 📂 Project Structure

- **`data_generator.py`**: A helper script to create mock data (synthetic numbers).
- **`model.py`**: Defines the "Brain" or architecture of the Neural Network.
- **`train.py`**: The "Teacher". It loops thousands of times to adjust the Brain until it understands the pattern.
- **`predict.py`**: The "Application". It loads the saved Brain and uses it to answer new questions.

---

## 🛠️ Setup & Installation

1. **Clone the repo** (or navigate to folder).
2. **Create a Virtual Environment** (The Sandbox):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
   ```
3. **Install Dependencies**:
   ```bash
   pip install torch 
   ```

---

## 🚦 How to Run

### Step 1: Train the Model
This runs the "Schooling" process. It will generate data, normalize it, feed it to the model, and save the result.
```bash
python3 train.py
```
*Output:* You will see the "Loss" number go down. When it finishes, it creates `liner_regression_model.pth`.

### Step 2: Make Predictions
Now that the model is trained/saved, we can use it like a real app.
```bash
python3 predict.py
```
*Output:* The model will take inputs (like `5.0`) and predict the output (approx `17.0`).

---

## 🧩 Key Technical Concepts Learned

### 1. The Neural Network (MLP)
We used a **Multilayer Perceptron**. It's not just a straight line equation; it has a "Hidden Layer" with ReLU activation. This means it has the *potential* to learn curved/complex patterns, even though we only taught it a simple line today.

### 2. Normalization (The "Exploding Gradient" Fix)
Neural Networks fail if you feed them massive numbers (like 5000). 
We fixed this by **Scaling**:
- We calculate the average of our data.
- We shift all data so the average is 0.
- This creates a stable mathematical environment for the model to learn.

### 3. Serialization (`.pth` files)
Just like saving a user to MongoDB, we must "Serialize" our model to disk. We save:
- The Weights (The learned logic).
- The Normalization Stats (The math needed to prepare inputs).

---

## 🔮 Next Steps
- Try changing the formula in `data_generator.py` to something complex like `y = x^2` and see if the model can learn a curve!
- Build a Flask/FastAPI server to serve this model over HTTP.

Happy Coding! 🚀
