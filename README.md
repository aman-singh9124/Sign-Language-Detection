# ✋ ASL Sign Language Detection System (6PM–10PM IST Only)

Welcome to the **ASL Sign Language Detection** project! This system uses a trained **CNN model** to identify hand gestures from American Sign Language (A–Z). It runs through a **Gradio-powered web interface** with support for **image uploads** and **real-time webcam detection** — but only during the time window of **6PM to 10PM IST** 🕕

---

## 🌟 Key Highlights

🔠 Recognizes ASL Alphabets (A–Z)  
🎯 CNN-based trained `.h5` model  
🖼️ Supports Image Upload & Webcam Input  
🕒 Works only between **6:00 PM to 10:00 PM (IST)**  
📱 User-friendly **Gradio GUI** (works in Colab too)

---

## 📂 Project Structure

📁 sign-language-detection/
├── sign_language_model.h5 ✅ Trained CNN model
├── Sign_Language_GUI.ipynb ✅ Gradio + Time Logic
├── requirements.txt ..Required Python packages
├── README.md 📘 This file


---

## 🚀 How to Run (Colab Preferred)

1. 📥 Upload your model `sign_language_model.h5`
2. ▶️ Run all cells in `Sign_Language_GUI.ipynb`
3. 🖼️ Use the **Upload Image** or **Webcam Detection** tab
4. 🕘 Make sure time is between **6PM and 10PM IST**

---

## 🖥️ Sample GUI Preview

| Image Upload Example             | Prediction Output       |
|----------------------------------|--------------------------|
| ![A Gesture](sample_test_images/A_test.jpg) | ✅ `Predicted: A`       |

---

## 🧠 Model Info

- Architecture: CNN with multiple Conv2D + MaxPooling layers
- Input Shape: 64x64x3 images
- Activation: ReLU → Softmax
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Output: 26 classes (A to Z)

---

## 🛠️ Dependencies

Install if running locally:

```bash
pip install tensorflow gradio opencv-python pillow


