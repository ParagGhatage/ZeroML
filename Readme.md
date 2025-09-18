<a href="https://zeroml.dev" target="_blank" rel="noopener noreferrer">
  <img src="https://github.com/ParagGhatage/ZeroML/blob/main/web/public/ZeroML_banner.png" alt="ZeroML Banner">
</a>


> **ZeroML** – Build. Train. Deploy. Version. Visualize. Optimize. All in one platform.

[![PyPI Version](https://img.shields.io/pypi/v/zeroml)](https://pypi.org/project/zeroml/)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/HF-Integration-orange)](https://huggingface.co/)
[![RunPod](https://img.shields.io/badge/RunPod-Integration-purple)](https://www.runpod.io/)

---

## 🌟 ZeroML – The Hybrid ML Platform

ZeroML is a **visual-first, fully extensible ML platform** that lets you:

* Build end-to-end ML pipelines with **drag-and-drop ease**
* Train, fine-tune, and optimize models efficiently
* Clean and version data with **prompt-driven automation**
* Deploy **production-grade APIs** in seconds

All pipelines, models, and datasets are **fully versioned** and ready for collaboration.

---

## 🧠 Hybrid Strategy

| Model Size | Training             | Deployment   | Versioning       |
| ---------- | -------------------- | ------------ | ---------------- |
| Small      | Local / Hugging Face | HF Endpoints | Hugging Face Hub |
| Large      | RunPod               | RunPod API   | Hugging Face Hub |

> Showcase fast, scale smart, and manage all your ML assets centrally.

---

## 🚀 Features

### 1️⃣ End-to-End ML Pipeline

* Drag-and-drop **pipeline builder**
* Prompt-driven **data cleaning & feature engineering**
* Real-time **training metrics** & **model visualization**

### 2️⃣ Deployment & Versioning

* Deploy anywhere: **HF Endpoints**, **RunPod**, or your own server
* Every dataset, model, and pipeline is **versioned for reproducibility**

### 3️⃣ Optimization & Tuning

* Hyperparameter tuning with **live feedback**
* GPU/CPU utilization optimization for **maximum efficiency**
* Smart batching, checkpointing, and memory management

### 4️⃣ Extensible & Modular

* Integrate your **custom libraries**
* Plugin system for **data processing**, **models**, or **deployment backends**

### 5️⃣ Visualizations

* Interactive **training curves**
* Feature importance & correlation maps
* Compare multiple models side by side

---


## 📚 Documentation

Comming soon...

---



# 🤝 Contributing

We welcome contributions! Whether it’s fixing bugs, adding features, or improving documentation, your help is highly appreciated. Follow the instructions below to get the project running locally.

---

## 🛠 Setup Instructions

### 1. Frontend

1. Navigate to the frontend folder:

```bash
cd web
````

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

* The frontend will be accessible at: [http://localhost:3000](http://localhost:3000)

---

### 2. Backend

1. Navigate to the backend folder:

```bash
cd api
```

2. Sync and run the backend:

```bash
uv sync
uv run main.py
```

#### Managing Dependencies

* **Add a dependency:**

```bash
uv add <dependency_name>
```

* **Remove a dependency:**

```bash
uv remove <dependency_name>
```

---

### ✅ Tips for Contributors

* Make sure to **pull the latest changes** before starting your work.
* Follow **consistent code formatting** (Prettier/ESLint recommended).
* Test your changes thoroughly before creating a pull request.
* Provide a clear **description of your changes** in the PR.

---

Thanks for contributing! Your help makes this project better for everyone. 🚀



## 🛡 License

ZeroML is licensed under the **Apache-2.0 License**

---

## 🔥 Join the ZeroML Revolution

Build. Train. Deploy. Version. Visualize. Optimize. All in one.

[🌐 Visit Website](https://zeroml.dev)
