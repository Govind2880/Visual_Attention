# 🧠 AI Text Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.11.8-blue?style=for-the-badge&logo=python)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.0-green?style=for-the-badge&logo=django)](https://djangoproject.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A sophisticated **AI-powered sentiment analysis web application** built with **fine-tuned DistilBERT** that achieves **91.2% accuracy** on IMDB dataset. Features real-time analysis, interactive attention visualization, and a modern glassmorphism UI with comprehensive REST API.

## ✨ Key Features

### 🎯 **Advanced AI Model**
- **Fine-tuned DistilBERT** for binary sentiment classification
- **91.2% accuracy** on IMDB movie reviews dataset
- **66M parameters** optimized for production deployment
- **Real-time inference** with sub-100ms response times

### 🔍 **Explainable AI with Attention Visualization**
- **Interactive attention heatmap** showing model focus
- **Multi-layer attention aggregation** from transformer heads
- **Hover tooltips** displaying attention scores
- **Color-coded word importance** with smooth animations

### 🌟 **Modern Web Interface**
- **Glassmorphism design** with animated gradients
- **Fully responsive** mobile-first layout
- **Real-time character counter** with validation
- **Interactive example buttons** for quick testing
- **Dynamic charts** with Chart.js visualization
- **Smooth animations** powered by Anime.js

### 🚀 **Production-Ready API**
- **RESTful endpoints** for integration
- **Batch processing** (up to 10 texts simultaneously)
- **Health monitoring** with system metrics
- **Rate limiting** (100 requests/hour)
- **Analytics tracking** with performance metrics
- **CORS support** for cross-origin requests

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Frontend      │    │   Django     │    │   AI Engine     │
│   (Glassmorphic │◄──►│   Backend    │◄──►│   DistilBERT    │
│    Interface)   │    │   + DRF      │    │   Fine-tuned    │
└─────────────────┘    └──────────────┘    └─────────────────┘
        │                       │                     │
        ▼                       ▼                     ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   UI Components │    │  API Layer   │    │   Attention     │
│   • Charts      │    │  • /predict  │    │   Visualization │
│   • Animations  │    │  • /batch    │    │   • Multi-layer │
│   • Interactions│    │  • /health   │    │   • Aggregation │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

## 📊 Model Performance

| Metric | Score | Details |
|--------|-------|---------|
| **Accuracy** | 91.2% | On IMDB test dataset |
| **Precision** | 91.0% | Binary classification |
| **Recall** | 91.3% | Positive class detection |
| **F1-Score** | 91.1% | Harmonic mean |
| **Model Size** | 267MB | Optimized safetensors format |
| **Inference Time** | <100ms | Average response time |
| **Parameters** | 66M | DistilBERT architecture |

## 🛠️ Technology Stack

### **Backend Framework**
- **Django 5.0** - Modern Python web framework
- **Django REST Framework 3.15.2** - API development
- **Gunicorn** - Production WSGI server
- **Python-dotenv** - Environment management

### **AI/ML Stack**
- **PyTorch 2.0+** - Deep learning framework
- **Transformers 4.21+** - Hugging Face library
- **DistilBERT** - Efficient transformer architecture
- **NumPy** - Numerical computations

### **Frontend Technologies**
- **Tailwind CSS** - Utility-first styling
- **Chart.js 4.4.0** - Interactive data visualization
- **Anime.js 3.2.1** - Smooth animations
- **Font Awesome 6.0** - Icon library
- **Inter Font** - Modern typography

### **Deployment & DevOps**
- **AWS Elastic Beanstalk** - Cloud deployment
- **Git LFS** - Large file version control
- **Safetensors** - Efficient model serialization

## 🚀 Quick Start

### Prerequisites

- **Python 3.11.8+** (recommended)
- **Git** with **Git LFS** support
- **8GB+ RAM** (for model loading)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

### 1️⃣ Clone Repository

```bash
# Clone the repository
git clone https://github.com/Govind2880/Visual_Attention.git
cd Visual_Attention

# Install Git LFS and pull model files (IMPORTANT!)
git lfs install
git lfs pull
```

### 2️⃣ Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Configuration

```bash
# Generate Django secret key and create .env file
python generate_secret_key.py

# Or manually create .env file with:
echo "DJANGO_SECRET_KEY=your-secret-key-here" > .env
echo "DEBUG=True" >> .env
echo "ALLOWED_HOSTS=localhost,127.0.0.1" >> .env
```

### 4️⃣ Database Setup

```bash
# Apply database migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser
```

### 5️⃣ Static Files

```bash
# Collect static files for production-like serving
python manage.py collectstatic --noinput
```

### 6️⃣ Run Application

```bash
# Start development server
python manage.py runserver

# Or specify port
python manage.py runserver 8000
```

**🎉 Open your browser:** `http://localhost:8000`

## 🔧 Project Structure

```
Visual_Attention/
├── 📁 attention_classifier/        # Django project configuration
│   ├── settings.py                 # Settings with environment variables
│   ├── urls.py                    # Main URL configuration
│   ├── wsgi.py                    # WSGI application
│   └── asgi.py                    # ASGI application (async)
├── 📁 classifier/                 # Main Django application
│   ├── views.py                   # Business logic & API endpoints
│   ├── serializers.py             # DRF serializers
│   ├── urls.py                    # App URL patterns
│   ├── 📁 templates/              # HTML templates
│   │   └── index.html             # Main UI template
│   └── 📁 static/                 # CSS/JS static files
│       └── css/style.css          # Custom styles
├── 📁 models/                     # AI model files (Git LFS)
│   ├── model.safetensors          # Model weights (267MB)
│   ├── config.json                # Model configuration
│   ├── tokenizer_config.json      # Tokenizer settings
│   └── special_tokens_map.json    # Special tokens
├── 📁 src/                        # AI utilities and core logic
│   ├── inference.py               # Model loading and inference
│   ├── visualize.py               # Attention visualization
│   └── train.py                   # Training script
├── 📁 model_training/             # Training scripts
│   └── train_classifier.py        # Production training script
├── 📁 app/                        # Additional applications
│   └── streamlit_app.py           # Streamlit demo (optional)
├── 📄 .env                        # Environment variables
├── 📄 .gitignore                  # Git ignore patterns
├── 📄 .gitattributes              # Git LFS configuration
├── 📄 requirements.txt            # Python dependencies
├── 📄 manage.py                   # Django management
├── 📄 generate_secret_key.py      # Secret key generator
├── 📄 procfile                    # Heroku deployment
└── 📄 LICENSE                     # MIT License
```

## 🌐 API Documentation

### 📝 Single Text Prediction

**Endpoint:** `POST /api/predict/`

```http
POST /api/predict/
Content-Type: application/json

{
    "text": "This movie was absolutely fantastic and amazing!"
}
```

**Response:**
```json
{
    "text": "This movie was absolutely fantastic and amazing!",
    "prediction": "Positive",
    "confidence": 0.9234,
    "probabilities": {
        "Negative": 0.0766,
        "Positive": 0.9234
    },
    "attention_html": "<span class='attention-word' style='...'>This</span>...",
    "metadata": {
        "processing_time_ms": 45.2,
        "text_length": 44,
        "word_count": 7,
        "timestamp": "2025-01-XX T XX:XX:XX"
    },
    "success": true
}
```

### 📊 Batch Prediction

**Endpoint:** `POST /api/batch-predict/`

```http
POST /api/batch-predict/
Content-Type: application/json

{
    "texts": [
        "Great movie, loved it!",
        "Terrible experience, waste of time.",
        "It was okay, nothing special."
    ]
}
```

**Response:**
```json
{
    "results": [
        {
            "text": "Great movie, loved it!",
            "prediction": "Positive",
            "confidence": 0.8945,
            "probabilities": { "Negative": 0.1055, "Positive": 0.8945 },
            "processing_time_ms": 32.1
        }
        // ... more results
    ],
    "summary": {
        "total_texts": 3,
        "successful_predictions": 3,
        "total_processing_time_ms": 98.7,
        "average_time_per_text_ms": 32.9
    },
    "success": true
}
```

### 🩺 Health Check

**Endpoint:** `GET /api/health/`

```http
GET /api/health/

Response:
{
    "status": "healthy",
    "model_loaded": true,
    "prediction_working": true,
    "system_info": {
        "model_type": "DistilBERT",
        "total_predictions": 1523,
        "average_confidence": "87.3%",
        "positive_rate": "58.6%",
        "avg_response_time_ms": "42.1"
    },
    "message": "AI Sentiment Analyzer is running"
}
```

### 📈 Analytics Dashboard

**Endpoint:** `GET /api/analytics/`

```http
GET /api/analytics/

Response:
{
    "analytics": {
        "total_predictions": 1523,
        "positive_count": 892,
        "negative_count": 631,
        "average_confidence": 87.3,
        "last_predictions": [...]
    },
    "summary": {
        "total_predictions": 1523,
        "accuracy_estimate": "91.2%",
        "model_size": "66M parameters",
        "response_time": "< 100ms average"
    }
}
```

## 🚀 Deployment Options

### AWS Elastic Beanstalk (Current Production)

**Already deployed at:** [visual-attention-app-env.eba-jjsphpfx.us-east-1.elasticbeanstalk.com](http://visual-attention-app-env.eba-jjsphpfx.us-east-1.elasticbeanstalk.com)

1. **Install EB CLI:**
   ```bash
   pip install awsebcli
   ```

2. **Initialize and Deploy:**
   ```bash
   eb init -p python-3.11 attention-classifier
   eb create attention-classifier-env
   eb deploy
   ```

3. **Set Environment Variables** in EB Console:
   - `DJANGO_SECRET_KEY`
   - `DEBUG=False`
   - `ALLOWED_HOSTS=your-domain.elasticbeanstalk.com`

### Alternative Deployment Platforms

<details>
<summary><strong>🐳 Docker Deployment</strong></summary>

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git git-lfs

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Pull LFS files
RUN git lfs pull

# Collect static files
RUN python manage.py collectstatic --noinput

EXPOSE 8000

# Start with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "attention_classifier.wsgi:application"]
```

Build and run:
```bash
docker build -t sentiment-analyzer .
docker run -p 8000:8000 -e DJANGO_SECRET_KEY="your-key" sentiment-analyzer
```

</details>

<details>
<summary><strong>🌐 Heroku Deployment</strong></summary>

1. **Install Heroku CLI** and login
2. **Create app:**
   ```bash
   heroku create your-app-name
   ```

3. **Set environment variables:**
   ```bash
   heroku config:set DJANGO_SECRET_KEY="your-secret-key"
   heroku config:set DEBUG=False
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

</details>

<details>
<summary><strong>🔧 Railway Deployment</strong></summary>

1. Connect GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on git push

</details>

## 🧪 Model Training

### Using Pre-trained Model (Recommended)

The repository includes a pre-trained model in the `models/` directory. This model was trained on IMDB dataset and achieves 91.2% accuracy.

### Training Your Own Model

```bash
# Navigate to training directory
cd model_training

# Install additional training dependencies (if needed)
pip install datasets scikit-learn tqdm

# Run training script
python train_classifier.py
```

**Training Configuration:**
- **Dataset:** IMDB movie reviews (50K samples)
- **Architecture:** DistilBERT-base-uncased
- **Training samples:** 10,000 (subset for demonstration)
- **Validation samples:** 2,000
- **Epochs:** 3
- **Batch size:** 4 (with gradient accumulation)
- **Learning rate:** 5e-5
- **Optimizer:** AdamW with weight decay

### Custom Dataset Training

To train on your own data:

1. **Prepare your data** in the format:
   ```json
   {"text": "Your text here", "label": 0}  // 0=Negative, 1=Positive
   ```

2. **Modify the training script:**
   ```python
   # In train_classifier.py
   # Replace IMDB dataset loading with your dataset
   dataset = load_dataset("your_dataset")
   ```

3. **Adjust hyperparameters** as needed for your domain

## 🎨 UI Customization

### Color Scheme

The application uses a modern glassmorphism design:

```css
/* Primary gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);

/* Glass cards */
backdrop-filter: blur(16px);
background: rgba(255, 255, 255, 0.95);

/* Attention colors */
High attention: rgba(220, 38, 38, 0.9)   /* Dark red */
Medium attention: rgba(248, 113, 113, 0.8) /* Medium red */
Low attention: rgba(252, 165, 165, 0.7)   /* Light red */
```

### Custom Themes

To create custom themes, modify the CSS variables in `classifier/templates/index.html`:

```css
:root {
  --primary-gradient: linear-gradient(135deg, #your-colors);
  --glass-bg: rgba(255, 255, 255, 0.95);
  --accent-color: #your-accent;
}
```

## 🔧 Advanced Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DJANGO_SECRET_KEY` | Django security key | - | ✅ |
| `DEBUG` | Debug mode | `False` | ❌ |
| `ALLOWED_HOSTS` | Comma-separated hosts | `*` | ❌ |
| `MODEL_PATH` | Path to model directory | `./models/` | ❌ |
| `PORT` | Server port | `8000` | ❌ |

### Performance Tuning

**Memory Optimization:**
```python
# In views.py - force CPU inference for memory-constrained environments
device = torch.device("cpu")
model.to(device)
```

**Caching:**
```python
# Add Redis caching for frequent predictions
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}
```

## 🔍 Troubleshooting

### Common Issues & Solutions

**🚨 Git LFS Model Files Not Downloaded**
```bash
# Ensure Git LFS is installed and pull files
git lfs install
git lfs pull

# Verify model files exist
ls -la models/
# Should show model.safetensors (267MB)
```

**🚨 CUDA Out of Memory**
```bash
# Force CPU inference
export CUDA_VISIBLE_DEVICES=""
python manage.py runserver
```

**🚨 Model Loading Errors**
```bash
# Check model path and files
python -c "
from pathlib import Path
model_dir = Path('models')
print('Model dir exists:', model_dir.exists())
print('Config exists:', (model_dir / 'config.json').exists())
"
```

**🚨 Static Files Not Loading**
```bash
# Collect static files
python manage.py collectstatic --clear --noinput

# Check STATIC_ROOT setting
python manage.py diffsettings | grep STATIC
```

**🚨 Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.11.8+
```

### Debug Mode

Enable verbose logging:
```bash
# Set environment variables
export DEBUG=True
export DJANGO_LOG_LEVEL=DEBUG

# Run with detailed output
python manage.py runserver --verbosity=2
```

## 📊 Performance Benchmarks

### Response Times (Average)
- **Single prediction:** 45ms
- **Batch prediction (5 texts):** 180ms
- **Model loading (first request):** 2-3s
- **Static file serving:** 10ms

### Memory Usage
- **Base Django app:** ~50MB
- **Model loaded:** ~300MB
- **Peak inference:** ~400MB
- **Recommended RAM:** 1GB+

### Scalability
- **Concurrent requests:** 10-50 (depending on hardware)
- **Throughput:** ~100 predictions/minute
- **Model sharing:** Single model instance serves all requests

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. **Fork and clone:**
   ```bash
   git clone https://github.com/Govind2880/Visual_Attention.git
   cd Visual_Attention
   ```

2. **Create development branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   # Additional dev tools
   pip install black flake8 pytest
   ```

4. **Make changes and test:**
   ```bash
   # Format code
   black .
   
   # Run tests
   python manage.py test
   
   # Check linting
   flake8 .
   ```

5. **Submit pull request:**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```

### Contribution Areas

- 🎯 **Model Improvements:** Multi-class, other architectures, fine-tuning
- 🌍 **Internationalization:** Multi-language support, translation
- 📊 **Analytics:** Advanced metrics, visualization, export features  
- 🎨 **UI/UX:** New themes, mobile optimization, accessibility
- 🔧 **Performance:** Caching, optimization, async processing
- 📖 **Documentation:** Tutorials, examples, API docs
- 🧪 **Testing:** Unit tests, integration tests, benchmarks

### Code Style

- **Python:** Follow PEP 8, use Black formatter
- **JavaScript:** Use ES6+, consistent indentation
- **HTML/CSS:** Semantic HTML, mobile-first CSS
- **Commit messages:** Use conventional commits format

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License - Copyright (c) 2025 Govind Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions...
```

## 🙏 Acknowledgments

- **Hugging Face Team** - For transformers library and model hosting
- **Distil* Team** - For DistilBERT architecture
- **Django Community** - For the excellent web framework
- **PyTorch Team** - For the deep learning framework
- **IMDB** - For the sentiment analysis dataset
- **AWS** - For cloud hosting infrastructure
- **Open Source Community** - For tools and inspiration

## 📞 Contact & Links

- **Author:** Govind Singh
- **GitHub:** [Repository Link](https://github.com/Govind2880/Visual_Attention)
- **Live Demo:** [AWS Elastic Beanstalk](http://visual-attention-app-env.eba-jjsphpfx.us-east-1.elasticbeanstalk.com)
- **Issues:** [GitHub Issues](https://github.com/Govind2880/Visual_Attention/issues)
---

<div align="center">

### ⭐ If this project helped you, please give it a star!

[![GitHub stars](https://img.shields.io/github/stars/Govind2880/Visual_Attention?style=social)](https://github.com/Govind2880/Visual_Attention/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Govind2880/Visual_Attention?style=social)](https://github.com/Govind2880/Visual_Attention/network/members)
[![GitHub issues](https://img.shields.io/github/issues/Govind2880/Visual_Attention)](https://github.com/Govind2880/Visual_Attention/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/Govind2880/Visual_Attention)](https://github.com/Govind2880/Visual_Attention/pulls)

**Built with ❤️ using AI, Django, and Modern Web Technologies**

*Transforming text into insights, one sentiment at a time* 🚀

</div>
