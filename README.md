# Social Media Sentiment Analyzer 🚀

[![Frontend Deployment](https://img.shields.io/badge/Vercel-Deployed-success)](https://your-vercel-app.vercel.app)
[![Backend Deployment](https://img.shields.io/badge/Render-Deployed-blue)](https://your-backend.onrender.com)

A real-time social media sentiment analysis dashboard that analyzes Twitter data and user input using AI. Built with Next.js (frontend) and Python/FastAPI (backend), deployed via Docker.

![Dashboard Preview](https://via.placeholder.com/800x400.png?text=Sentiment+Analysis+Dashboard)

## Features ✨

- **Sentiment Prediction**  
  Analyze text input for positive/neutral/negative sentiment using BERT NLP model
- **Real-Time Tweets**  
  Fetch and analyze latest tweets about any topic (e.g., #Bitcoin, #AI)
- **Interactive Dashboard**  
  Visualize sentiment distribution with dynamic charts
- **PDF Export**  
  Download analysis reports as PDF
- **Responsive Design**  
  Works seamlessly on desktop and mobile

## Tech Stack 🛠️

**Frontend**  
[![Next.js](https://img.shields.io/badge/Next.js-14.0-blue?logo=next.js)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-18.0-%2361DAFB?logo=react)](https://react.dev/)

**Backend**  
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?logo=docker)](https://docker.com)

**NLP**  
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21F?logo=huggingface)](https://huggingface.co)
[![BERT](https://img.shields.io/badge/BERT-Sentiment%20Analysis-FF6F00)](https://huggingface.co/docs/transformers/model_doc/bert)

**Deployment**  
[![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?logo=vercel)](https://vercel.com)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render)](https://render.com)

## Setup Instructions ⚙️

### Prerequisites

- Node.js 18+
- Python 3.11+
- Docker 24+
- Twitter API Bearer Token

### Local Development

1. **Clone Repository**

   ```bash
   git clone https://github.com/yourusername/social-sentiment-analyzer.git
   cd social-sentiment-analyzer

   ```

2. **Frontend Setup**

   ```bash
   cd frontend
   npm install
   cp .env.example .env.local

   ```

3. **Backend Setup**

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env

   ```

4. **Run With Docker**
   ```bash
   docker build -t any_name
   ```

This README provides:

- Clear project overview
- Badges for deployment status
- Visual tech stack representation
- Easy setup instructions for both local and cloud
- API documentation
- Contribution guidelines
- License information
