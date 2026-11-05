# Sudarshana — Cloud & Production Deployment Guide

This guide explains how **Sudarshana** runs native forensic tools (`stegseek`, `steghide`, `zsteg`, `binwalk`, `exiftool`) in production so that **any user in the world** can access the platform via their browser (desktop or mobile) without needing Kali Linux or Docker installed on their local machine.

---

## 🏗️ Architectural Overview

```
 [User Browser (Chrome/Safari/Mobile)]
                │
                │ HTTPS (Port 443 / Port 8501)
                ▼
 [Cloud Server: Docker Container (Debian 12 Bookworm)]
 ┌────────────────────────────────────────────────────────┐
 │  Sudarshana Web Platform (Streamlit / Python 3.11)     │
 │                                                        │
 │  ┌──────────────────────────────────────────────────┐  │
 │  │ 1. AI Steganalysis Ensemble (PyTorch Multi-CNN)   │  │
 │  │    (Weights: fusion_best.pt / branch_b_best.pt)  │  │
 │  ├──────────────────────────────────────────────────┤  │
 │  │ 2. Automated CTF & Flag Hunter (Universal Regex)  │  │
 │  │    (Arbitrary Regex, Key Harvesting, Polyglots)  │  │
 │  ├──────────────────────────────────────────────────┤  │
 │  │ 3. Isolated Subprocess Workers (Sandboxed):       │  │
 │  │    • zsteg (Ruby Gem)                            │  │
 │  │    • steghide (C++ native)                       │  │
 │  │    • stegseek (Ultra-fast rockyou GPU/CPU)       │  │
 │  │    • binwalk & exiftool                          │  │
 │  │    • /usr/share/wordlists/rockyou.txt            │  │
 │  └──────────────────────────────────────────────────┘  │
 └────────────────────────────────────────────────────────┘
```

When a user visits your deployed URL:
1. They upload an image directly through their browser.
2. The cloud container receives the file into a memory buffer / secure ephemeral temp directory (`/tmp/stego_XXXXXX`).
3. The server runs both the AI model and the sandboxed CLI subprocesses (`zsteg`, `stegseek`, `steghide`).
4. Any discovered flags (matching `.....{......}` or custom regex), carved polyglot files, and structural anomalies are sent back to the browser in real time.
5. Ephemeral temp files are destroyed immediately to preserve privacy and confidentiality.

---

## 🚀 Deployment Options

### Option 1: Hugging Face Spaces (Free & Instant)
Hugging Face Spaces is the most popular platform for hosting AI and cybersecurity apps. It provides free CPU/GPU compute, free HTTPS, and native Docker support.

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your space (e.g., `sudarshana-forensics`), select **Docker** as the SDK, and choose **Blank**.
3. Clone your new space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/sudarshana-forensics
   cd sudarshana-forensics
   ```
4. Copy the project files into the folder and create a `README.md` with frontmatter:
   ```yaml
   ---
   title: Sudarshana AI Forensics & CTF Solver
   emoji: 🔵
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 8501
   pinned: false
   ---
   ```
5. Commit and push:
   ```bash
   git add .
   git commit -m "Deploy Sudarshana unified image"
   git push
   ```
6. Hugging Face will automatically build your Docker container, install all tools (`zsteg`, `steghide`, `stegseek`), and provide a live public URL (e.g., `https://huggingface.co/spaces/<your-username>/sudarshana-forensics`).

---

### Option 2: VPS (DigitalOcean / Hetzner / AWS EC2 / Linode) — Recommended for Full Control
For \$4–\$6/month, you get a dedicated Linux VPS with full access to run unlimited wordlist attacks and high-speed processing.

1. **Launch an Ubuntu 22.04 / 24.04 droplet/instance.**
2. **Install Docker & Docker Compose:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
   apt-get install -y docker-compose-plugin
   ```
3. **Clone your repository:**
   ```bash
   git clone https://github.com/itz-mridul/EDI_Project.git /opt/sudarshana
   cd /opt/sudarshana
   ```
4. **Start the platform with Docker Compose:**
   ```bash
   docker compose up -d --build
   ```
5. **Set up Nginx & Free SSL (Let's Encrypt):**
   ```bash
   apt install -y nginx certbot python3-certbot-nginx
   ```
   Add a reverse proxy in `/etc/nginx/sites-available/sudarshana`:
   ```nginx
   server {
       server_name forensics.yourdomain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_read_timeout 86400;
       }
   }
   ```
   Enable site and obtain SSL certificate:
   ```bash
   ln -s /etc/nginx/sites-available/sudarshana /etc/nginx/sites-enabled/
   certbot --nginx -d forensics.yourdomain.com
   ```
   Now anyone can access your site securely at `https://forensics.yourdomain.com`!

---

### Option 3: Serverless Container (Google Cloud Run / AWS App Runner)
Ideal if you want auto-scaling with **zero costs when idle** (scales down to 0 instances):

1. **Build and push image to Google Artifact Registry:**
   ```bash
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/sudarshana:latest
   ```
2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy sudarshana \
     --image gcr.io/<PROJECT_ID>/sudarshana:latest \
     --platform managed \
     --port 8501 \
     --memory 2Gi \
     --cpu 2 \
     --allow-unauthenticated
   ```
Google Cloud Run will generate a public HTTPS URL automatically.
