# Vocalis: Real-Time AI Communication Coach

Vocalis is a real-time AI voice coach that live-analyzes your pacing, energy, and presence to deliver targeted nudges that measurably improve your delivery in high-stakes professional scenarios.

Built for the AWS Hackathon on Devpost.

## 🚀 Features

* **Real-Time Telemetry:** Analyzes streaming audio frame-by-frame via WebSockets to instantly score Pace, Energy (Vocal Vitality), and Delivery Presence (Command & Composure).
* **Targeted AI Nudges:** Isolates the single biggest delivery flaw and provides an actionable, in-the-moment correction (e.g., "Slow down to stay calm and clear").
* **Multi-Attempt Loop:** Tracks delta scores across retries, visually proving your real-time improvement on the dashboard.
* **100% Local Inference:** Powered by the Strands SDK and local Qwen 2.5 (7B) via Ollama, ensuring absolute privacy, zero cloud reliance for audio, and sub-3-second response times.
* **Premium Dashboard:** A sleek, dark-mode SaaS UI built with vanilla JS/HTML/CSS for a smooth, distraction-free coaching experience.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, WebSockets
* **Agent Logic:** Strands SDK
* **Local LLM:** Ollama (Qwen 2.5 7B)
* **Frontend:** Vanilla JavaScript, HTML5, CSS3

## 💻 Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone git@github.com:officialjeff9/vocalis.git
   cd vocalis
