# XIM Chatbot

## Setup Instructions

### 1. Clone repo
git clone <repo_url>
cd XIM-Chatbot

### 2. Create virtual environment
python -m venv venv

### 3. Activate virtual environment

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### 4. Install dependencies
pip install -r requirements.txt

### 5. Set API key

Windows:
set NVIDIA_API_KEY=your_key

Mac/Linux:
export NVIDIA_API_KEY=your_key

Or use .env file.

### 6. Run server
uvicorn api:app --reload

### 7. Navigate localhost
change directory to where your front end files are located inside XIM-Chatbot

python -m http.server 3000

Server runs at:
http://127.0.0.1:8000

