# Quick Start - Free Deepfake Detector

## 🚀 Get Started in 3 Steps

### Step 1: Start the Backend

Open a terminal and run:

**Windows:**
```bash
cd backend
start_free.bat
```

**Mac/Linux:**
```bash
cd backend
chmod +x start_free.sh
./start_free.sh
```

Wait for: `✅ API ready!`

### Step 2: Start the Frontend

Open another terminal and run:

```bash
npm run dev
```

Wait for: `✓ Ready in X.Xs`

### Step 3: Test It!

1. Open your browser: `http://localhost:3000/deepfake-check`
2. Upload any image (JPG, PNG, or WebP)
3. Click "Analyze for Deepfakes"
4. View results!

## 📝 Optional: Environment Variables

Create a file named `.env.local` in the root directory:

```bash
# Backend URL (default works locally)
FREE_DEEPFAKE_API_URL=http://localhost:5000

# Optional: Google reverse image search (100 free queries/day)
GOOGLE_CSE_API_KEY=your_key_here
GOOGLE_CSE_ID=your_id_here
```

**Note**: Reverse image search works WITHOUT API keys - it just opens Google/TinEye/Yandex in a new tab.

## ✅ What You Get

- **Free Detection**: No API costs, all runs locally
- **Image Analysis**: Checks for AI-generated or manipulated content
- **Clear Results**: Simple verdict + explanation + confidence score
- **Reverse Search**: Links to find similar images online
- **Privacy-First**: No third-party APIs (unless you add Google CSE)

## 🔧 Troubleshooting

**Backend won't start?**
```bash
cd backend
pip install -r requirements_free.txt
python deepfake_free.py
```

**Frontend error?**
```bash
npm install
npm run dev
```

**Connection refused?**
- Make sure backend is running on port 5000
- Check: `http://localhost:5000/health`

## 📖 Full Documentation

See `DEEPFAKE_DETECTOR_README.md` for complete details.

## 🎯 Test Images

Try uploading:
- ✅ Regular photos from your phone
- ✅ AI-generated images (from DALL-E, Midjourney, etc.)
- ✅ Screenshots
- ✅ Social media images

The system will analyze and give you a verdict!

---

**That's it!** You now have a free, open-source deepfake detector running on your machine. 🎉

