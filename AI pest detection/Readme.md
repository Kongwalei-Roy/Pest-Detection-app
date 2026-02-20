# AI Pest Detection System (Computer Vision)

End-to-end deep learning pest detection:
- Train a lightweight classifier (MobileNetV3)
- Evaluate accuracy + confusion matrix
- Export to TensorFlow Lite
- Serve predictions via FastAPI
- Simple web demo UI

## Folder Structure
data/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg

## 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate   # windows
pip install -r requirements.txt
