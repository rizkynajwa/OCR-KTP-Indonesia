# OCR-KTP-Indonesia

### Prerequisites
* Flask
```
pip install flask
```
* Numpy
```
pip install numpy
```
* OpenCV
```
pip install opencv-contrib-python
```
* Pandas
```
pip install flask
```
* PyTesseract
```
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-ind
pip install pytesseract
```
* PIL
```
pip install pillow
```
* TextDistance
```
pip install textdistance
```

## Running the Program
To run the program, use command below:
```
export FLASK_APP=api.py
flask run
```
or alternatively using this command:
```
python api.py
```

### Notes for KTP Detection using YOLO
1. Uncomment on line 7 and 29 in ```api.py```
2. Download trained weights [here]() and put in data/yolo/ folder
3. Change file name on line 13 in ```yolo_detect.py```
4. Run the program

## Acknowledgments
* https://github.com/enningxie/KTP-OCR
* https://github.com/jeffreyevan/OCR
