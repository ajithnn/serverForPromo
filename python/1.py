#!/app/python/py/bin/python
from PIL import Image
import sys

Img = Image.open(sys.argv[1] + '/assets/CurrentImg.jpg').convert('L')
Img.save(sys.argv[1] + '/assets/CurrentImg.jpg',"jpeg")
