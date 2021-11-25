import requests

import json
file_data = {'image1': open(r'C:\Users\qq203\Desktop\image_2.jpg', 'rb'),
             'image2': open(r'C:\Users\qq203\Desktop\bike_020.png', 'rb')}
user_info = {'cnt': 2, 'type': 'pic', 'info': 'hsj'}
r = requests.post("http://127.0.0.1:5000/upload", data=user_info, files=file_data)
print(r.text)
