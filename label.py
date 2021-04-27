from bs4 import BeautifulSoup
import requests
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
import openpyxl

N = 1 # label number

def denoise(img,filename):
    image_gaussian_compare = np.hstack([
        cv2.GaussianBlur(img, (3, 3), 0),
    ])
    ret, th1 = cv2.threshold(image_gaussian_compare, 130, 255, cv2.THRESH_BINARY)
    cv2.imwrite(filename, th1)
    return th1


if __name__ == '__main__':
    df = pd.DataFrame(columns=['filename','label'])
    
    link = "https://course.ncku.edu.tw/index.php?c=verifycode&0.7103330270402433"  # 取得圖片來源連結
    for i in range(0, N):
        if not os.path.exists("train"):
            os.mkdir("train")  # 建立資料夾
        
        img = requests.get(link)  # 下載圖片
        now = datetime.now()
        dt_string = now.strftime("%m-%d-%H%M%S")

        filename = "train/" + dt_string + ".jpg"
        with open(filename, "wb") as file:  # 開啟資料夾及命名圖片檔
            file.write(img.content)  # 寫入圖片的二進位碼
        img = cv2.imread(filename,0)
        img = denoise(img,filename)
        cv2.imwrite(filename, img)
        cv2.imshow('result', img)
        cv2.waitKey(0)
        code = input("code: ")
        data = [filename,code]
        df = df.append(pd.Series(data, index=['filename','label']), ignore_index=True)
    print(df)

    df.to_csv('data.csv', mode='a',index = None,header=0)
