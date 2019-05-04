import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib

matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文

#传入的直方图要求是个字典，每个灰度对应着概率
def drawHist(hist,name):
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist)-1#x轴长度，也就是灰度级别
    axis_params = []
    axis_params.append(0)
    axis_params.append(x_size)

    #plt.figure()
    if name != None:
        plt.title(name)
    plt.bar(tuple(keys),tuple(values))#绘制直方图
    #plt.show()

#将灰度数组映射为直方图字典
def arrayToHist(grayArray,nums):
    if(len(grayArray.shape) != 2):
        print("length error")
        return None
    w,h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if(hist.get(grayArray[i][j]) is None):
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    #normalize
    n = w*h
    for key in hist.keys():
        hist[key] = float(hist[key])/n
    return hist

#计算累计直方图计算出新的均衡化的图片，nums为灰度数,256
def equalization(grayArray,h_s,nums):
    #计算累计直方图
    tmp = 0.0
    h_acc = h_s.copy()
    for i in range(256):
        tmp += h_s[i]
        h_acc[i] = tmp

    if(len(grayArray.shape) != 2):
        print("length error")
        return None
    w,h = grayArray.shape
    des = np.zeros((w,h),dtype = np.uint8)
    for i in range(w):
        for j in range(h):
            des[i][j] = int((nums - 1)* h_acc[grayArray[i][j] ] +0.5)
    return des

#直方图匹配函数，接受原始图像和目标灰度直方图
def histMatch(grayArray,h_d):
    #计算累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray,256)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    #计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if (np.fabs(h_acc[j] - h1_acc[i]) < minv):
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des

#给图像加噪声
def addNosie(grayArray,typee = "",ratio = 0.05):
    grayArray = grayArray.copy()
    w,h = grayArray.shape
    #print(w*h)
    if typee == "SAP":#椒盐噪声
        nums = int(w*h*ratio)
        pos = np.random.randint(0,w*h,nums)
        #print(len(pos))
        for i in range(nums):
            x = (int)(pos[i]/h)
            y = (int)(pos[i]%h)
            #print(str(x) + " "+str(y))
            if np.random.rand() > 0.5:
                grayArray[x][y] = 255
            else:
                grayArray[x][y] = 0
        return grayArray
    elif typee == "Gauss":
        noise = np.random.normal(0,20,(w,h))
        noise = noise.astype("float")
        grayArray = grayArray + noise
        grayArray = np.where(grayArray >  255,255,(np.where(grayArray<0,0,grayArray)))
        grayArray = grayArray.astype("uint8")
        return grayArray

#中值滤波
def middleFilter(array,size = 3,step = 1):
    arrayNew = array.copy()
    w,h = array.shape
    for i in range(0,w,step):
        for j in range(0,h,step):
            a = []
            for x in range(i - (size-1)//2,i + (size-1)//2):
                for y in range(j - (size-1)//2,j + (size-1)//2):
                    if x > -1 and x < w and y >-1 and y < h:
                        a.append(array[x][y])
            a.sort()
            arrayNew[i][j] = a[(len(a)-1)//2]
    return arrayNew

#边缘提取
def findContours(array,type = "Sobel"):
    contours = array.copy()
    w,h = contours.shape
    if type == "Sobel":
        for i in range(0,w):
            for j in range(0,h):
                if (i-1 >-1 and j-1>-1 and i+1<w and j+1<h):
                    sx = array[i-1][j-1] + 2*array[i][j-1] + array[i+1][j-1]\
                    -(array[i-1][j+1] + 2*array[i][j+1] + array[i+1][j+1])

                    sy = array[i+1][j-1] + 2*array[i+1][j] + array[i+1][j+1]\
                    -(array[i-1][j-1] + 2*array[i-1][j] + array[i-1][j+1])
                    contours[i][j] = (sx**2+sy**2)**0.5
        print("sobel")
        return contours
    elif type == "Roberts":
        for i in range(w):
            for j in range(h):
                if(i + 1<w and j + 1<h):
                    contours[i][j] = max(abs(array[i+1][j+1] - array[i][j])\
                    ,abs(array[i+1,j] - array[i][j + 1]))
        print("roberts")
        return contours
    else:
        for i in range(0,w):
            for j in range(0,h):
                if (i-1 >-1 and j-1>-1 and i+1<w and j+1<h):
                    sx = array[i-1][j-1] + array[i][j-1] + array[i+1][j-1]\
                    -(array[i-1][j+1] + array[i][j+1] + array[i+1][j+1])

                    sy = array[i+1][j-1] + array[i+1][j] + array[i+1][j+1]\
                    -(array[i-1][j-1] + array[i-1][j] + array[i-1][j+1])
                    contours[i][j] = (sx**2+sy**2)**0.5

        print("prewitt")
        return contours

#功能一，直方图均衡化
def Equal(imdir):
    #打开文件并灰度化
    im_s = Image.open(imdir).convert("L")
    im_s = np.array(im_s)
    print(np.shape(im_s))

    #开始绘图，分成四个部分
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(im_s,cmap = 'gray')
    plt.title("原始灰度图")
    #plt.show()

    #创建原始直方图
    plt.subplot(2,2,3)
    hist_s = arrayToHist(im_s,256)
    drawHist(hist_s,"原始直方图")

    #计算均衡化的新的图片，根据累计直方图
    im_d = equalization(im_s,hist_s,256)
    plt.subplot(2,2,2)
    plt.imshow(im_d,cmap="gray")
    plt.title("均衡的灰度图")

    #根据新的图片的数组，计算新的直方图
    plt.subplot(2,2,4)
    hist_d = arrayToHist(im_d,256)
    drawHist(hist_d,"均衡直方图")

    plt.show()

#功能二，直方图匹配，参数是两个图片的地址
def Match(imdir,imdir_match):
    #直方图匹配
    #打开文件并灰度化
    im_s = Image.open(imdir).convert("L")
    im_s = np.array(im_s)
    print(np.shape(im_s))
    #打开文件并灰度化
    im_match = Image.open(imdir_match).convert("L")
    im_match = np.array(im_match)
    print(np.shape(im_match))
    #开始绘图
    plt.figure()

    #原始图和直方图
    plt.subplot(2,3,1)
    plt.title("原始图片")
    plt.imshow(im_s,cmap='gray')

    plt.subplot(2,3,4)
    hist_s = arrayToHist(im_s,256)
    drawHist(hist_s,"原始直方图")

    #match图和其直方图
    plt.subplot(2,3,2)
    plt.title("match图片")
    plt.imshow(im_match,cmap='gray')

    plt.subplot(2,3,5)
    hist_m = arrayToHist(im_match,256)
    drawHist(hist_m,"match直方图")

    #match后的图片及其直方图
    im_d = histMatch(im_s,hist_m)#将目标图的直方图用于给原图做均衡，也就实现了match
    plt.subplot(2,3,3)
    plt.title("match后的图片")
    plt.imshow(im_d,cmap='gray')

    plt.subplot(2,3,6)
    hist_d = arrayToHist(im_d,256)
    drawHist(hist_d,"match后的直方图")

    plt.show()

#功能三，加噪声及中值滤波
def NosieAndFilter(imdir):
    im_s = Image.open(imdir).convert("L")
    im_s = np.array(im_s)

    hist_s = arrayToHist(im_s,256)

    im_d = equalization(im_s,hist_s,256)

    plt.figure()

    plt.subplot(2,3,1)
    plt.imshow(im_s,cmap="gray")
    plt.title("原图")

    plt.subplot(2,3,2)
    plt.imshow(im_d,cmap="gray")
    plt.title("均衡的灰度图")

    im_noise = addNosie(im_d,typee="SAP",ratio =0.03)
    plt.subplot(2,3,3)
    plt.imshow(im_noise,cmap="gray")
    plt.title("5%椒盐噪声")

    plt.subplot(2,3,4)
    a1 = middleFilter(im_noise,size=3,step = 1)
    plt.imshow(a1,cmap='gray')
    plt.title("3*3")

    plt.subplot(2,3,5)
    a3 = middleFilter(im_noise,size=7,step = 1)
    plt.imshow(a3,cmap='gray')
    plt.title("7*7")

    plt.subplot(2,3,6)
    a5 = middleFilter(im_noise,size=15,step = 1)
    plt.imshow(a5,cmap='gray')
    plt.title("15*15")

    plt.show()

#功能四，边缘提取
def Contour(imdir):
    im_s = Image.open(imdir).convert("L")
    im_s = np.array(im_s)

    im_noise =addNosie(im_s,typee='SAP',ratio =0.05)
    #边缘提取
    plt.figure()

    plt.subplot(2,3,1)
    plt.imshow(im_noise,cmap='gray')
    plt.title("带噪声的原图")

    plt.subplot(2,3,2)
    a3 = middleFilter(im_noise,size=5,step = 1)
    plt.imshow(a3,cmap='gray')
    plt.title("5*5中值滤波")

    b3 = findContours(a3,"Prewitt")
    plt.subplot(2,3,4)
    plt.imshow(b3,cmap='gray')
    plt.title('prewitt')
    b1 = findContours(a3,type="Sobel")
    plt.subplot(2,3,5)
    plt.imshow(b1,cmap='gray')
    plt.title("sobel")

    b2 = findContours(a3,type="Roberts")
    plt.subplot(2,3,6)
    plt.imshow(b2,cmap='gray')
    plt.title("Roberts")

    

    plt.show()

'''
dir1 = "./hw1_s.jpg"
Equal(dir1)

match1 = "./hw1_s2.jpg"
match2 = "./hw1_s22.jpg"
Match(match1,match2)
'''
dir4 = './bjt.jpg'

NosieAndFilter(dir4)

#Contour(dir4)
