import glob
import cv2
import numpy as np
import pickle
import xml.etree.ElementTree as ET
import re

def calc_hog(pos_img_paths, neg_img_paths):
    winSize = (64, 32)
    blockSize = (4, 4)
    blockStride = (4, 4)
    cellSize = (4, 4)
    nbins = 9
    # derivAperture = 1
    # winSigma = 4.
    # histogramNormType = 0
    # L2HysThreshold = 2.0000000000000001e-01
    # gammaCorrection = 0
    # nlevels = 64

    # HOG特徴の計算とラベリング
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    train = []
    label = []
    # 正解画像・非正解画像のHOG特徴を計算してラベルを付け
    for path in pos_img_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, winSize)
        train.append(hog.compute(img))
        label.append(1)

    for path in neg_img_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, winSize)
        train.append(hog.compute(img))
        label.append(0)

    return np.array(train), np.array(label, dtype=int)


def do_training():
    pos_img_paths = glob.glob('train_data/pos/*')
    neg_img_paths = glob.glob('train_data/neg/*')
    train, label = calc_hog(pos_img_paths, neg_img_paths)

    # Hog特徴からSVM識別器の作成
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(0.01)
    svm.train(train, cv2.ml.ROW_SAMPLE, label)
    svm.save('hog_train.xml')

    tree = ET.parse('hog_train.xml')
    root = tree.getroot()
    # now this is really dirty, but after ~3h of fighting OpenCV its what happens :-)
    SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
    rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
    svmvec = [float(x) for x in re.sub('\s+', ' ', SVs.text).strip().split(' ')]
    svmvec.append(-rho)
    pickle.dump(svmvec, open("svm.pickle", 'wb'))


def detect():
    win_size = (64, 32)
    block_size = (4, 4)
    block_stride = (4, 4)
    cell_size = (4, 4)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    svm = pickle.load(open("svm.pickle", 'rb'))
    hog.setSVMDetector(np.array(svm))

    test_img_paths = glob.glob('test/*')
    for i, path in enumerate(test_img_paths):
        img = cv2.imread(path)
        height, width, channels = img.shape[:3]
        img = cv2.resize(img, (int(width/2), int(height/2)))
        hogParams = {'winStride': (8, 8)}
        # rects, weights = hog.detectMultiScale(img, (4, 4), (8, 8), 1.05)
        rects, weights = hog.detectMultiScale(img, **hogParams)

        for (x, y, w, h) in rects:
            print(x, ",", y, ",", w, ",", h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        print(path + " was processed")
        cv2.imwrite("results/" + str(i) + ".png", img)


def main():
    print("Now do training ...")
    do_training()
    print("it has been finished")
    print("Now do detect ...")
    detect()
    print("application will be shut down")


if __name__ == '__main__':
    main()

# HOG特徴量の計算
# def calc_hog(img_paths, bin_n=32)
#     hists = []
#     for path in img_paths:
#         # 画像をフレースケールで読み込み
#         gray = cv2.imread(path, 0)
#         # 縦・横方向のエッジ画像を生成
#         gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
#         gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
#         # エッジ勾配の角度と大きさを算出
#         mag, ang = cv2.cartToPolar(gx, gy)
#         # 勾配方向の量子化(16方向)
#         bins = np.int32(bin_n*ang/(2*np.pi))
#         # 勾配方向ヒストグラムを計算
#         hist = np.bincount(bins.ravel(), mag.ravel(), bin_n)
#         hists.append(hist)
#
#     return np.array(hists, np.float32)

# def train():
#     # 正解画像・非正解画像のファイルパスを取得
#     pos_img_paths = glob.glob('pos/*')
#     neg_img_paths = glob.glob('neg/*')
#
#     # HOG特徴量の計算
#     pos_hogs = calc_hog(pos_img_paths)
#     neg_hogs = calc_hog(neg_img_paths)
#     hogs = np.r_[pos_hogs, neg_hogs]
#
#     # ラベル用配列の生成( 正解:1, 非正解:0 )
#     pos_labels = np.ones(len(pos_img_paths), np.int32)
#     neg_labels = np.zeros(len(neg_img_paths), np.int32)
#     labels = np.array([np.r_[pos_labels, neg_labels]])
#
#     # HOG特徴量をSVMで学習
#     svm = cv2.ml.SVM_create()
#     svm.setType(cv2.ml.SVM_C_SVC)
#     svm.setKernel(cv2.ml.SVM_RBF)
#     #svm.setDegree(0.0)
#     svm.setGamma(5.4)
#     svm.setC(2.7)
#     #svm.setNu(0.0)
#     #svm.setP(0.0)
#     #svm.setClassWeights(None)
#     svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
#     svm.train(hogs, cv2.ml.ROW_SAMPLE, labels)
#     svm.save('hog_train.xml')


