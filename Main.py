import glob
import cv2
import numpy as np


def compute_hog(hog, images, features):
    count = 0
    for img in images:
        dst = cv2.resize(img, hog.winSize)
        features.append(hog.compute(dst))
        count += 1

    print('count = ', count)
    return features


def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


def read_samples(folder_name):
    images = []
    img_paths = glob.glob(folder_name)
    for path in img_paths:
        src = cv2.imread(path)
        images.append(src)

    return images


def get_features(hog, features, labels):
    pos_images = read_samples('train_data/pos/*')
    compute_hog(hog, pos_images, features)
    [labels.append(1) for _ in range(len(pos_images))]

    neg_images = read_samples('train_data/neg/*')
    compute_hog(hog, neg_images, features)
    [labels.append(-1) for _ in range(len(neg_images))]

    return features, labels


def svm_train(svm, features, labels):
    svm.train(np.array(features), cv2.ml.ROW_SAMPLE, np.array(labels))


def hog_train(svm):
    features = []
    labels = []

    win_size = (120, 60)
    block_size = (16, 16)
    block_stride = (4, 4)
    cell_size = (4, 4)
    bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)

    # get hog features
    get_features(hog, features, labels)

    # svm training
    print('svm training...')
    svm_train(svm, features, labels)
    print('svm training complete...')

    hog.setSVMDetector(get_svm_detector(svm))
    hog.save('hog_detector.bin')


def svm_config():
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0)
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)
    svm.setC(0.01)
    svm.setType(cv2.ml.SVM_EPS_SVR)

    return svm


def train():
    # svm config
    svm = svm_config()
    # hog training
    hog_train(svm)


def detect():
    hog = cv2.HOGDescriptor()
    hog.load('hog_detector.bin')

    test_img_paths = glob.glob('test/*')
    for i, path in enumerate(test_img_paths):
        img = cv2.imread(path)
        rects, scores = hog.detectMultiScale(img, winStride=(8, 8), padding=(0, 0), scale=1.05)

        for (x, y, w, h) in rects:
            print(x, ",", y, ",", w, ",", h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        print(path + " was processed")
        cv2.imwrite("results/" + str(i) + ".png", img)


def main():
    print("Now do training ...")
    train()
    print("it has been finished")
    print("Now do detect ...")
    detect()
    print("application will be shut down")


if __name__ == '__main__':
    main()
