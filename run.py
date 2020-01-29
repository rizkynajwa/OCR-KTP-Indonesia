import cv2
import pytesseract
import os
import textdistance
import numpy as np
import pandas as pd
from operator import itemgetter, attrgetter

ROOT_PATH = os.getcwd()
IMAGE_PATH = os.path.join(ROOT_PATH, 'Kywa.jpg')
LINE_REC_PATH = os.path.join(ROOT_PATH, 'data/ID_CARD_KEYWORDS.csv')
CITIES_REC_PATH = os.path.join(ROOT_PATH, 'data/CITIES.csv')
RELIGION_REC_PATH = os.path.join(ROOT_PATH, 'data/RELIGIONS.csv')
MARRIAGE_REC_PATH = os.path.join(ROOT_PATH, 'data/MARRIAGE_STATUS.csv')
NEED_COLON = [3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21]
NEXT_LINE = 9
ID_NUMBER = 3


def ocr_raw(image_path):
    # (1) Read
    img_raw = cv2.imread(image_path)
    image = cv2.resize(img_raw, (50 * 16, 500))
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    id_number = return_id_number(image, img_gray)
    # img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    # (2) Threshold
    cv2.fillPoly(img_gray, pts=[np.asarray([(540, 150), (540, 499), (798, 499), (798, 150)])], color=(255, 255, 255))
    th, threshed = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    # (3) Detect
    result_raw = pytesseract.image_to_string(threshed, lang="ind")
    return result_raw, id_number

def strip_op(result_raw):
    result_list = result_raw.split('\n')
    new_result_list = []
    for tmp_result in result_list:
        if tmp_result.strip(' '):
            new_result_list.append(tmp_result)
    return new_result_list

# NIK_NUMBER
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


def return_id_number(image, gray):
    # ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    # print (np.array(gradX).shape)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)
    threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = threshCnts
    cur_img = image.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
    copy = image.copy()
    locs = []

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        # ar = w / float(h)
        # if ar >10:
        # if (w > 40 ) and (h > 10 and h < 20):
        if h > 10 and w > 100 and x < 300:
            img = cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            locs.append((x, y, w, h, w * h))
    locs = sorted(locs, key=itemgetter(1), reverse=False)

    # print(locs[1][0])
    nik = image[locs[1][1] - 10:locs[1][1] + locs[1][3] + 10, locs[1][0] - 10:locs[1][0] + locs[1][2] + 10]
    # cv_show('nik', nik)

    img = cv2.imread("module.png")
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 66, 255, cv2.THRESH_BINARY_INV)[1]
    refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
    # cv2.imshow('888', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    refCnts = sort_contours(refCnts, method="left-to-right")[0]
    digits = {}

    for (i, c) in enumerate(refCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        # if
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        digits[i] = roi
    # cv_show('digits[i]', digits[i])
    # nik = np.clip(nik, 0, 255)
    # nik = np.array(nik,np.uint8)

    gray_nik = cv2.cvtColor(nik, cv2.COLOR_BGR2GRAY)
    # gray_nik = cv2.GaussianBlur(gray_nik, (3, 3), 0)
    # ret_nik, thresh_nik = cv2.threshold(gray_nik, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('8', gray_nik)
    # cv2.waitKey(0)
    group = cv2.threshold(gray_nik, 127, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow('9', group)

    digitCnts, hierarchy_nik = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    nik_r = nik.copy()

    cv2.drawContours(nik_r, digitCnts, -1, (0, 0, 255), 3)

    gX = locs[1][0]
    gY = locs[1][1]
    gW = locs[1][2]
    gH = locs[1][3]

    ctx = sort_contours(digitCnts, method="left-to-right")[0]

    locs_x = []

    for (i, c) in enumerate(ctx):

        (x, y, w, h) = cv2.boundingRect(c)


        if h > 10 and w > 10:
            img = cv2.rectangle(nik_r, (x, y), (x + w, y + h), (0, 255, 0), 2)
            locs_x.append((x, y, w, h))

    # digitCnts = sort_contours(digitCnts, method="left-to-right")[0]
    output = []
    groupOutput = []

    for c in locs_x:

        (x, y, w, h) = c
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        # cv_show('roi',roi)

        scores = []

        for (digit, digitROI) in digits.items():

            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)


        groupOutput.append(str(np.argmax(scores)))

    cv2.rectangle(image, (gX - 5, gY - 5),
                  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    output.extend(groupOutput)
    return ''.join(output)


def main():
    raw_df = pd.read_csv(LINE_REC_PATH, header=None)
    cities_df = pd.read_csv(CITIES_REC_PATH, header=None)
    religion_df = pd.read_csv(RELIGION_REC_PATH, header=None)
    marriage_df = pd.read_csv(MARRIAGE_REC_PATH, header=None)
    result_raw, id_number = ocr_raw(IMAGE_PATH)
    result_list = strip_op(result_raw)

    print('-------origin rec-------')
    print(result_list)
    loc2index = dict()
    for i, tmp_line in enumerate(result_list):
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word_, tmp_word.strip(':')) for
                            tmp_word_ in
                            raw_df[0].values]
            tmp_sim_np = np.asarray(tmp_sim_list)
            arg_max = np.argmax(tmp_sim_np)
            if tmp_sim_np[arg_max] >= 0.6:
                loc2index[(i, j)] = arg_max

    print('--------processed------------')
    last_result_list = []
    useful_info = False
    for i, tmp_line in enumerate(result_list):
        tmp_list = []
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_word = tmp_word.strip(':')
            if (i, j) in loc2index:
                useful_info = True
                if loc2index[(i, j)] == NEXT_LINE:
                    last_result_list.append(tmp_list)
                    tmp_list = []
                tmp_list.append(raw_df[0].values[loc2index[(i, j)]])
                if loc2index[(i, j)] in NEED_COLON:
                    tmp_list.append(':')
            elif tmp_word == ':' or tmp_word == '':
                continue
            else:
                tmp_list.append(tmp_word)
        if useful_info:
            if len(last_result_list) > 2 and ':' not in tmp_list:
                last_result_list[-1].extend(tmp_list)
            else:
                last_result_list.append(tmp_list)
    # print(last_result_list)

    for tmp_data in last_result_list:
        if '—' in tmp_data:
            tmp_data.remove('—')
        if 'PROVINSI' in tmp_data or 'KABUPATEN' in tmp_data or 'KOTA' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for
                                tmp_word_ in cities_df[0].values]
                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)
                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 1] = cities_df[0].values[arg_max]

        if 'NIK' in tmp_data:
            if len(id_number) != 16:
                id_number = tmp_data[2]
                if "D" in id_number:
                    id_number = id_number.replace("D", "0")
                if "?" in id_number:
                    id_number = id_number.replace("?", "7")
                if "L" in id_number:
                    id_number = id_number.replace("L", "1")
                while len(tmp_data) > 2:
                    tmp_data.pop()
                tmp_data.append(id_number)
            else:
                while len(tmp_data) > 3:
                    tmp_data.pop()
                tmp_data[2] = id_number

        if 'Agama' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for
                                tmp_word_ in religion_df[0].values]
                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)
                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 1] = religion_df[0].values[arg_max]

        if 'Status' in tmp_data or 'Perkawinan' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[2:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for
                                tmp_word_ in marriage_df[0].values]
                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)
                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 2] = marriage_df[0].values[arg_max]
        if 'Alamat' in tmp_data:
            for tmp_index in range(len(tmp_data)):
                if "!" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "I")
                if "1" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("1", "I")
                if "i" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "I")
    for tmp_data in last_result_list:
        print(' '.join(tmp_data))


if __name__ == '__main__':
    main()
