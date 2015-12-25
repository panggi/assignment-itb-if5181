#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

from flask import Flask, render_template, request
from logging import Formatter, FileHandler
from PIL import Image
from scipy import misc
from werkzeug import secure_filename
from functools import wraps, update_wrapper
from datetime import datetime
from numpy import *
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import uuid
import cv2
import random

#----------------------------------------------------------------------------#
# Constants
#----------------------------------------------------------------------------#
UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG'])

#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
# app = Flask(__name__, static_url_path = "", static_folder = "static")
app.config.from_object('config')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('pages/placeholder.home.html')


@app.route('/histogram', methods=['GET', 'POST'])
def histogram():
    if request.method == 'GET':
        return render_template('pages/placeholder.histogram.html')

    if request.method == 'POST':
        random_char = str(uuid.uuid4())
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'histogram_image_source_' + random_char))

        hist_source_img = Image.open(os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/histogram_image_source_' + random_char)
        hist_source_img = list(hist_source_img.getdata())

        hist_r = [0]*256
        hist_g = [0]*256
        hist_b = [0]*256

        for pixel in hist_source_img:
          hist_r[pixel[0]] += 1
          hist_g[pixel[1]] += 1
          hist_b[pixel[2]] += 1

        x = range(len(hist_r))
        plt.plot(x,hist_r,'r', x,hist_g,'g', x,hist_b,'b')
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/histogram_' + random_char + '.png')
        plt.close()

        return render_template('pages/placeholder.histogram.post.html', random_char=random_char)

@app.route('/chaincode', methods=['GET', 'POST'])
def chaincode():
    global chaincode_source_img
    global chaincode_bwim
    global binary
    global chaincode_bw
    global point
    global firstpix
    global alphanumeric

    alphanumeric = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0','1', '2',
        '3', '4', '5', '6', '7', '8', '9']

    random_char_chaincode = str(uuid.uuid4())

    if request.method == 'GET':
        font_image = '/static/chaincode/H.jpg'
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'chaincode_' + random_char_chaincode))
        font_image = '/static/uploads/chaincode_' + random_char_chaincode

    chaincode_dic = open(os.path.dirname(os.path.realpath(__file__)) + '/static/dictionary/chaincode_arial.txt', 'r')
    belok_dic = open(os.path.dirname(os.path.realpath(__file__)) + '/static/dictionary/kode_belok_arial.txt', 'r')
    dictionary_cc = np.loadtxt(chaincode_dic, dtype = str, delimiter='||')
    dictionary_kb = np.loadtxt(belok_dic, dtype = str, delimiter='||')

    chaincode_source_img = misc.imread(os.path.dirname(os.path.realpath(__file__)) + font_image)
    chaincode_bwim = (0.2989 * chaincode_source_img[:,:,0] + 0.587 * chaincode_source_img[:,:,1] + 0.114 * chaincode_source_img[:,:,2]).astype(np.uint8) #grayscale
    binary =  chaincode_bwim < 128 #blackwhite
    chaincode_bw = np.argwhere(binary)[0] #return index dari array hasil dari operasi boolean/ index dimana chaincode_bwim < 128 rubah jadi list biasa
    point = (chaincode_bw - (0,1)).tolist()
    firstpix = chaincode_bw.tolist()
    chaincode = get_chaincode()
    kodebelok = kode_belok(chaincode)

    classification = classify(chaincode, dictionary_cc, kodebelok, dictionary_kb)

    if request.method == 'GET':
        return render_template('pages/placeholder.chaincode.html', chaincode=chaincode, kodebelok=kodebelok, classification=classification)

    if request.method == 'POST':
        return render_template('pages/placeholder.chaincode.post.html', chaincode=chaincode, kodebelok=kodebelok, random_char_chaincode=random_char_chaincode, classification=classification)

def get_direction(firstpix, point):
    dir = 0
    row = firstpix[0]
    col = firstpix[1]
    if point == [row, col+1]:
        dir = 0
    if point == [row-1, col+1]:
        dir = 1
    if point == [row-1, col]:
        dir = 2
    if point == [row-1, col-1]:
        dir = 3
    if point == [row, col-1]:
        dir = 4
    if point == [row+1, col-1]:
        dir = 5
    if point == [row+1, col]:
        dir = 6
    if point == [row+1, col+1]:
        dir = 7

    return dir

def get_index(dir, firstpix):
    row = firstpix[0]
    col = firstpix[1]
    if dir == 0:
        grid = [row, col+1]
    if dir == 1:
        grid = [row-1, col+1]
    if dir == 2:
        grid = [row-1, col]
    if dir == 3:
        grid = [row-1, col-1]
    if dir == 4:
        grid = [row, col-1]
    if dir == 5:
        grid = [row+1, col-1]
    if dir == 6:
        grid = [row+1, col]
    if dir == 7:
        grid = [row+1, col+1]

    return grid

def get_chaincode():
    chainCode = []
    curInd = firstpix
    backtrack = point
    flag = copy.copy(curInd)
    flagstat = False

    while flagstat == False:
        pixVal = binary[backtrack[0],backtrack[1]]
        direction = get_direction(curInd,backtrack)

        while pixVal != True:
            direction = (direction+1) % 8
            newpixel = get_index(direction, curInd)
            pixVal = binary[newpixel[0],newpixel[1]]

            if pixVal == False:
                backtrack = newpixel
            else:
                curInd = newpixel

        chainCode.append(direction)
        # print direction
        if curInd == flag:
            flagstat = True
    strcc = ''.join(str(e) for e in chainCode)
    return strcc


def kode_belok(code):
    belok = ""
    for i in range(0,len(code)-1):
        dir = int(code[i])
        tambah = (dir+4)%8
        next = int(code[i+1])
        if next == (dir+1)%8 or next == (dir+2)%8 or next == (dir+3)%8:
            belok = belok + '-'
        elif next == (tambah+1)%8 or next==(tambah+2)%8 or next==(tambah+3)%8:
            belok = belok + '+'
        else:
            continue
    return belok

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def classify(cc, dictionaryCC, belok, dictionaryKB):
    plateNum = ""
    # iterate every feature extracted from test picture
    # plateCC = list of chain codes from test picture
    # plateKB = list of kode belok from test picture
    # dictionaryXX = list of features from training picture
    # for every index in plateCC, iterate this:

    matchScore = [] # store total matching score
    matchScoreCC = [] # store matching score according to chain code
    matchScoreKB = [] # store matching score according to kode belok

        # compare every feature with feature in dictionary
        # for every index in dictionaryCC, iterate this:
    for j in xrange(len(dictionaryCC)):

        # for every chain code in dictionary, calculate its
        # levenshtein distance
        # save it to matchScoreCC
        distanceCC = levenshtein(cc, dictionaryCC[j])
        matchScoreCC.append(distanceCC)

        # do the same thing with kode belok
        # save it to matchScoreKB
        distanceKB = levenshtein(belok, dictionaryKB[j])
        matchScoreKB.append(distanceKB)

        # sum each score in matchScoreCC with its matchScoreKB counterpart
        score = matchScoreCC[j]+matchScoreKB[j]
        matchScore.append(score)

    chosen = matchScore.index(min(matchScore))
    plateNum += alphanumeric[chosen]
    return plateNum

@app.route('/thinning', methods=['GET', 'POST'])
def thinning():
    global thinning_source_img
    global thinning_bw
    global random_char_thinning

    random_char_thinning = str(uuid.uuid4())

    if request.method == 'GET':
        font_image = '/static/thinning/B_comic.jpg'
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'thinning_' + random_char_thinning))
        font_image = '/static/uploads/thinning_' + random_char_thinning

    thinning_source_img = misc.imread(os.path.dirname(os.path.realpath(__file__)) + font_image)
    thinning_bw = np.zeros((thinning_source_img.shape[0], thinning_source_img.shape[1]))
    get_bw()

    tulang = zhang_suen(thinning_bw)
    smpg = simpang(tulang)
    ujg = ujung(tulang)
    training_data = retrieve_data_training()
    pengenalan = testing_data(training_data,ujg,smpg)

    if request.method == 'GET':
        return render_template('pages/placeholder.thinning.html', pengenalan=pengenalan)

    if request.method == 'POST':
        thinning = zhang_suen(thinning_bw, 'post')
        return render_template('pages/placeholder.thinning.post.html', thinning=thinning, random_char_thinning=random_char_thinning, pengenalan=pengenalan)

def get_bw():
    for row in xrange(thinning_source_img.shape[0]):
        for col in xrange(thinning_source_img.shape[1]):
            if(np.sum(thinning_source_img[row][col]))/3 > 128:
                thinning_bw[row][col] = 0
            else:
                thinning_bw[row][col] = 1

def zhang_suen(obj, method='GET'):
    print obj.shape
    erase = [0]
    while erase:
        erase = []
        for row in range(1, obj.shape[0]-1):
            for col in xrange(1, obj.shape[1]-1):
                neighbors = [obj[row-1][col], obj[row-1][col+1], obj[row][col+1],
                obj[row+1][col+1], obj[row+1][col], obj[row+1][col-1],
                obj[row][col-1], obj[row-1][col-1]]

                p2, p3, p4, p5, p6, p7, p8, p9 = neighbors
                black = sum(neighbors)
                #transition
                transition = 0

                for p in xrange(len(neighbors)):
                    if neighbors[p - 1] < neighbors[p]:
                        transition += 1

                if (obj[row][col] == 1 and
                    2 <= black <= 6 and
                    transition == 1 and
                    p2*p4*p6 == 0 and
                    p4*p6*p8 == 0):
                    erase.append((row,col))
        for row, col in erase: obj[row][col] = 0

        erase = []
        for row in xrange(1, obj.shape[0]-1):
            for col in xrange(1, obj.shape[1]-1):
                neighbors = [obj[row-1][col], obj[row-1][col+1], obj[row][col+1],
                obj[row+1][col+1], obj[row+1][col], obj[row+1][col-1],
                obj[row][col-1], obj[row-1][col-1]]

                p2, p3, p4, p5, p6, p7, p8, p9 = neighbors
                black = sum(neighbors)
                #transition
                transition = 0

                for p in xrange(len(neighbors)):
                    if neighbors[p - 1] < neighbors[p]:
                        transition += 1

                if (obj[row][col] == 1 and
                    2 <= black <= 6 and
                    transition == 1 and
                    p2*p4*p8 == 0 and
                    p2*p6*p8 == 0):
                    erase.append((row,col))
        for row, col in erase: obj[row][col] = 0

    plt.imshow(obj, cmap = 'Greys')

    if method == 'post':
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/thinning_result_' + random_char_thinning + '.png')
        plt.close()

    return obj

def simpang(obj):
    intersection = 0
    obj = obj.astype(bool)
    for row in range(1, obj.shape[0]-1):
        for col in xrange(1, obj.shape[1]-1):
            n = [obj[row-1,col], obj[row-1,col+1], obj[row,col+1],
            obj[row+1,col+1], obj[row+1,col], obj[row+1,col-1],
            obj[row,col-1], obj[row-1,col-1],obj[row-1,col]]

            # p2, p3, p4, p5, p6, p7, p8, p9 = n
            # print np.diff(n), np.sum(np.diff(n))
            if np.sum(np.diff(n))/2 >= 3:
                intersection += 1
                # sum_intersection = np.sum(np.diff(n))

            #print n

    return intersection

def ujung(obj):
    endpoint = 0
    # blackindex = []
    obj = obj.astype(int)
    for row in range(1, obj.shape[0]-1):
        for col in xrange(1, obj.shape[1]-1):

            if obj[row,col] == 1:
                n = [obj[row-1][col], obj[row-1][col+1], obj[row][col+1],
                     obj[row+1][col+1], obj[row+1][col], obj[row+1][col-1],
                     obj[row][col-1], obj[row-1][col-1]]

                p2, p3, p4, p5, p6, p7, p8, p9 = n
                black = sum(n)
                # pri   nt np.diff(n), sum(np.diff(n))
                if black == 1:
                    endpoint += 1
                    # blackindex.append((row,col))
                    # print n, black
                    # if sum(np.diff(n)) == 2: # np.sum(np.diff(n))
                    #     endpoint += 1
                    # sum_intersection = np.sum(np.diff(n))
                    #print n

    # print blackindex
    return endpoint

def retrieve_data_training():
    with open(os.path.dirname(os.path.realpath(__file__)) + '/static/dictionary/zhangsuen_arial.txt', 'r') as f:
        training = f.read().split('\n')

    # with open('/Users/panggi/Desktop/learn/flask/flask-boilerplate/static/dictionary/zhangsuen_arial.txt', 'r') as f:
    #     training = f.read().split('\n')
    # training_list = []
    # for i in len(training):
    #     training_list = training[i].split()

    training_split = [instance.split(',') for instance in training]
    training_int = [[int(instance[0]), int(instance[1]), instance[2]] for instance in training_split]

    # print training_list
    return training_int

def testing_data(training_int,u,s):

    some = 'tidak ketemu'
    ketemu = False
    i = 0
    while not ketemu:
        if training_int[i][0] == s and training_int[i][1] == u:
            some = training_int[i][2]
            # print training_int[i][0], training_int[i][1]
            ketemu = True
        i += 1
    return some

@app.route('/thresholding', methods=['GET', 'POST'])
def thresholding():
    if request.method == 'GET':
        return render_template('pages/placeholder.otsu.html')

    if request.method == 'POST':
        random_char = str(uuid.uuid4())
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'otsu_image_source_' + random_char))

        #convert grayscale
        img = misc.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'otsu_image_source_' + random_char))
        grayscale = img.dot( [0.299, 0.587, 0.144])
        rows, cols = np.shape(grayscale)

        #create 256 histogram
        hist = np.histogram(grayscale, 256)[0]
        total = rows * cols
        thre = otsu(hist, total)

        figure  = plt.figure( figsize=(14, 6) )
        figure.canvas.set_window_title( 'Otsu thresholding' )

        axes    = figure.add_subplot(121)
        axes.set_title('Original')
        axes.get_xaxis().set_visible( False )
        axes.get_yaxis().set_visible( False )
        axes.imshow( img, cmap='Greys_r' )

        axes    = figure.add_subplot(122)
        axes.set_title('Otsu thresholding')
        axes.get_xaxis().set_visible( False )
        axes.get_yaxis().set_visible( False )
        axes.imshow( grayscale >= thre, cmap='Greys_r' )

        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/otsu_result_' + random_char + '.png')
        plt.close()

        return render_template('pages/placeholder.otsu.post.html', random_char=random_char)

def otsu(hist, total):
    bins = len(hist)

    sum_of_total = 0 #total pixel
    for x in xrange(0, bins):
        sum_of_total += x * hist[x]

    weight_back = 0.0
    sum_back = 0.0
    variance = []

    for thres in xrange(0, bins):
        weight_back += hist[thres]
        if weight_back == 0:
            continue

        weight_fore = total - weight_back
        if weight_fore == 0:
            break

        sum_back += thres * hist[thres]
        mean_back = sum_back/ weight_back
        mean_fore = (sum_of_total - sum_back)/ weight_fore

        variance.append( weight_back * weight_fore * (mean_back - mean_fore)**2 )

    # find the threshold with maximum variances between classes
	otsu_thres = argmax(variance)
    return otsu_thres

@app.route('/faceparts', methods=['GET', 'POST'])
def faceparts():
    face_cascade = cv2.CascadeClassifier(os.path.dirname(os.path.realpath(__file__)) + '/static/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(os.path.dirname(os.path.realpath(__file__)) + '/static/haarcascades/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier(os.path.dirname(os.path.realpath(__file__)) + '/static/haarcascades/Mouth.xml')
    nose_cascade = cv2.CascadeClassifier(os.path.dirname(os.path.realpath(__file__)) + '/static/haarcascades/Nose.xml')

    if request.method == 'GET':
        return render_template('pages/placeholder.faceparts.html')

    if request.method == 'POST':
        random_char = str(uuid.uuid4())
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'faceparts_image_source_' + random_char))

        #convert grayscale
        img = misc.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'faceparts_image_source_' + random_char))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            i = 0
            while i < len(eyes)-1:
                ex1,ey1,ew1,eh1 = eyes[i]
                ex2,ey2,ew2,eh2 = eyes[i+1]
                if abs(ex1-ex2) > 20 and abs(ey1-ey2)<10:
                    cv2.rectangle(roi_color,(ex1,ey1),(ex1+ew1,ey1+eh1),(0,255,0),2)
                    cv2.rectangle(roi_color,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),2)

                i = i+1
            nose = nose_cascade.detectMultiScale(roi_gray)
            for (nx,ny,nw,nh) in nose:
               cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)
            mouth = mouth_cascade.detectMultiScale(roi_gray)
            for (mx,my,mw,mh) in mouth:
                if (my > my+(mh/2)):
                    cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,255,0),2)

        plt.imshow(img)
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/faceparts_result_' + random_char + '.png')
        plt.close()

        return render_template('pages/placeholder.faceparts.post.html', random_char=random_char)

@app.route('/sobel', methods=['GET', 'POST'])
def sobel():
    if request.method == 'GET':
        return render_template('pages/placeholder.sobel.html')

    if request.method == 'POST':
        random_char = str(uuid.uuid4())
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'sobel_image_source_' + random_char))

        #convert grayscale
        img = misc.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'sobel_image_source_' + random_char))

        new_img = grayscale(img)
        filter1 = matrix_convolution_row()
        filter2 = matrix_convolution_col()

        img_convolve = edge_detection_sobel(new_img,filter1,filter2)
        img_homogen = edge_detection_homogen(new_img)
        img_diff = edge_detection_diff(new_img)

        figure  = plt.figure( figsize=(15, 7) )
        figure.canvas.set_window_title( 'Image Convolve' )

        axes    = figure.add_subplot(121)
        axes.set_title('Original')
        axes.get_xaxis().set_visible( False )
        axes.get_yaxis().set_visible( False )
        axes.imshow( img, cmap='Greys_r' )

        axes    = figure.add_subplot(122)
        axes.set_title('Sobel')
        axes.get_xaxis().set_visible( False )
        axes.get_yaxis().set_visible( False )
        axes.imshow( img_convolve, cmap='Greys_r' )

        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/sobel_result_' + random_char + '.png')
        plt.close()

        return render_template('pages/placeholder.sobel.post.html', random_char=random_char)

def grayscale(obj):

    # get image grayscale
    return np.dot(obj[...,:3], [0.299, 0.587, 0.144])

def histogram_gc(img_gc):

    histogram_img_gc = []
    for row in xrange(0,img_gc.shape[0]-1):
        for col in xrange(0, img_gc.shape[1]-1):

            color = img_gc[row][col]
            histogram_img_gc[color] += 1

    return histogram_img_gc

def matrix_convolution_row():

    # matriks untuk mendapatkan color value yang baru menggunakan sobel
    mc_row = np.asarray([-1,-2,-1,0,0,0,1,2,1])
    return mc_row

def matrix_convolution_col():

    # matriks untuk mendapatkan color value yang baru menggunakan sobel
    mc_col = np.asarray([-1,0,1,-2,0,2,-1,0,1])
    return mc_col

def edge_detection_sobel(imggrey,filter1,filter2):

    global sc1, sc2
    arr_sobel = np.zeros((imggrey.shape[0],imggrey.shape[1]))
    for row in range(1,imggrey.shape[0]-1):
        for col in range(1, imggrey.shape[1]-1):
            slice = imggrey[row-1:row+2, col-1:col+2]
            rslice = slice.reshape(9)

            c1 = [rslice[0]*filter1[0],rslice[1]*filter1[1],
                  rslice[2]*filter1[2],rslice[3]*filter1[3],
                  rslice[4]*filter1[4],rslice[5]*filter1[5],
                  rslice[6]*filter1[6],rslice[7]*filter1[7],
                  rslice[8]*filter1[8]]

            c2 = [rslice[0]*filter2[0],rslice[1]*filter2[1],
                  rslice[2]*filter2[2],rslice[3]*filter2[3],
                  rslice[4]*filter2[4],rslice[5]*filter2[5],
                  rslice[6]*filter2[6],rslice[7]*filter2[7],
                  rslice[8]*filter2[8]]

            sc1 = sum(c1)
            sc2 = sum(c2)

            arr_sobel[row][col] = abs(sc1) + abs(sc2)

    # print arr_sobel

    r_arr_sobel = arr_sobel.reshape(arr_sobel.shape[0]*arr_sobel.shape[1])

    min_number = min(r_arr_sobel)
    max_number = max(r_arr_sobel)
    # print max_number
    # print min_number

    normal = ((r_arr_sobel - min_number)*255)/(max_number - min_number)
    normal = normal.reshape(imggrey.shape[0],imggrey.shape[1])

    return normal

def edge_detection_homogen(imggrey):

    homogen = np.zeros((imggrey.shape[0],imggrey.shape[1]))
    for row in range(1, imggrey.shape[0]-1):
        for col in range(1, imggrey.shape[1]-1):

            value = [abs(imggrey[row][col]-imggrey[row-1][col-1]),abs(imggrey[row][col]-imggrey[row-1][col]),
                     abs(imggrey[row][col]-imggrey[row-1][col+1]),abs(imggrey[row][col]-imggrey[row][col-1]),
                     abs(imggrey[row][col]-imggrey[row][col+1]),abs(imggrey[row][col]-imggrey[row+1][col-1]),
                     abs(imggrey[row][col]-imggrey[row+1][col]),abs(imggrey[row][col]-imggrey[row+1][col+1])]

            max_number = max(value)
            homogen[row][col] = max_number

    return homogen

def edge_detection_diff(imggrey):

    diff = np.zeros((imggrey.shape[0],imggrey.shape[1]))
    for row in range(1, imggrey.shape[0]-1):
        for col in range(1, imggrey.shape[1]-1):
            value = [abs(imggrey[row-1][col-1]-imggrey[row+1][col+1]),abs(imggrey[row-1][col]-imggrey[row+1][col]),
                     abs(imggrey[row-1][col+1]-imggrey[row+1][col-1]),abs(imggrey[row][col-1]-imggrey[row][col+1])]

            max_number = max(value)
            diff[row][col] = max_number

    return diff

@app.route('/prewitt', methods=['GET', 'POST'])
def prewitt():

    if request.method == 'GET':
        return render_template('pages/placeholder.prewitt.html')

    if request.method == 'POST':
        random_char = str(uuid.uuid4())
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'prewitt_image_source_' + random_char))

        #convert grayscale
        img = misc.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'prewitt_image_source_' + random_char))

        new_img = grayscale(img)
        filter_prewitt_1 = matrix_convolution_prewitt_1()
        filter_prewitt_2 = matrix_convolution_prewitt_2()
        filter_prewitt_3 = matrix_convolution_prewitt_3()
        filter_prewitt_4 = matrix_convolution_prewitt_4()
        filter_prewitt_5 = matrix_convolution_prewitt_5()
        filter_prewitt_6 = matrix_convolution_prewitt_6()
        filter_prewitt_7 = matrix_convolution_prewitt_7()
        filter_prewitt_8 = matrix_convolution_prewitt_8()

        img_edge_d2_prewitt = edge_detection_d2(new_img,filter_prewitt_1,filter_prewitt_2,filter_prewitt_3,filter_prewitt_4,filter_prewitt_5,filter_prewitt_6,filter_prewitt_7,filter_prewitt_8)

        figure  = plt.figure( figsize=(15, 7) )
        figure.canvas.set_window_title( 'Image Convolve' )

        axes    = figure.add_subplot(121)
        axes.set_title('Original')
        axes.get_xaxis().set_visible( False )
        axes.get_yaxis().set_visible( False )
        axes.imshow( img, cmap='Greys_r' )

        axes    = figure.add_subplot(122)
        axes.set_title('Prewitt')
        axes.get_xaxis().set_visible( False )
        axes.get_yaxis().set_visible( False )
        axes.imshow( img_edge_d2_prewitt, cmap='Greys_r' )

        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/prewitt_result_' + random_char + '.png')
        plt.close()

        return render_template('pages/placeholder.prewitt.post.html', random_char=random_char)

def grayscale(obj):

    # get image grayscale
    return np.dot(obj[...,:3], [0.299, 0.587, 0.144])

def histogram_gc(img_gc):

    histogram_img_gc = []
    for row in xrange(0,img_gc.shape[0]-1):
        for col in xrange(0, img_gc.shape[1]-1):

            color = img_gc[row][col]
            histogram_img_gc[color] += 1

    return histogram_img_gc

def matrix_convolution_prewitt_1():
    # matriks prewitt
    mc_1 = np.asarray([1,1,1,0,0,0,-1,-1,-1])
    return mc_1

def matrix_convolution_prewitt_2():
    # matriks prewitt
    mc_2 = np.asarray([0,1,1,-1,0,1,-1,-1,0])
    return mc_2

def matrix_convolution_prewitt_3():
    # matriks prewitt
    mc_3 = np.asarray([-1,0,1,-1,0,1,-1,0,1])
    return mc_3

def matrix_convolution_prewitt_4():
    # matriks prewitt
    mc_4 = np.asarray([-1,-1,0,-1,0,1,0,1,1])
    return mc_4

def matrix_convolution_prewitt_5():
    # matriks prewitt
    mc_5 = np.asarray([-1,-1,-1,0,0,0,1,1,1])
    return mc_5

def matrix_convolution_prewitt_6():
    # matriks prewitt
    mc_6 = np.asarray([0,-1,-1,1,0,-1,1,1,0])
    return mc_6

def matrix_convolution_prewitt_7():
    # matriks prewitt
    mc_7 = np.asarray([1,0,-1,1,0,-1,1,0,-1])
    return mc_7

def matrix_convolution_prewitt_8():
    # matriks prewitt
    mc_8 = np.asarray([1,1,0,1,0,-1,0,-1,-1])
    return mc_8

def edge_detection_d2(imggrey,filter1,filter2,filter3,filter4,filter5,filter6,filter7,filter8):

    arr_kirsch = np.zeros((imggrey.shape[0],imggrey.shape[1]))
    for row in range(1,imggrey.shape[0]-1):
        for col in range(1, imggrey.shape[1]-1):
            slice = imggrey[row-1:row+2, col-1:col+2]
            rslice = slice.reshape(9)

            c1 = [rslice[0]*filter1[0],rslice[1]*filter1[1],
                  rslice[2]*filter1[2],rslice[3]*filter1[3],
                  rslice[4]*filter1[4],rslice[5]*filter1[5],
                  rslice[6]*filter1[6],rslice[7]*filter1[7],
                  rslice[8]*filter1[8]]

            c2 = [rslice[0]*filter2[0],rslice[1]*filter2[1],
                  rslice[2]*filter2[2],rslice[3]*filter2[3],
                  rslice[4]*filter2[4],rslice[5]*filter2[5],
                  rslice[6]*filter2[6],rslice[7]*filter2[7],
                  rslice[8]*filter2[8]]

            c3 = [rslice[0]*filter3[0],rslice[1]*filter3[1],
                  rslice[2]*filter3[2],rslice[3]*filter3[3],
                  rslice[4]*filter3[4],rslice[5]*filter3[5],
                  rslice[6]*filter3[6],rslice[7]*filter3[7],
                  rslice[8]*filter3[8]]

            c4 = [rslice[0]*filter4[0],rslice[1]*filter4[1],
                  rslice[2]*filter4[2],rslice[3]*filter4[3],
                  rslice[4]*filter4[4],rslice[5]*filter4[5],
                  rslice[6]*filter4[6],rslice[7]*filter4[7],
                  rslice[8]*filter4[8]]

            c5 = [rslice[0]*filter5[0],rslice[1]*filter5[1],
                  rslice[2]*filter5[2],rslice[3]*filter5[3],
                  rslice[4]*filter5[4],rslice[5]*filter5[5],
                  rslice[6]*filter5[6],rslice[7]*filter5[7],
                  rslice[8]*filter5[8]]

            c6 = [rslice[0]*filter6[0],rslice[1]*filter6[1],
                  rslice[2]*filter6[2],rslice[3]*filter6[3],
                  rslice[4]*filter6[4],rslice[5]*filter6[5],
                  rslice[6]*filter6[6],rslice[7]*filter6[7],
                  rslice[8]*filter6[8]]

            c7 = [rslice[0]*filter7[0],rslice[1]*filter7[1],
                  rslice[2]*filter7[2],rslice[3]*filter7[3],
                  rslice[4]*filter7[4],rslice[5]*filter7[5],
                  rslice[6]*filter7[6],rslice[7]*filter7[7],
                  rslice[8]*filter7[8]]

            c8 = [rslice[0]*filter8[0],rslice[1]*filter8[1],
                  rslice[2]*filter8[2],rslice[3]*filter8[3],
                  rslice[4]*filter8[4],rslice[5]*filter8[5],
                  rslice[6]*filter8[6],rslice[7]*filter8[7],
                  rslice[8]*filter8[8]]

            sc1 = sum(c1)
            sc2 = sum(c2)
            sc3 = sum(c3)
            sc4 = sum(c4)
            sc5 = sum(c5)
            sc6 = sum(c6)
            sc7 = sum(c7)
            sc8 = sum(c8)

            arr_kirsch[row][col] = abs(sc1) + abs(sc2) + abs(sc3) + abs(sc4) + abs(sc5) + abs(sc6) + abs(sc7) + abs(sc8)

    r_arr_kirsch = arr_kirsch.reshape(arr_kirsch.shape[0]*arr_kirsch.shape[1])

    min_number = min(r_arr_kirsch)
    max_number = max(r_arr_kirsch)
    # print max_number
    # print min_number

    normal = ((r_arr_kirsch - min_number)*255)/(max_number - min_number)
    normal = normal.reshape(imggrey.shape[0],imggrey.shape[1])

    return normal

@app.route('/kirsch', methods=['GET', 'POST'])
def kirsch():

    if request.method == 'GET':
        return render_template('pages/placeholder.kirsch.html')

    if request.method == 'POST':
        random_char = str(uuid.uuid4())
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'kirsch_image_source_' + random_char))

        #convert grayscale
        img = misc.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'kirsch_image_source_' + random_char))

        new_img = grayscale(img)
        filter_kirsch_1 = matrix_convolution_kirsch_1()
        filter_kirsch_2 = matrix_convolution_kirsch_2()
        filter_kirsch_3 = matrix_convolution_kirsch_3()
        filter_kirsch_4 = matrix_convolution_kirsch_4()
        filter_kirsch_5 = matrix_convolution_kirsch_5()
        filter_kirsch_6 = matrix_convolution_kirsch_6()
        filter_kirsch_7 = matrix_convolution_kirsch_7()
        filter_kirsch_8 = matrix_convolution_kirsch_8()

        img_edge_d2_kirsch = edge_detection_d2(new_img,filter_kirsch_1,filter_kirsch_2,filter_kirsch_3,filter_kirsch_4,filter_kirsch_5,filter_kirsch_6,filter_kirsch_7,filter_kirsch_8)

        figure  = plt.figure( figsize=(15, 7) )
        figure.canvas.set_window_title( 'Image Convolve' )

        axes    = figure.add_subplot(121)
        axes.set_title('Original')
        axes.get_xaxis().set_visible( False )
        axes.get_yaxis().set_visible( False )
        axes.imshow( img, cmap='Greys_r' )

        axes    = figure.add_subplot(122)
        axes.set_title('Kirsch')
        axes.get_xaxis().set_visible( False )
        axes.get_yaxis().set_visible( False )
        axes.imshow( img_edge_d2_kirsch, cmap='Greys_r' )

        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/kirsch_result_' + random_char + '.png')
        plt.close()

        return render_template('pages/placeholder.kirsch.post.html', random_char=random_char)

def grayscale(obj):

    # get image grayscale
    return np.dot(obj[...,:3], [0.299, 0.587, 0.144])

def histogram_gc(img_gc):

    histogram_img_gc = []
    for row in xrange(0,img_gc.shape[0]-1):
        for col in xrange(0, img_gc.shape[1]-1):

            color = img_gc[row][col]
            histogram_img_gc[color] += 1

    return histogram_img_gc


def matrix_convolution_kirsch_1():
    # matriks kirsch
    mc_1 = np.asarray([5,5,5,-3,0,-3,-3,-3,-3])
    return mc_1

def matrix_convolution_kirsch_2():
    # matriks kirsch
    mc_2 = np.asarray([5,5,-3,5,0,-3,-3,-3,-3])
    return mc_2

def matrix_convolution_kirsch_3():
    # matriks kirsch
    mc_3 = np.asarray([5,-3,-3,5,0,-3,5,-3,-3])
    return mc_3

def matrix_convolution_kirsch_4():
    # matriks kirsch
    mc_4 = np.asarray([-3,-3,-3,5,0,-3,5,5,-3])
    return mc_4

def matrix_convolution_kirsch_5():
    # matriks kirsch
    mc_5 = np.asarray([-3,-3,-3,-3,0,-3,5,5,5])
    return mc_5

def matrix_convolution_kirsch_6():
    # matriks kirsch
    mc_6 = np.asarray([-3,-3,-3,-3,0,5,-3,5,5])
    return mc_6

def matrix_convolution_kirsch_7():
    # matriks kirsch
    mc_7 = np.asarray([-3,-3,5,-3,0,5,-3,-3,5])
    return mc_7

def matrix_convolution_kirsch_8():
    # matriks kirsch
    mc_8 = np.asarray([-3,5,5,-3,0,5,-3,-3,-3])
    return mc_8

def edge_detection_d2(imggrey,filter1,filter2,filter3,filter4,filter5,filter6,filter7,filter8):

    arr_kirsch = np.zeros((imggrey.shape[0],imggrey.shape[1]))
    for row in range(1,imggrey.shape[0]-1):
        for col in range(1, imggrey.shape[1]-1):
            slice = imggrey[row-1:row+2, col-1:col+2]
            rslice = slice.reshape(9)

            c1 = [rslice[0]*filter1[0],rslice[1]*filter1[1],
                  rslice[2]*filter1[2],rslice[3]*filter1[3],
                  rslice[4]*filter1[4],rslice[5]*filter1[5],
                  rslice[6]*filter1[6],rslice[7]*filter1[7],
                  rslice[8]*filter1[8]]

            c2 = [rslice[0]*filter2[0],rslice[1]*filter2[1],
                  rslice[2]*filter2[2],rslice[3]*filter2[3],
                  rslice[4]*filter2[4],rslice[5]*filter2[5],
                  rslice[6]*filter2[6],rslice[7]*filter2[7],
                  rslice[8]*filter2[8]]

            c3 = [rslice[0]*filter3[0],rslice[1]*filter3[1],
                  rslice[2]*filter3[2],rslice[3]*filter3[3],
                  rslice[4]*filter3[4],rslice[5]*filter3[5],
                  rslice[6]*filter3[6],rslice[7]*filter3[7],
                  rslice[8]*filter3[8]]

            c4 = [rslice[0]*filter4[0],rslice[1]*filter4[1],
                  rslice[2]*filter4[2],rslice[3]*filter4[3],
                  rslice[4]*filter4[4],rslice[5]*filter4[5],
                  rslice[6]*filter4[6],rslice[7]*filter4[7],
                  rslice[8]*filter4[8]]

            c5 = [rslice[0]*filter5[0],rslice[1]*filter5[1],
                  rslice[2]*filter5[2],rslice[3]*filter5[3],
                  rslice[4]*filter5[4],rslice[5]*filter5[5],
                  rslice[6]*filter5[6],rslice[7]*filter5[7],
                  rslice[8]*filter5[8]]

            c6 = [rslice[0]*filter6[0],rslice[1]*filter6[1],
                  rslice[2]*filter6[2],rslice[3]*filter6[3],
                  rslice[4]*filter6[4],rslice[5]*filter6[5],
                  rslice[6]*filter6[6],rslice[7]*filter6[7],
                  rslice[8]*filter6[8]]

            c7 = [rslice[0]*filter7[0],rslice[1]*filter7[1],
                  rslice[2]*filter7[2],rslice[3]*filter7[3],
                  rslice[4]*filter7[4],rslice[5]*filter7[5],
                  rslice[6]*filter7[6],rslice[7]*filter7[7],
                  rslice[8]*filter7[8]]

            c8 = [rslice[0]*filter8[0],rslice[1]*filter8[1],
                  rslice[2]*filter8[2],rslice[3]*filter8[3],
                  rslice[4]*filter8[4],rslice[5]*filter8[5],
                  rslice[6]*filter8[6],rslice[7]*filter8[7],
                  rslice[8]*filter8[8]]

            sc1 = sum(c1)
            sc2 = sum(c2)
            sc3 = sum(c3)
            sc4 = sum(c4)
            sc5 = sum(c5)
            sc6 = sum(c6)
            sc7 = sum(c7)
            sc8 = sum(c8)

            arr_kirsch[row][col] = abs(sc1) + abs(sc2) + abs(sc3) + abs(sc4) + abs(sc5) + abs(sc6) + abs(sc7) + abs(sc8)

    r_arr_kirsch = arr_kirsch.reshape(arr_kirsch.shape[0]*arr_kirsch.shape[1])

    min_number = min(r_arr_kirsch)
    max_number = max(r_arr_kirsch)
    # print max_number
    # print min_number

    normal = ((r_arr_kirsch - min_number)*255)/(max_number - min_number)
    normal = normal.reshape(imggrey.shape[0],imggrey.shape[1])

    return normal

# Error handlers.

@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')


#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
# if __name__ == '__main__':
#     app.run()

# Or specify port manually:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5181))
    app.run(host='0.0.0.0', port=port)
