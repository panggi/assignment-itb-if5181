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
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import uuid

#----------------------------------------------------------------------------#
# Constants
#----------------------------------------------------------------------------#
UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg'])

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
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/histogram/histogram_' + random_char + '.png')
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
    random_char_chaincode = str(uuid.uuid4())

    if request.method == 'GET':
        font_image = '/static/chaincode/A_arial.jpg'
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'chaincode_' + random_char_chaincode))
        font_image = '/static/uploads/chaincode_' + random_char_chaincode

    chaincode_source_img = misc.imread(os.path.dirname(os.path.realpath(__file__)) + font_image)
    chaincode_bwim = (0.2989 * chaincode_source_img[:,:,0] + 0.587 * chaincode_source_img[:,:,1] + 0.114 * chaincode_source_img[:,:,2]).astype(np.uint8) #grayscale
    binary =  chaincode_bwim < 128 #blackwhite
    chaincode_bw = np.argwhere(binary)[0] #return index dari array hasil dari operasi boolean/ index dimana chaincode_bwim < 128 rubah jadi list biasa
    point = (chaincode_bw - (0,1)).tolist()
    firstpix = chaincode_bw.tolist()
    chaincode = get_chaincode()
    kodebelok = kode_belok(chaincode)

    if request.method == 'GET':
        return render_template('pages/placeholder.chaincode.html', chaincode=chaincode, kodebelok=kodebelok)

    if request.method == 'POST':
        return render_template('pages/placeholder.chaincode.post.html', chaincode=chaincode, kodebelok=kodebelok, random_char_chaincode=random_char_chaincode)

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

# not used yet
def wr(content):
    with open(os.path.dirname(os.path.realpath(__file__)) + '/static/chaincode/kamus','w') as f:
        f.write(str(content))

def rd(content):
    with open(os.path.dirname(os.path.realpath(__file__)) + '/static/chaincode/kamus','r') as f:
        raw = f.read().split('\n')


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

    if request.method == 'GET':
        thinning = zhang_suen(thinning_bw, 'get')
        return render_template('pages/placeholder.thinning.html', thinning=thinning)

    if request.method == 'POST':
        thinning = zhang_suen(thinning_bw, 'post')
        return render_template('pages/placeholder.thinning.post.html', thinning=thinning, random_char_thinning=random_char_thinning)

def get_bw():
    for row in xrange(thinning_source_img.shape[0]):
        for col in xrange(thinning_source_img.shape[1]):
            if(np.sum(thinning_source_img[row][col]))/3 > 128:
                thinning_bw[row][col] = 0
            else:
                thinning_bw[row][col] = 1

def zhang_suen(obj, method):
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

    if method == 'get':
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/thinning/thinning.png')
        plt.close()

    if method == 'post':
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '/static/uploads/thinning_result_' + random_char_thinning + '.png')
        plt.close()


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
