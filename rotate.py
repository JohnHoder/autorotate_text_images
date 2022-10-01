import cv2
import pytesseract
import urllib
import numpy as np
import re
import urllib.request
import os, argparse
from PIL import Image

# sudo dnf -y install tesseract-langpack-osd
# sudo dnf -y install tesseract-langpack-deu       
# sudo dnf -y install tesseract-langpack-fra

# Installs: https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/

output_directory = ''

picture_re = re.compile(r'.*\.jpg$', re.IGNORECASE)

def autorotate(pic_path, rotatedSuffix):
    """ This function autorotates a picture """
     
    # Uncomment the line below to provide path to tesseract manually
    # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    #imagex = cv2.imread(path)
 
    imagex = Image.open(pic_path) #opens as RGB and not BRG which OpenCV uses
    #pic_array = np.array(imagex)

    # Read image from URL
    # Taken from https://stackoverflow.com/questions/21061814/how-can-i-read-an-image-from-an-internet-url-in-python-cv2-scikit-image-and-mah
    # https://i.ibb.co/4mm9WvZ/book-rot.jpg
    # https://i.ibb.co/M7jwWR2/book.jpg
    # https://i.ibb.co/27bKNJ8/book-rot2.jpg
    #
    #resp = urllib.request.urlopen('https://i.ibb.co/27bKNJ8/book-rot2.jpg')
    #image = np.asarray(bytearray(resp.read()), dtype="uint8")

    image = np.asarray(imagex)

    #image = cv2.imdecode(pic_array, cv2.IMREAD_COLOR) # Initially decode as color
  
    #  TAKEN FROM: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
 

    print("OSD STATS:")
    rot_data = pytesseract.image_to_osd(gray)
    print("[OSD] " + rot_data)
    rot = re.search('(?<=Rotate: )\d+', rot_data).group(0)

    angle = float(rot)
    if angle > 0:
        angle = 360 - angle
    print("[ANGLE] " + str(angle))


    # ANOTHER METHOD, BUT SIZE OF FILE SEEMS TO BE EVEN WORSE
    # #################
    # #################
    # rotated_yolo = imagex.rotate(angle, expand=True)
    # write_path = output_directory + os.path.basename(pic_path)[:-4] + '_rotated' '.jpg'
    # rotated_yolo.save(write_path, subsampling=0, quality=100)

    # #################
    # #################
    
    # rotate the image to deskew it
    ##################################
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (nW, nH))
    #rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) 
    
    #Nice but takes time
    #print("OSD STATS AFTER:")
    #print(pytesseract.image_to_osd(rotated));

    #Rotated image can be saved here
    write_path = output_directory + os.path.basename(pic_path)[:-4] + rotatedSuffix + '.jpg'
    print(write_path)
    # Do not forget to convert from BRG to RGB
    cv2.imwrite(write_path, cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))

    #Detect text by ORC
    #print("[TEXT]")
    # Run tesseract OCR on image
    #text = pytesseract.image_to_string(rotated, lang='fra', config="--psm 1")
    # Print recognized text
    #print(text.encode(encoding='UTF-8'))



def process_directory(path, recursive=False, rotatedSuffix=''):
    """ This function processes all elements from a directory """

    if not os.path.isdir(path):
        print(path, 'is not a directory')

    else:
        for elt in os.listdir(path):
            elt_path = os.path.join(path, elt)
            if os.path.isdir(elt_path) and recursive:
                process_directory(elt_path, recursive)

            elif os.path.isfile(elt_path):
                if re.match(picture_re, elt_path):
                    print("=== Processing %s ===" % (elt))
                    #for i in range(2): # for some reason, I have to do it twice
                    #    if autorotate(elt_path):
                    #        print 'autorotate: %s/%s' % (path, elt)
                    autorotate(elt_path, rotatedSuffix)


# def getOutputFolder(fullpath, path='./', output_folder='rotated/'):
#     #Create the output folder if it does not exist
#     if fullpath == (None or 0):
#         folderpath = path + output_folder
#     else:
#         folderpath = fullpath

#     folderExists = os.path.exists(folderpath)
#     if not folderExists:
#         os.makedirs(folderpath)
#         print(folderpath + " directory has been created.")
#     return folderpath


#####################################
### TODO: recursive does not work ###
#####################################

parser = argparse.ArgumentParser()

if __name__ == '__main__':
    parser.add_argument('--dir', '-d', nargs='+')
    parser.add_argument('--recursive', '-r', action='store_true')
    parser.add_argument('--file', '-f', nargs='+')
    parser.add_argument('--output', '-o', nargs='+')
    parser.add_argument('--rewrite', '-rw', action='store_true')
    parser.add_argument('--suffix', '-s', action='store_true')
    args = parser.parse_args()

    #Create the output folder if it does not exist
    if args.output != None:
        output_directory = args.output[0] + '/'
    else:
        if args.dir != None:
            output_directory = args.dir[0] + '/'
        elif args.file != None:
            # TODO: iterate thru all the files
            output_directory = os.path.dirname(os.path.abspath(args.file[0])) + '/'
            #print(output_directory)
        else:
            raise Exception('ERROR')

    if args.rewrite == False:
        print("Rewrite: OFF")
        output_directory = output_directory + 'rotated/'
    else:
        print("Rewrite: ON")

    folderExists = os.path.exists(output_directory)
    if not folderExists:
        os.makedirs(output_directory)
        print(output_directory + " directory has been created.")

    if args.dir != None:
        if len(args.dir) >= 1:
            for d in args.dir:
                if args.suffix == False:
                    process_directory(d, args.recursive)
                else:
                    process_directory(d, args.recursive, '_rotated')

    elif args.file != None:
        for f in args.file:
            autorotate(f)    
    else:
        print('-d or -f argument missing')

    
