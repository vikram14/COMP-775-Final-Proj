import cv2
import numpy as np

BG=1
FG=2

def getUserSeeds(image_path,image_size,scale_factor=2,pointer_radius=2,hsv=False):
    fg_blue=[225,0,0]
    bg_red=[0,0,255]

    def getSeeds(image,scale_factor=1,radius=3):
        if(len(image.shape)==2):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        seeds = np.zeros((image.shape[0],image.shape[1]))
        alphas=np.zeros((image.shape[0],image.shape[1]))
        image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        def selectRoi():
            print('select ROI.')
            r=cv2.selectROI('Select ROI. Unselected Region will be considered as a part of the background.', image)
            cv2.destroyAllWindows()
            if r[2]==0 or r[3]==0:
                return None
            r=(r[0],r[1],r[0]+r[2],r[1]+r[3])
            cv2.rectangle(image,(r[0],r[1]),(r[2],r[3]),bg_red,2)
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if not (y>r[1] and y<r[3] and x>r[0] and x<r[2]):
                        seeds[y//scale_factor,x//scale_factor]=BG
            roi=(r[0]//scale_factor,r[1]//scale_factor,r[2]//scale_factor,r[3]//scale_factor)
            return roi

        def drawPoint(x, y, pixelType):
            if pixelType == FG:
                color, code = fg_blue, FG
            else:
                color, code = bg_red, BG
            cv2.circle(image, (x, y), radius, color, -1)
            cv2.circle(seeds, (x // scale_factor, y // scale_factor), radius // scale_factor, code, -1)

        def onMouse(event, x, y, flags, pixelType):
            global drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                drawPoint(x, y, pixelType)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                drawPoint(x, y, pixelType)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False

        def seed(pixelType):
            global drawing
            drawing=False
            print(f"Seeds for {'FG' if pixelType==FG else 'BG'}")
            windowname=f"Seeds for {'FG' if pixelType==FG else 'BG'}"
            cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(windowname, onMouse, pixelType)
            while(True):
                cv2.imshow(windowname,image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

        r = selectRoi()
        cv2.imshow('Press b to select just BG (press b), any other key for both FG and BG',alphas)
        g=cv2.waitKey(0)
        cv2.destroyAllWindows()
        # if(g & 0xff ==ord('b')): #GRABCUT
        #     # alpha:unselected cells will be considered foreground for GrabCut
        #     seed(BG)
        #     alphas[np.where(seeds!=BG)]=1
        # else:
            # alpha:unselected cells will be considered background for GrabCut
        seed(FG)
        seed(BG)
        alphas[np.where(seeds==FG)]=1
        return image, seeds,alphas,r
    
    image = cv2.imread(image_path)#'./images/CorpusCallosumImages/corpus_callosum_1.png')
    
    image_orig =cv2.copyTo(image,None,None)
    image= cv2.resize(image,image_size)
    image_resized= cv2.copyTo(image,None,None)
    if hsv:
        image_resized= cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
    image, seed,alphas,r= getSeeds(image,scale_factor,pointer_radius)
    cv2.imwrite(image_path[:-4]+f'_user_seeded.jpg',image)
    return seed,alphas,r,image_orig,image_resized
