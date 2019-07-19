import cv2
import os


def frames2video(framedir, videodir):
    '''
    framedir: images in this framedir will need to be writen into a video
    videodir: saved videodir
    '''
    fps = 24
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    extend_list=['jpg','png','bmp']
    img_list = os.listdir(framedir)
    img_list = [img for img in img_list if img[-3:] in extend_list]
    assert len(img_list)!=0, 'There is no image in specified directory. Is the path right??'
    img_0 = cv2.imread(os.path.join(framedir, img_list[0]))
    H, W, C = img_0.shape
    img_list.sort()
    videoWriter = cv2.VideoWriter('{}.avi'.format(videodir), fourcc, fps, (W,H))
    for img in img_list:
        img = cv2.imread(os.path.join(framedir, img))
        if img.shape != img_0.shape:
            videoWriter.release()
            print('Shape of images in this directory is not the same!!')
            return
        videoWriter.write(img)
    videoWriter.release()
    print('Image in: {} have been writen into video: {}'.format(framedir, videodir))


def video2frames(videodir, framedir):
    '''
    videodir: video need to be captured
    framedir: where to save frames2video
    '''
    cap = cv2.VideoCapture(videodir)
    if not cap.isOpened():
        print('Video: {} if open failed!!'.format(videodir))
    frame_count = 1
    success = True
    while success:
        success, frame = cap.read()
        params = []
        #params.append(cv.CV_IMWRITE_PXM_BINARY)
        params.append(1)
        cv2.imwrite(os.path.join(framedir,'{}.jpg'.format(str(frame_count).zfill(6))), frame, params)
        frame_count += 1
    cap.release()
    print('Frames in {} have been extracted in {}'.format(videodir, framedir))


if __name__=='__main__':
    videodir = ['data/UCY/univ/students003.avi',
                'data/UCY/zara/zara01/crowds_zara01.avi',
                'data/UCY/zara/zara02/crowds_zara02.avi',
                'data/ETH/ewap_dataset/seq_eth/seq_eth.avi',
                'data/ETH/ewap_dataset/seq_hotel/seq_hotel.avi']
    for datadir in videodir:
        framedir = '/'.join(datadir.split('/')[:-1]+['frames'])
        print(framedir)
        if not os.path.exists(framedir):
            os.makedirs(framedir)
        video2frames(datadir, framedir)