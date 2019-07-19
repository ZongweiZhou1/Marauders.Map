import numpy as np
import argparse
import os.path as osp
from matplotlib import animation
import matplotlib.pyplot as plt
import cv2
import seaborn


class trajectory():
    def __init__(self, trackid, anno_data):
        # anno_data: N x 4, (frame_id, trackid, pos_x, pos_y)
        self.trackid = trackid
        self.handler = None
        self.frameID = anno_data[:, 0]
        self.pointsx = anno_data[:, 2]
        self.pointsy = anno_data[:, 3]
        self.color = seaborn.xkcd_rgb[seaborn.xkcd_rgb.keys()[int(trackid)%200*3]]
        self.exhibit_pointsx = []
        self.exhibit_pointsy = []
        # prediction
        self.pred_handler = None
        self.pred_pointsx = []
        self.pred_pointsy = []


def parse():
    parser = argparse.ArgumentParser('args of loaded dataset')
    parser.add_argument('--dataset', default='seq_eth', type=str,
                        help='dataset name', choices=['seq_eth', 'seq_hotel', 'zara01', 'zara02', 'univ'])
    parser.add_argument('--ghost_len', default=10, type=int,
                        help='ghost length of each trajectory')
    parser.add_argument('--pred', default=False, type=bool,
                        help='whether need to display predictions')

    args = parser.parse_args()
    return args


def plot_sequence_images(image_dir, init_image, anno_data, args, interval_frame=8):
    """
    :param image_dir:           image directory (*.jpg)
    :param anno_image:          image for initialization
    :param anno_data:           N x 4, frameId, trackId, pos_x, pos_y
    """
    ghost_len = args.ghost_len
    pred = args.pred
    dataset = args.dataset

    anno_data = anno_data[np.argsort(anno_data[:, 0])]
    fig, ax = plt.subplots()
    image_handler = ax.imshow(init_image)
    trajectory_dict = {}
    # extract trajectories
    trackids = set(anno_data[:, 1])
    for trackid in trackids:
        curr_anno = anno_data[anno_data[:, 1] == trackid]
        curr_anno = curr_anno[np.argsort(curr_anno[:, 0])]  # ascent by frameid
        trajectory_dict[trackid] = trajectory(trackid, anno_data=curr_anno)
        trajectory_dict[trackid].handler, = ax.plot([],[], color=trajectory_dict[trackid].color,
                                                   marker='o', markersize=2)
        if pred:
            trajectory_dict[trackid].pred_handler, = ax.plot([],[], color=trajectory_dict[trackid].color,
                                                   marker='*', markersize=2)

    frameids = set(anno_data[:, 0])

    def get_frame():
        for i, frameid in enumerate(frameids):
            img = cv2.imread(osp.join(image_dir,
                        '{}.jpg'.format(str(int(frameid*interval_frame)).zfill(6))))
            if img is None:
                break
            im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            exists_trackids = []
            for trackid in trackids:
                ghost_frame = np.logical_and(trajectory_dict[trackid].frameID >frameid-ghost_len,
                                             trajectory_dict[trackid].frameID <= frameid)
                if len(trajectory_dict[trackid].exhibit_pointsx) != 0:
                    exists_trackids.append(trackid)
                if ghost_frame.sum() > 0:
                    trajectory_dict[trackid].exhibit_pointsx = trajectory_dict[trackid].pointsx[ghost_frame]
                    trajectory_dict[trackid].exhibit_pointsy = trajectory_dict[trackid].pointsy[ghost_frame]
                else:
                    trajectory_dict[trackid].exhibit_pointsx = []
                    trajectory_dict[trackid].exhibit_pointsy = []
            if pred:
                predictions = predict(dataset, anno_data, frameid)
                for i, trackid in enumerate(predictions[:,0]):
                    trajectory_dict[trackid].pred_pointsx.append(predictions[i, 1])
                    trajectory_dict[trackid].pred_pointsy.append(predictions[i, 2])

            yield im, exists_trackids

    def update_frame(new_frame):
        handlers = []
        image_handler.set_array(new_frame[0])
        handlers.append(image_handler)
        for trackid in new_frame[1]:
            trajectory_dict[trackid].handler.set_data(trajectory_dict[trackid].exhibit_pointsx,
                                                      trajectory_dict[trackid].exhibit_pointsy)
            handlers.append(trajectory_dict[trackid].handler)
            if pred:
                trajectory_dict[trackid].pred_handler.set_data(trajectory_dict[trackid].pred_pointsx[-10:],
                                                               trajectory_dict[trackid].pred_pointsy[-10:])
                handlers.append(trajectory_dict[trackid].pred_handler)
        return handlers

    anim = animation.FuncAnimation(fig, update_frame, frames=get_frame, interval=5, repeat=False)
    plt.show()
    plt.close()


def read_anno(anno_file, image_size):
    """ read annotations and rearange them for display
    :param anno_file:       annotation filepath, csv format, each column is an item
    :param image_size:      H, W
    :return:
    """
    with open(anno_file, 'r') as f:
        lines = f.readlines()
        anno_dataT = np.array([map(float, items.split(',')) for items in lines]).astype(np.float32)
        anno_data = anno_dataT.transpose((1, 0))
        anno_data = anno_data[:, [0, 1, 3, 2]]    # x, y
        anno_data[:, 2] = (1 + anno_data[:, 2]) * image_size[1]/2.0
        anno_data[:, 3] = (1 + anno_data[:, 3]) * image_size[0]/2.0
    return anno_data


def predict(dataset, anno_data, frameId):
    """ An API for prediction
    :param anno_data:           annotation data
    :param frameId:             current frame id
    :param dataset:             current dataset
    :return
        preds:                  None or array of N x 3, where each row denotes [trackid, pos_x, pos_y]
    """
    preds = None
    return preds


if __name__=='__main__':
    args = parse()
    if args.dataset.startswith('seq_'):  # ETH
        data_dir = '/data/zwzhou/Data/Traj/ETH/ewap_dataset/{}'.format(args.dataset)
    else:  # UCY
        if args.dataset.startswith('zara'):
            data_dir = '/data/zwzhou/Data/Traj/UCY/zara/{}'.format(args.dataset)
        else:
            data_dir = '/data/zwzhou/Data/Traj/UCY/univ/'
    anno_file = osp.join(data_dir, 'pixel_pos_interpolate.csv'.format(args.dataset))
    init_img_path = osp.join(data_dir, 'frames/000001.jpg')
    init_image = cv2.cvtColor(cv2.imread(init_img_path), cv2.COLOR_BGR2RGB)
    image_size = init_image.shape[:2]
    if args.dataset == 'zara01':
        interval_frame=8
    elif args.dataset == 'zara02':
        interval_frame=12
    elif args.dataset == 'univ':
        interval_frame = 4
    else:
        interval_frame = 1
    anno_data = read_anno(anno_file, image_size)
    image_dir = osp.join(data_dir, 'frames')

    plot_sequence_images(image_dir, init_image, anno_data, args, interval_frame)
