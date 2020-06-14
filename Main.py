import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from darknet import Darknet
import dataset
from utils import *
from MeshPly import MeshPly
import cv2
preds_trans        = []
preds_rot          = []
preds_corners2D    = []
gts_trans = []
gts_rot = []
gts_corners2D = []

# Create new directory
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)



def valid(datacfg, cfgfile, weightfile, outfile):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Parse configuration files
    options = read_data_cfg(datacfg)

    valid_images = options['valid']
    meshname = options['mesh']
    backupdir = options['backup']
    name = options['name']
    if not os.path.exists(backupdir):
        makedirs(backupdir)

    # Parameters
    prefix = 'results'
    seed = int(time.time())
    gpus = '0'  # Specify which gpus to use
    test_width = 544
    test_height = 544
    torch.manual_seed(seed)
    use_cuda = True
    visualize = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    save = False
    testtime = True
    use_cuda = True
    num_classes = 1
    eps = 1e-5
    notpredicted = 0
    conf_thresh = 0.1
    nms_thresh = 0.4
    match_thresh = 0.5
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    if save:
        makedirs(backupdir + '/test')
        makedirs(backupdir + '/test/gt')
        makedirs(backupdir + '/test/pr')


    # Read object model information, get 3D bounding box corners
    mesh = MeshPly(meshname)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(vertices)
    # diam          = calc_pts_diameter(np.array(mesh.vertices))

    diam = float(options['diam'])

    # Read intrinsic camera parameters
    internal_calibration = get_camera_intrinsic()

    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(cfgfile)
    # model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()

    # Get the parser for the test dataset
    valid_dataset = dataset.listDataset(valid_images, shape=(test_width, test_height),
                                        shuffle=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(), ]))


    valid_batchsize = 1

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs)

    logging("   Testing {}...".format(name))
    logging("   Number of test samples: %d" % len(test_loader.dataset))
    # Iterate through test batches (Batch size for test data is 1)
    count = 0
    z = np.zeros((3, 1))
    print("--------------------------------------------")
    for batch_idx, (data, target) in enumerate(test_loader):
        # Images
        print("-----------------idx---------------------------",batch_idx)
        print("-------------------data-------------------------", data.shape)
        print("--------------------target------------------------", target.shape)
        img = data[0, :, :, :]
        img = img.numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))

        # Pass data to GPU
        if use_cuda:
            data = data.cuda()

            target = target.cuda()

        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)


        # Forward pass
        output = model(data).data

        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, conf_thresh, num_classes)


        # Iterate through all images in the batch
        for i in range(output.size(0)):

            # For each image, get all the predictions
            boxes = all_boxes[i]

            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths = target[i].view(-1, 21)

            # Get how many object are present in the scene
            num_gts = truths_length(truths)

            # Iterate through each ground-truth object
            for k in range(num_gts):
                box_gt = [truths[k][1], truths[k][2], truths[k][3], truths[k][4], truths[k][5], truths[k][6],
                          truths[k][7], truths[k][8], truths[k][9], truths[k][10], truths[k][11], truths[k][12],
                          truths[k][13], truths[k][14], truths[k][15], truths[k][16], truths[k][17], truths[k][18], 1.0,
                          1.0, truths[k][0]]
                best_conf_est = -1

                # If the prediction has the highest confidence, choose it as our prediction for single object pose estimation
                for j in range(len(boxes)):
                    if (boxes[j][18] > best_conf_est):

                        match = corner_confidence9(box_gt[:18], torch.FloatTensor(boxes[j][:18]))

                        box_pr = boxes[j]

                        best_conf_est = boxes[j][18]

                # Denormalize the corner predictions
                corners2D_gt = np.array(np.reshape(box_gt[:18], [9, 2]), dtype='float32')
                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * 1920
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * 1080
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * 1920
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * 1080
                # corners2D_gt[:, 0] = corners2D_gt[:, 0] * 640
                # corners2D_gt[:, 1] = corners2D_gt[:, 1] * 480
                # corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
                # corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480
                preds_corners2D.append(corners2D_pr)
                gts_corners2D.append(corners2D_gt)
                # print("corners gt:\n",gts_corners2D)
                # print("corners pr:\n", preds_corners2D)
                # Compute [R|t] by pnp
                R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                          dtype='float32'), corners2D_gt,
                                 np.array(internal_calibration, dtype='float32'))
                R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                          dtype='float32'), corners2D_pr,
                                 np.array(internal_calibration, dtype='float32'))
                #Eulerangle




                # # Compute pixel error
                Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
                Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
                proj_2d_gt = compute_projection(vertices, Rt_gt, internal_calibration)
                proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration)
                proj_corners_gt = np.transpose(compute_projection(corners3D, Rt_gt, internal_calibration))
                proj_corners_pr = np.transpose(compute_projection(corners3D, Rt_pr, internal_calibration))

                if visualize:
                    # Visualize
                    plt.xlim((0, 1920))
                    plt.ylim((0, 1080))
                    plt.imshow(scipy.misc.imresize(img, (1080, 1920)))
                    plt.plot(corners2D_gt[1][0], corners2D_gt[1][1], 'r*')
                    plt.plot(corners2D_gt[2][0], corners2D_gt[2][1], 'r*')
                    plt.plot(corners2D_gt[3][0], corners2D_gt[3][1], 'r*')
                    plt.plot(corners2D_gt[4][0], corners2D_gt[4][1], 'r*')
                    plt.plot(corners2D_gt[5][0], corners2D_gt[5][1], 'r*')
                    plt.plot(corners2D_gt[6][0], corners2D_gt[6][1], 'r*')
                    plt.plot(corners2D_gt[7][0], corners2D_gt[7][1], 'r*')
                    plt.plot(corners2D_gt[8][0], corners2D_gt[8][1], 'r*')

                    # plt.plot(corners2D_pr[0][0], corners2D_pr[0][1], 'b*')
                    # plt.xlim((0, 640))
                    # plt.ylim((0, 480))
                    # plt.imshow(scipy.misc.imresize(img, (480, 640)))
                    # Projections
                    for edge in edges_corners:
                        # plt.plot(proj_corners_gt[edge, 0], proj_corners_gt[edge, 1], color='g', linewidth=2.0)
                        plt.plot(corners2D_gt[edge, 0], corners2D_gt[edge, 1], color='r', linewidth=2.0)
                        plt.plot(proj_corners_pr[edge, 0], proj_corners_pr[edge, 1], color='b', linewidth=2.0)
                    plt.gca().invert_yaxis()
                    # print(corners2D_pr[0])
                    # print(corners2D_pr)

                    # print(EulerAngle)
                    # plt.show()

                    print("_________")
                    plt.show()


if __name__ == '__main__':
    # import sys
    # if len(sys.argv) == 4:
    #     datacfg = sys.argv[1]
    #     cfgfile = sys.argv[2]
    #     weightfile = sys.argv[3]
    #     outfile = 'comp4_det_test_'
    #     valid(datacfg, cfgfile, weightfile, outfile)
    # else:
    #     print('Usage:')
    #     print(' python valid.py datacfg cfgfile weightfile')
    # datacfg = 'cfg/milk.data'
    # cfgfile = 'cfg/yolo-pose.cfg'
    # weightfile = 'backup/milk/model.weights'
    # datacfg = 'cfg/small_duck.data'
    # cfgfile = 'cfg/yolo-pose.cfg'
    # weightfile = 'backup/small_duck/model.weights'

    datacfg = 'cfg/retangle.data'
    cfgfile = 'cfg/yolo-pose.cfg'
    weightfile = 'backup/retangle/model.weights'
    outfile = 'comp4_det_test_'
    valid(datacfg, cfgfile, weightfile, outfile)
