from torchvision import datasets, transforms
import scipy.io
import warnings
import matplotlib.pyplot as plt
from utils import *
import threading

warnings.filterwarnings("ignore")

from darknet import Darknet
from utils import *
from MeshPly import MeshPly


class Yolo6dDetector:
    def __init__(self):
        print('initialization........')
        self.cfgfile = 'cfg/yolo-pose.cfg'
        self.outfile = 'output_file'
        self.object_weight = 'backup/small_duck/model.weights'
        self.ply_model = './LINEMOD/small_duck/small_duck.ply'
        self.model = Darknet(self.cfgfile)
        self.num_classes = 1
        self.conf_thresh = 0.1
        self.test_width = 544
        self.test_height = 544
        self.R_pr = None
        self.r_pr = None
        self.Rt_pr = None
        self.visualize = True
        self.data = None

    def loadData(self,empty_one,empty_two):
        print("in python detector loadData\n")
        print("%s\n"%empty_one)
        print("%s\n"%empty_two)
        self.model.load_weights(self.object_weight)
        self.model.eval()
        self.mesh = MeshPly(self.ply_model)
        self.vertices = np.c_[np.array(self.mesh.vertices),
                              np.ones((len(self.mesh.vertices), 1))].transpose()
        self.corners3D = get_3D_corners(self.vertices)
        self.internal_calibration = get_camera_intrinsic() # Read intrinsic camera parameters
        self.edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                         [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]


    def setColorImg(self,img):
        print("in python detector setColorImg\n")
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        img = img.resize((self.test_width, self.test_height))
        transform = transforms.Compose([transforms.ToTensor(), ])
        self.data = Variable(transform(img).view(1,3,544,544))

        # Images
        img_show = self.data[0, :, :, :]
        img_show = img_show.numpy().squeeze()
        self.img_show = np.transpose(img_show, (1, 2, 0))
        # print('successful')
        # print(img)

    def detection(self):
        print("in python detector detection\n")

        #forward pass
        output = self.model(self.data).cuda()

        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, self.conf_thresh, self.num_classes)

        # Iterate through all images in the batch
        for i in range(output.size(0)):
            # For each image, get all the predictions
            boxes = all_boxes[i]
            best_conf_est = -1

            # If the prediction has the highest confidence, choose it as our prediction for single object pose estimation
            for j in range(len(boxes)):
                if (boxes[j][18] > best_conf_est):
                    # match = corner_confidence9(box_gt[:18], torch.FloatTensor(boxes[j][:18]))
                    box_pr = boxes[j]
                    best_conf_est = boxes[j][18]

            # Denormalize the corner predictions
            corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
            corners2D_pr[:, 0] = corners2D_pr[:, 0] * 1920
            corners2D_pr[:, 1] = corners2D_pr[:, 1] * 1080

            # Compute [R|t] by pnp

            self.R_pr, self.t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)),
                                          dtype='float32'), corners2D_pr,
                                 np.array(self.internal_calibration, dtype='float32'))
            # self.R_pr.append(R_pr)
            # self.r_pr.append(t_pr)

            # # Compute pixel error
            self.Rt_pr = np.concatenate((self.R_pr, self.t_pr), axis=1)

            # proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration)
            proj_corners_pr = np.transpose(compute_projection(self.corners3D, self.Rt_pr, self.internal_calibration))

            if self.visualize:
                # Visualize
                plt.xlim((0, 1920))
                plt.ylim((0, 1080))
                plt.imshow(scipy.misc.imresize(self.img_show, (1080, 1920)))
                # Projections
                for edge in self.edges_corners:
                    # plt.plot(proj_corners_gt[edge, 0], proj_corners_gt[edge, 1], color='g', linewidth=3.0)
                    plt.plot(proj_corners_pr[edge, 0], proj_corners_pr[edge, 1], color='b', linewidth=3.0)
                plt.gca().invert_yaxis()
                plt.show()


    def getReuslt(self):
        print("in python detector getReuslt\n")
        print(self.R_pr)
        print(self.t_pr)



    def setDepthImg(self, img):


        print("in python detector setColorImg\n")
        #cv2.imshow('test',img)
        #cv2.waitKey(0)

if __name__ == '__main__':

    img1 = cv2.imread('000023.jpg') #目前默认数据输入的格式是OpenCV读取图片的形式，例如下面所示：

    x = Yolo6dDetector()
    x.loadData('1111','2222')
    a = x.setColorImg(img1) #img1 是传入的数据
    x.detection()
    x.getReuslt()

    # t =threading.Thread(target=run_test)
    # t.start()