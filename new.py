from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path
from utils.sort import Sort
from utils.actions import actionPredictor
from utils.joint_preprocess import *
import sys
import cv2
import numpy as np
import time
import settings

tracker = Sort(settings.sort_max_age, settings.sort_min_hit)
poseEstimator = TfPoseEstimator(
        get_graph_path('mobilenet_thin'), target_size=(432, 368))


#当前为人体姿态估计模式
def zi(show):
    humans = poseEstimator.inference(show)
    show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)
    return show
#当前为多人跟踪模式

def more(show):
	humans = poseEstimator.inference(show)
	show, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
	height = show.shape[0]                   
	width = show.shape[1]
	if bboxes:
		result = np.array(bboxes)
		det = result[:, 0:5]
		det[:, 0] = det[:, 0] * width
		det[:, 1] = det[:, 1] * height
		det[:, 2] = det[:, 2] * width
		det[:, 3] = det[:, 3] * height
		trackers = tracker.update(det)

		for d in trackers:
			xmin = int(d[0])
			ymin = int(d[1])
			xmax = int(d[2])
			ymax = int(d[3])
			label = int(d[4])
			cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
										  (int(settings.c[label % 32, 0]),
										   int(settings.c[label % 32, 1]),
										   int(settings.c[label % 32, 2])), 4)
	return show
#'当前为人体行为识别模式
def xiwei(show):
                current=[]
                previous=[]
                humans = poseEstimator.inference(show)
                ori = np.copy(show)
                show, joints, bboxes, xcenter, sk= TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
                height = show.shape[0]
                width = show.shape[1]
                if bboxes:
                    result = np.array(bboxes)
                    det = result[:, 0:5]
                    det[:, 0] = det[:, 0] * width
                    det[:, 1] = det[:, 1] * height
                    det[:, 2] = det[:, 2] * width
                    det[:, 3] = det[:, 3] * height
                    trackers = tracker.update(det)
                    current = [i[-1] for i in trackers]

                    if len(previous) > 0:
                        for item in previous:
                            if item not in current and item in data:
                                del data[item]
                            if item not in current and item in memory:
                                del memory[item]

                    previous = current
                    for d in trackers:
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2])
                        ymax = int(d[3])
                        label = int(d[4])
                        try:
                            j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                        except:
                            j = 0
                        if joint_filter(joints[j]):
                            joints[j] = joint_completion(joint_completion(joints[j]))
                            if label not in data:
                                data[label] = [joints[j]]
                                memory[label] = 0
                            else:
                                data[label].append(joints[j])

                            if len(data[label]) == settings.L:
                                pred = actionPredictor().move_status(data[label])
                                if pred == 0:
                                    pred = memory[label]
                                else:
                                    memory[label] = pred
                                data[label].pop(0)

                                location = data[label][-1][1]
                                if location[0] <= 30:
                                    location = (51, location[1])
                                if location[1] <= 10:
                                    location = (location[0], 31)

                                cv2.putText(show, settings.move_status[pred], (location[0] - 30, location[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 255, 0), 2)

                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)																   
                    return show								   
									   
									   
									   
									   
									   