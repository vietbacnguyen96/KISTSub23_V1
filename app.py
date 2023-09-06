from flask import Flask, render_template, Response, jsonify
import cv2
import requests
import base64
import json                    
import time
import threading
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm


from utils.service.TFLiteFaceAlignment import * 
from utils.service.TFLiteFaceDetector import * 
from utils.functions import *

app = Flask(__name__)


parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25',
                    help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true",
                    default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02,
                    type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4,
                    type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true",
                    default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.8, type=float,
                    help='visualization_threshold')
args = parser.parse_args()

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



# path = "/home/vkist1/frontend_facerec_VKIST/"
path = "./"

fd_0 = UltraLightFaceDetecion(path + "utils/service/weights/RFB-320.tflite", conf_threshold=0.98)
fd_1 = UltraLightFaceDetecion(path + "utils/service/weights/RFB-320.tflite", conf_threshold=0.98)
fa_0 = CoordinateAlignmentModel(path + "utils/service/weights/coor_2d106.tflite")
fa_1 = CoordinateAlignmentModel(path + "utils/service/weights/coor_2d106.tflite")

# url = 'http://192.168.0.100:5052/'
url = 'https://dohubapps.com/user/daovietanh190499/5000/'

api_list = [url + 'facerec', url + 'FaceRec_DREAM', url + 'FaceRec_3DFaceModeling', url + 'check_pickup']
api_index = 0
extend_pixel = 50
crop_image_size = 100


# ------------VKIST ---------------
# vkist_6
# secret_key = '13971a9f-1b2d-46bb-b829-d395431448fd'

# ----------- HHSC -----------------------
# hhsc_3
secret_key = "6c24a661-7bc6-4c28-b057-8c4919285205"


predict_labels = []

def face_recognize(frame, isTinyFace=False):
    global api_index

    _, encimg = cv2.imencode(".jpg", frame)
    img_byte = encimg.tobytes()
    img_str = base64.b64encode(img_byte).decode('utf-8')
    new_img_str = "data:image/jpeg;base64," + img_str
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}

    payload = json.dumps({"secret_key": secret_key,
                            'local_register' : isTinyFace,
                            "img": new_img_str,
                             })

    response = requests.post(api_list[api_index], data=payload, headers=headers, timeout=100)

    try:
        # for id, name, picker_name, profileID, picker_profile_face_id, timestamp in zip( 
        for id, name, profileID, timestamp in zip( 
                                                                                        response.json()['result']['id'],
                                                                                        response.json()['result']['identities'],
                                                                                        # response.json()['result']['picker_profile_names'],
                                                                                        response.json()['result']['profilefaceIDs'],
                                                                                        # response.json()['result']['pickerProfileFaceIds'],
                                                                                        response.json()['result']['timelines']
                                                                                        ):
            print('Server response', response.json()['result']['identities'])
            if id != -1:
                # response_time_s = time.time() - seconds
                # print("Server's response time: " + "%.2f" % (response_time_s) + " (s)")
                # print('picker_profile_face_id', picker_profile_face_id)
                cur_profile_face = None
                cur_picker_profile_face = None

                if profileID is not None:
                    cur_url = url + 'images/' + secret_key + '/' + profileID
                    cur_profile_face = np.array(Image.open(requests.get(cur_url, stream=True).raw))
                    # cur_profile_face = cv2.resize(cur_profile_face, (crop_image_size, crop_image_size))
                    cur_profile_face = cv2.cvtColor(cur_profile_face, cv2.COLOR_BGR2RGB)

                    _, encimg = cv2.imencode(".jpg", cur_profile_face)
                    img_byte = encimg.tobytes()
                    img_str = base64.b64encode(img_byte).decode('utf-8')
                    cur_profile_face = "data:image/jpeg;base64," + img_str


                # if picker_profile_face_id is not None:
                #     cur_url = url + 'images/' + secret_key + '/' + picker_profile_face_id
                #     cur_picker_profile_face = np.array(Image.open(requests.get(cur_url, stream=True).raw))
                #     # cur_picker_profile_face = cv2.resize(cur_picker_profile_face, (crop_image_size, crop_image_size))
                #     cur_picker_profile_face = cv2.cvtColor(cur_picker_profile_face, cv2.COLOR_BGR2RGB)

                    # _, encimg = cv2.imencode(".jpg", cur_picker_profile_face)
                    # img_byte = encimg.tobytes()
                    # img_str = base64.b64encode(img_byte).decode('utf-8')
                    # cur_picker_profile_face = "data:image/jpeg;base64," + img_str

                frame = cv2.resize(frame, (crop_image_size, crop_image_size))
                _, encimg = cv2.imencode(".jpg", frame)
                img_byte = encimg.tobytes()
                img_str = base64.b64encode(img_byte).decode('utf-8')
                new_img_str = "data:image/jpeg;base64," + img_str

                # predict_labels.append([id, name, picker_name, new_img_str, cur_profile_face, cur_picker_profile_face, timestamp])
                predict_labels.append([id, name, new_img_str, cur_profile_face, timestamp])

    except requests.exceptions.RequestException:
        print(response.text)

def get_frame_0():
    # Open the webcam stream
    webcam_0 = cv2.VideoCapture(0)

    frame_width = int(webcam_0.get(3))
    frame_height = int(webcam_0.get(4))

    prev_frame_time = 0
    new_frame_time = 0
    queue = []

    count = 0
    frequency = 4

    while True:
        count += 1
        # Read a frame from the stream
        ret, orig_image = webcam_0.read()
        # orig_image = cv2.flip(orig_image, 1)
        final_frame = orig_image.copy()
        if (count % frequency) == 0:
            count = 0

            temp_boxes, _ = fd_0.inference(orig_image)

            # Draw boundary boxes around faces
            draw_box(final_frame, temp_boxes, color=(125, 255, 125))

            # Find landmarks of each face
            temp_marks = fa_0.get_landmarks(orig_image, temp_boxes)

            # -------------------------------------- Draw landmarks of each face ---------------------------------------------
            # for bbox_I, landmark_I in zip(temp_boxes, temp_resized_marks):
            #     landmark_I = landmark_I * (1 / scale_ratio)
            #     draw_landmark(final_frame, landmark_I, color=(125, 255, 125))

            #     # Show rotated raw face image
            #     xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])
            #     xmin -= extend_pixel
            #     xmax += extend_pixel
            #     ymin -= 2 * extend_pixel
            #     ymax += extend_pixel

            #     xmin = 0 if xmin < 0 else xmin
            #     ymin = 0 if ymin < 0 else ymin
            #     xmax = frame_width if xmax >= frame_width else xmax
            #     ymax = frame_height if ymax >= frame_height else ymax

            #     face_I = orig_image[ymin:ymax, xmin:xmax]
            #     face_I = align_face(face_I, landmark_I[34], landmark_I[88])

            #     cv2.imshow('Rotated raw face image', face_I)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # ----------------------------------------------------------------------------------------------------------------

            for bbox_I, landmark_I in zip(temp_boxes, temp_marks):
                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])

                xmin -= extend_pixel
                xmax += extend_pixel
                ymin -= extend_pixel
                ymax += extend_pixel

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                xmax = frame_width if xmax >= frame_width else xmax
                ymax = frame_height if ymax >= frame_height else ymax

                face_I = orig_image[ymin:ymax, xmin:xmax]
                rotated_face_I = align_face(face_I, landmark_I[34], landmark_I[88])

                # --------------------------------- Show rotated resized face image ----------------------------------------------
                # cv2.imshow('Rotated resized face image', rotated_face_I)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # ----------------------------------------------------------------------------------------------------------------

                queue = [t for t in queue if t.is_alive()]
                if len(queue) < 3:
                    queue.append(threading.Thread(target=face_recognize, args=(rotated_face_I,)))
                    queue[-1].start()

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))

            cv2.putText(final_frame, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

            # Convert the frame to a jpeg image
            ret, jpeg = cv2.imencode('.jpg', final_frame)

            # Return the image as bytes
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def get_frame_1():
    # Open the webcam stream
    webcam_1 = cv2.VideoCapture('rtsp://admin:pilot2214@192.168.50.14:554/Streaming/channels/1/')
    # webcam_1 = cv2.VideoCapture(0)

    frame_width = int(webcam_1.get(3))
    frame_height = int(webcam_1.get(4))

    prev_frame_time = 0
    new_frame_time = 0
    queue = []

    count = 0
    frequency = 12

    while True:
        count += 1
        # Read a frame from the stream
        ret, orig_image = webcam_1.read()
        # orig_image = cv2.flip(orig_image, 1)
        final_frame = orig_image.copy()
        if (count % frequency) == 0:
            count = 0

            img = np.float32(orig_image)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            # tic = time.time()
            loc, conf, landms = net(img)  # forward pass
            # current_time = time.time()
            # print('net forward time: {:.4f}'.format(time.time() - tic))
            # fps =int(1/(current_time -tic ))
            # FPS = "{:.4f}".format(fps)


            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(
                0), prior_data, cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(
                0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # show image
            if args.save_image:
                for b in dets:
                    if b[4] < args.vis_thres:
                        continue
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))

                    xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    
                    w = xmax - xmin
                    h = ymax - ymin

                    ratio = 0.1
                    xmin -= int(w*ratio)
                    xmax += int(w*ratio) 
                    ymin -= int(h*ratio)
                    ymax += int(h*ratio)

                    xmin = 0 if xmin < 0 else xmin
                    ymin = 0 if ymin < 0 else ymin
                    xmax = frame_width if xmax >= frame_width else xmax
                    ymax = frame_height if ymax >= frame_height else ymax

                    # cv2.imwrite('facial_images/' + str(time.time()) + '.jpg', final_frame[ymin:ymax, xmin:xmax])
                    cv2.rectangle(final_frame, (xmin, ymin), (xmax, ymax), (125, 255, 125), 5)
                    # cx = b[0]
                    # cy = b[1] + 12
                    # cv2.putText(final_frame, text, (cx, cy),
                    #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    # cv2.putText(frame,FPS,(10,15),
                    #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))
                    # cv2.circle(final_frame, (b[5], b[6]), 1, (0, 0, 255), 4)
                    # cv2.circle(final_frame, (b[7], b[8]), 1, (0, 255, 255), 4)
                    # cv2.circle(final_frame, (b[9], b[10]), 1, (255, 0, 255), 4)
                    # cv2.circle(final_frame, (b[11], b[12]), 1, (0, 255, 0), 4)
                    # cv2.circle(final_frame, (b[13], b[14]), 1, (255, 0, 0), 4)

                    queue = [t for t in queue if t.is_alive()]
                    if len(queue) < 3:
                        queue.append(threading.Thread(target=face_recognize, args=(orig_image[ymin:ymax, xmin:xmax], True,)))
                        queue[-1].start()

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))

            cv2.putText(final_frame, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

            # Convert the frame to a jpeg image
            ret, jpeg = cv2.imencode('.jpg', final_frame)

            # Return the image as bytes
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def get_frame_2():
    # Open the webcam stream
    webcam_1 = cv2.VideoCapture('rtsp://admin:pilot2214@192.168.50.14:554/Streaming/channels/1/')
    # webcam_1 = cv2.VideoCapture(0)

    frame_width = int(webcam_1.get(3))
    frame_height = int(webcam_1.get(4))

    prev_frame_time = 0
    new_frame_time = 0
    queue = []

    count = 0
    frequency = 4

    while True:
        count += 1
        # Read a frame from the stream
        ret, orig_image = webcam_1.read()
        # orig_image = cv2.flip(orig_image, 1)
        final_frame = orig_image.copy()
        if (count % frequency) == 0:
            count = 0

            temp_boxes, _ = fd_1.inference(orig_image)

            # Draw boundary boxes around faces
            draw_box(final_frame, temp_boxes, color=(125, 255, 125))

            # Find landmarks of each face
            temp_marks = fa_1.get_landmarks(orig_image, temp_boxes)

            # ---------------------------------------- Draw landmarks of each face ---------------------------------------------
            # for bbox_I, landmark_I in zip(temp_boxes, temp_resized_marks):
            #     landmark_I = landmark_I * (1 / scale_ratio)
            #     draw_landmark(final_frame, landmark_I, color=(125, 255, 125))

            #     # Show rotated raw face image
            #     xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])
            #     xmin -= extend_pixel
            #     xmax += extend_pixel
            #     ymin -= 2 * extend_pixel
            #     ymax += extend_pixel

            #     xmin = 0 if xmin < 0 else xmin
            #     ymin = 0 if ymin < 0 else ymin
            #     xmax = frame_width if xmax >= frame_width else xmax
            #     ymax = frame_height if ymax >= frame_height else ymax

            #     face_I = orig_image[ymin:ymax, xmin:xmax]
            #     face_I = align_face(face_I, landmark_I[34], landmark_I[88])

            #     cv2.imshow('Rotated raw face image', face_I)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # ----------------------------------------------------------------------------------------------------------------

            for bbox_I, landmark_I in zip(temp_boxes, temp_marks):
                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])

                xmin -= extend_pixel
                xmax += extend_pixel
                ymin -= extend_pixel
                ymax += extend_pixel

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                xmax = frame_width if xmax >= frame_width else xmax
                ymax = frame_height if ymax >= frame_height else ymax

                face_I = orig_image[ymin:ymax, xmin:xmax]
                rotated_face_I = align_face(face_I, landmark_I[34], landmark_I[88])

                # ------------------------------- Show rotated resized face image ----------------------------------------------
                # cv2.imshow('Rotated resized face image', rotated_face_I)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # ----------------------------------------------------------------------------------------------------------------

                queue = [t for t in queue if t.is_alive()]
                if len(queue) < 3:
                    queue.append(threading.Thread(target=face_recognize, args=(rotated_face_I,)))
                    queue[-1].start()

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))

            cv2.putText(final_frame, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

            # Convert the frame to a jpeg image
            ret, jpeg = cv2.imencode('.jpg', final_frame)

            # Return the image as bytes
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_0')
def video_feed_0():
    return Response(get_frame_0(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_1')
def video_feed_1():
    return Response(get_frame_1(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_2')
def video_feed_2():
    return Response(get_frame_2(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    global predict_labels
    if len(predict_labels) > 3:
        predict_labels = predict_labels[-3:]
    newest_data = list(reversed(predict_labels))
    return jsonify({'info': newest_data})

if __name__ == '__main__':
    if not os.path.exists('facial_images'):
        os.makedirs('facial_images')


    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    # elif args.network == "resnet50":
    #     cfg = cfg_re50
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    app.run(debug=True)
