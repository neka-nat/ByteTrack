import streamlit as st
from loguru import logger
import base64
import datetime

import cv2
import numpy as np
import pandas as pd

import torch

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

import argparse
import os
import time

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    # parser.add_argument("--camid", type=str, default="./videos/20211119_nonoichi.mkv", help="webcam demo camera id")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/mot/yolox_x_mix_det.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def imageflow_demo(predictor, args):
    max_length = 100000
    image_loc = st.empty()
    table = st.empty()
    download_link = st.empty()
    n_video = 8
    cols = st.columns(3) + st.columns(3) + st.columns(3)
    charts = []
    for i, col in enumerate(cols[:n_video]):
        with col:
            charts.append(st.line_chart(pd.DataFrame(np.zeros((1,)), columns=[f"cam{i}"]), height=120))
    columns = ["time"] + [f"count_video_{i}" for i in range(n_video)]
    df = pd.DataFrame(columns=columns)
    cap = cv2.VideoCapture(args.camid)
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    id_list = set()
    range_counts = np.zeros((3, 3))
    start_time = datetime.datetime.now()
    counts_prev = None
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
                timer.toc()
                online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                          fps=1. / timer.average_time)
            else:
                timer.toc()
                online_im = img_info['raw_img']
            image_loc.image(online_im[:, :, ::-1], output_format="PNG")
            counts = np.zeros((3, 3))
            w_u = img_info['width'] // 3
            h_u = img_info['height'] // 3
            for (tlx, tly, w, h), online_id in zip(online_tlwhs, online_ids):
                center = (tlx + w / 2, tly + h / 2)
                counts[min(2, int(center[1] / h_u)), min(2, int(center[0] / w_u))] += 1
                if online_id not in id_list:
                    id_list.add(online_id)
                    range_counts[min(2, int(center[1] / h_u)), min(2, int(center[0] / w_u))] += 1
            counts = counts.reshape(1, 9)[:, :8]
            if datetime.datetime.now() - start_time >= datetime.timedelta(seconds=1800):
                tmp_counts = range_counts.reshape(1, 9)[:, :8]
                for i, chart in enumerate(charts):
                    chart.add_rows(pd.DataFrame(tmp_counts[:, i], columns=[f"cam{i}"]))
                start_time = datetime.datetime.now()
                id_list = set()
                range_counts = np.zeros((3, 3))
            if counts_prev is None or abs(counts - counts_prev).sum() > 0:
                df = df.append(pd.Series([datetime.datetime.now()] + counts[0].astype(int).tolist(),
                                         index=columns),
                               ignore_index=True)
                table.dataframe(df)
                download_link.markdown(get_table_download_link(df), unsafe_allow_html=True)
            if len(df) > max_length:
                df = df.iloc[(max_length // 2):, :]
            counts_prev = counts
        else:
            break
        frame_id += 1


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)
    
    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    imageflow_demo(predictor, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
