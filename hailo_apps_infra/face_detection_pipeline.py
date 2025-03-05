import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from hailo_apps_infra.hailo_rpi_common import (
    get_default_parser,
    detect_hailo_arch,
)
from hailo_apps_infra.gstreamer_helper_pipelines import(
    QUEUE,
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    OVERLAY_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
)
from hailo_apps_infra.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback
)



# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerFaceDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data):
        parser = get_default_parser()
        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to costume labels JSON file",
        )
        args = parser.parse_args()
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.vdevice_group_id=1
        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45


        # Determine the architecture if not specified
        if args.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = args.arch

        # Set the HEF file path based on the arch
        if self.arch == "hailo8":
            self.yolo_hef_path = os.path.join(self.current_path, '../resources/yolov8m.hef')
        else:  # hailo8l
            self.yolo_hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')

        # Set the post-processing shared object file
        self.yolo_post_process_so = os.path.join(self.current_path, '../resources/libyolo_hailortpp_postprocess.so')
        self.yolo_post_function_name = "filter_letterbox"
        # User-defined label JSON file
        self.labels_json = os.path.join(self.current_path, '../resources/libyolo_hailortpp_postprocess.so')

        self.face_detection_hef_path = os.path.join(self.current_path, '../resources/scrfd_10g.hef')
        self.face_detection_config = os.path.join(self.current_path, '../resources/scrfd.json')
        self.face_detection_post = os.path.join('/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libscrfd_post.so')

        self.vms_cropper_so = os.path.join('/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libvms_croppers.so')
        self.face_align_post = os.path.join(self.current_path, '../resources/libvms_face_align.so')
        self.face_recognition_hef_path = os.path.join(self.current_path, '../resources/arcface_mobilefacenet_v1.hef')
        self.face_recognition_post = os.path.join('/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libface_recognition_post.so')
        self.local_gallery_file = os.path.join(self.current_path, '../resources/face_recognition_local_gallery.json')

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the process title
        setproctitle.setproctitle("Hailo Detection App")

        self.create_pipeline()

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(self.video_source, self.video_width, self.video_height)

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{self.pre_detector_pipeline()}'
            f'{self.object_detection_pipeline()}'
            f'{self.face_detection_pipeline()}'
            f'{self.face_tracker_pipeline()}'
            f'{self.face_recognition_pipeline()}'
            f'{self.embeddings_gallery_pipeline()}'
            f'{self.overlay_pipeline()}'
            f'{self.user_callback_pipeline()} ! '
            f'{self.display_pipeline()}'
        )
        print(pipeline_string)
        return pipeline_string
    
    def pre_detector_pipeline(self):
        pre_detection_pipeline = (
            f'queue name=pre_detector_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'tee name=t hailomuxer name=hmux t. ! '
            f'queue name=detector_bypass_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hmux. t. ! '
            f'videoscale name=face_videoscale method=0 n-threads=2 add-borders=false qos=false ! '
            f'video/x-raw, pixel-aspect-ratio=1/1 ! '
        )
        return pre_detection_pipeline
    
    def object_detection_pipeline(self):
        object_detection_pipeline = (
            f'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet hef-path={self.yolo_hef_path} scheduling-algorithm=1 vdevice_group_id={self.vdevice_group_id} \
                batch-size=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! '
            f'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter function-name={self.yolo_post_function_name} so-path={self.yolo_post_process_so} qos=false ! '
        )
        return object_detection_pipeline
    
    def face_detection_pipeline(self):
        face_detection_pipeline = (
            f'queue name=pre_face_detector_infer_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet hef-path={self.face_detection_hef_path} scheduling-algorithm=1 vdevice_group_id={self.vdevice_group_id} ! '
            f'queue name=detector_post_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter so-path={self.face_detection_post} name=face_detection_hailofilter qos=false config-path={self.face_detection_config} function_name=scrfd_10g ! '
            f'queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hmux. hmux. ! '
        )
        return face_detection_pipeline
    
    def face_tracker_pipeline(self):
        tracker_pipeline= (
            f'queue name=pre_tracker_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailotracker name=hailo_face_tracker class-id=-1 kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.9 \
                    keep-new-frames=2 keep-tracked-frames=6 keep-lost-frames=8 keep-past-metadata=true qos=false ! '
            f'queue name=hailo_post_tracker_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
        )
        return tracker_pipeline
    
    def face_recognition_pipeline(self):
        inference_pipeline = (
            f'hailocropper so-path={self.vms_cropper_so} function-name=face_recognition internal-offset=true name=cropper2 hailoaggregator name=agg2 cropper2. ! '
            f'queue name=bypess2_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg2. cropper2. ! '
            f'queue name=pre_face_align_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter so-path={self.face_align_post} name=face_align_hailofilter use-gst-buffer=true qos=false ! '
            f'queue name=detector_pos_face_align_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet hef-path={self.face_recognition_hef_path} scheduling-algorithm=1 vdevice_group_id={self.vdevice_group_id} ! '
            f'queue name=recognition_post_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter function-name=arcface_rgb so-path={self.face_recognition_post} name=face_recognition_hailofilter qos=false ! '
            f'queue name=recognition_pre_agg_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg2. agg2. ! '
        )

        return inference_pipeline
    
    def embeddings_gallery_pipeline(self):
        gallery_pipeline = (
            f'queue name=hailo_pre_gallery_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailogallery gallery-file-path={self.local_gallery_file} load-local-gallery=true similarity-thr=.4 gallery-queue-size=20 class-id=-1 ! '
        )
        return gallery_pipeline
    
    def overlay_pipeline(self):
        user_callback_pipeline = (
            f'queue name=hailo_pre_draw2 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'hailooverlay name=hailo_overlay qos=false show-confidence=false local-gallery=false line-thickness=5 font-thickness=2 landmark-point-radius=8 ! '
        )
        return user_callback_pipeline
    
    def user_callback_pipeline(self):
        user_callback_pipeline = (
            f'{QUEUE(name=f"identity_callback_q")} ! '
            f'identity name=identity_callback '
        )
        return user_callback_pipeline
    
    def display_pipeline(self):
        display_pipeline = (
            f'queue name=hailo_post_draw leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'videoconvert name=sink_videoconvert n-threads=2 qos=false ! '
            f'queue name=hailo_display_q_0 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
            f'fakevideosink sync=false '
        )
        return display_pipeline
    
if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerFaceDetectionApp(app_callback, user_data)
    app.run()
