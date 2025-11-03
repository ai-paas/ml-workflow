import importlib
import os
import sys
from functools import lru_cache
from io import BytesIO
from typing import IO

import cv2
import numpy as np
import torch

# YOLOX 패키지 import를 위해 프로젝트 루트 경로 추가
# predictor/app/model_manager/yolox에서 프로젝트 루트로 가려면 ../../../../ (4단계 위)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logging_inference import logger
from model_manager.base import BaseModelManager

from YOLOX.yolox.data.data_augment import preproc

# YOLOX 폴더 내의 yolox 패키지 import
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.utils.boxes import postprocess
from YOLOX.yolox.utils.visualize import vis

FILE_INPUT_TYPE = "이미지"
ACCELERATOR = "cpu, gpu"


class YoloxModelManager(BaseModelManager):
    def __init__(self):
        """
        YOLOX 모델 매니저 초기화
        """
        super().__init__()
        self.model = None
        self.config_path = None
        self.device = None
        self.classes = COCO_CLASSES
        self.conf = 0.25
        self.iou = 0.45
        self.model_path = None
        self.exp = None

    def load_model(self, model_name: str, run_id: str):
        """
        YOLOX 모델 로드 로직 구현
        :param model_name: 모델 이름 (모델 가중치 파일 경로)
        :param run_id: MLflow run ID (모델 가중치 경로로 사용)
        """
        try:
            self.model_name = model_name
            self.run_id = run_id

            # MLflow에서 모델 가중치 다운로드
            model_weights_path = self._load_artifacts(run_id, model_name)
            self.model_path = model_weights_path

            # model_weights_path에서 파일명 추출하여 config_path 생성
            # 예: ~~/yolox_s.pth -> YOLOX.exps.default.yolox_s
            if os.path.isfile(model_weights_path):
                # 파일인 경우: 파일명에서 확장자 제거
                filename = os.path.basename(model_weights_path)
                model_name_without_ext = os.path.splitext(filename)[0]
            elif os.path.isdir(model_weights_path):
                # 디렉토리인 경우: 디렉토리 내 .pth 파일 찾기
                pth_files = [f for f in os.listdir(model_weights_path) if f.endswith(".pth")]
                if not pth_files:
                    raise ValueError(f"디렉토리 내 .pth 파일을 찾을 수 없습니다: {model_weights_path}")
                # 첫 번째 .pth 파일 사용
                filename = pth_files[0]
                model_name_without_ext = os.path.splitext(filename)[0]
                # 실제 모델 파일 경로 업데이트
                self.model_path = os.path.join(model_weights_path, filename)
            else:
                raise ValueError(f"모델 가중치 경로가 유효하지 않습니다: {model_weights_path}")

            # config_path 생성: YOLOX.exps.default.{모델명}
            self.config_path = f"YOLOX.exps.default.{model_name_without_ext}"

            logger.info(f"모델 가중치 경로: {self.model_path}")
            logger.info(f"Config 경로: {self.config_path}")

            # Config 모듈 import
            current_exp = importlib.import_module(self.config_path)
            self.exp = current_exp.Exp()

            # 모델 생성
            model = self.exp.get_model()

            # 디바이스 설정 (나중에 predict에서 실제로 설정됨)
            self.device = torch.device("cpu")  # 기본값, predict에서 업데이트됨

            model.to(self.device)
            model.eval()

            # 체크포인트 로드
            ckpt = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(ckpt["model"])

            self.model = model
            logger.info(f"YOLOX 모델 로드 완료: {self.config_path}, 가중치: {self.model_path}")

            return self.model

        except Exception as e:
            logger.error(f"YOLOX 모델 로드 실패: {e}")
            raise ValueError(f"YOLOX 모델 로드 실패: {str(e)}")

    def preprocess_data(self, data: str | bytes | IO[bytes], image_size: int = 640):
        """
        이미지 전처리 로직 구현
        :param data: 이미지 데이터 (바이너리 또는 파일 객체)
        :param image_size: 입력 이미지 크기
        :return: 전처리된 이미지 텐서와 비율 정보
        """
        try:
            # 이미지 읽기
            if isinstance(data, bytes):
                nparr = np.frombuffer(data, dtype=np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(data)

            if image is None:
                raise ValueError("이미지를 읽을 수 없습니다")

            # 원본 이미지 크기 저장
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            logger.info(f"원본 이미지 크기: {original_size}")

            # 전처리
            if image_size is not None:
                ratio = min(image_size / image.shape[0], image_size / image.shape[1])
                img, _ = preproc(image, input_size=(image_size, image_size))
            else:
                default_size = 640
                ratio = min(default_size / image.shape[0], default_size / image.shape[1])
                img, _ = preproc(image, input_size=(default_size, default_size))
                image_size = default_size

            # 텐서로 변환
            img = torch.from_numpy(img).to(self.device).unsqueeze(0).float()

            return img, ratio, original_size, image_size

        except Exception as e:
            logger.error(f"이미지 전처리 중 오류 발생: {e}")
            raise ValueError(f"이미지 전처리 실패: {str(e)}")

    def predict(self, data: str | bytes | IO[bytes], device_str: str):
        """
        YOLOX 모델 추론 로직 구현
        :param data: 입력 데이터 (이미지 바이트)
        :param device_str: 추론 장치 문자열 ("cpu" 또는 "gpu")
        :return: 예측 결과 딕셔너리
        """
        try:
            if not self.model or not self.exp:
                logger.error(f"ModelManager not ready: model={self.model}, exp={self.exp}")
                raise ValueError("모델 추론이 준비되지 않았습니다")

            # 디바이스 설정
            self.device = torch.device("cuda" if torch.cuda.is_available() and device_str == "gpu" else "cpu")
            self.model = self.model.to(self.device)

            logger.info(
                f"start YOLOX inference\n"
                f"  device={device_str}\n"
                f"  model={type(self.model)}\n"
                f"  data_type={type(data)}"
            )

            # 이미지 크기 가져오기 (exp에서 기본값 사용)
            image_size = getattr(self.exp, "test_size", 640)

            # 데이터 전처리
            img_tensor, ratio, original_size, processed_size = self.preprocess_data(data, image_size)

            # 모델 추론
            with torch.no_grad():
                prediction_result = self.model(img_tensor)

            # 후처리
            original_predictions = postprocess(
                prediction=prediction_result,
                num_classes=len(COCO_CLASSES),
                conf_thre=self.conf,
                nms_thre=self.iou,
            )[0]

            # 결과 처리
            output = original_predictions.cpu()
            bboxes = output[:, 0:4]
            bboxes /= ratio
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            # 결과를 리스트로 변환
            predictions = []
            for bbox, score, cls_id in zip(bboxes, scores, cls):
                predictions.append(
                    {
                        "score": float(score.item()),
                        "label": (
                            COCO_CLASSES[int(cls_id.item())]
                            if int(cls_id.item()) < len(COCO_CLASSES)
                            else f"unknown_{int(cls_id.item())}"
                        ),
                        "box": [float(x.item()) for x in bbox],  # [x1, y1, x2, y2]
                    }
                )

            # 신뢰도 기준으로 정렬 및 상위 결과만 반환
            predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)[:50]  # 최대 50개

            logger.info(f"YOLOX 추론 완료: {len(predictions)}개 객체 검출")

            result = {
                "predictions": predictions,
                "image_info": {
                    "original_size": {"width": original_size[0], "height": original_size[1]},
                    "model_input_size": {"width": processed_size, "height": processed_size},
                },
            }

            return result

        except Exception as e:
            logger.error(f"YOLOX 추론 중 오류 발생: {e}")
            raise e

    def _clear_model(self):
        """
        메모리 정리
        """
        if self.model is not None:
            del self.model
        self.model = None
        self.exp = None
        self.config_path = None
        self.model_path = None
        self.device = None


@lru_cache
def get_model_manager():
    model_manager = YoloxModelManager()
    return model_manager


model_manager = get_model_manager()
