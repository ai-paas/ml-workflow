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

from app.logging_inference import logger
from app.model_manager.base import BaseModelManager

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

            # 디바이스 설정 (나중에 predict에서 실제로 설정됨)
            self.device = torch.device("cpu")  # 기본값, predict에서 업데이트됨

            # 먼저 체크포인트를 로드하여 실제 클래스 수 확인
            ckpt = torch.load(self.model_path, map_location=self.device)

            # checkpoint에서 실제 클래스 수 추출
            # head.cls_preds.0.weight의 shape에서 클래스 수를 알 수 있음
            checkpoint_num_classes = None
            for key in ckpt["model"].keys():
                if "head.cls_preds.0.weight" in key:
                    # shape: [num_classes, 128, 1, 1]
                    checkpoint_num_classes = ckpt["model"][key].shape[0]
                    logger.info(f"체크포인트에서 감지된 클래스 수: {checkpoint_num_classes}")
                    break

            if checkpoint_num_classes is None:
                raise ValueError("체크포인트에서 클래스 수를 감지할 수 없습니다.")

            # Config 모듈 import
            current_exp = importlib.import_module(self.config_path)
            self.exp = current_exp.Exp()

            # 체크포인트의 클래스 수로 exp 설정
            logger.info(f"Exp의 기본 num_classes: {self.exp.num_classes}")
            self.exp.num_classes = checkpoint_num_classes
            logger.info(f"Exp의 num_classes를 {checkpoint_num_classes}로 설정")

            # 모델 생성
            model = self.exp.get_model()

            # 생성된 모델의 실제 클래스 수 확인
            if hasattr(model, "head") and hasattr(model.head, "num_classes"):
                logger.info(f"생성된 모델의 head.num_classes: {model.head.num_classes}")

                # 모델의 클래스 수와 checkpoint의 클래스 수가 다르면 head를 재생성
                if model.head.num_classes != checkpoint_num_classes:
                    logger.warning(
                        f"모델 클래스 수({model.head.num_classes})와 체크포인트 클래스 수({checkpoint_num_classes})가 불일치"
                    )
                    logger.info("모델 head를 재생성합니다...")

                    # YOLOX head를 올바른 클래스 수로 재생성
                    import torch.nn as nn

                    from YOLOX.yolox.models import YOLOXHead

                    # 기존 head의 설정 가져오기
                    in_channels = model.head.in_channels if hasattr(model.head, "in_channels") else [256, 512, 1024]
                    strides = model.head.strides if hasattr(model.head, "strides") else [8, 16, 32]

                    # 새로운 head 생성
                    model.head = YOLOXHead(
                        num_classes=checkpoint_num_classes,
                        strides=strides,
                        in_channels=in_channels,
                        act="silu",
                        depthwise=False,
                    )
                    logger.info(f"새로운 head 생성 완료. num_classes: {model.head.num_classes}")

            model.to(self.device)
            model.eval()

            # 체크포인트 로드
            logger.info("체크포인트 state_dict 로드 시작...")
            model.load_state_dict(ckpt["model"])
            logger.info("체크포인트 state_dict 로드 완료")

            self.model = model
            logger.info(
                f"YOLOX 모델 로드 완료: {self.config_path}, 가중치: {self.model_path}, 클래스 수: {checkpoint_num_classes}"
            )

            return self.model

        except Exception as e:
            logger.error(f"YOLOX 모델 로드 실패: {e}")
            raise ValueError(f"YOLOX 모델 로드 실패: {str(e)}")

    def preprocess_data(self, data: str | bytes | IO[bytes], image_size: tuple = (640, 640)):
        """
        이미지 전처리 로직 구현
        :param data: 이미지 데이터 (바이너리 또는 파일 객체)
        :param image_size: 입력 이미지 크기 튜플 (height, width) - YOLOX 형식
        :return: 전처리된 이미지 텐서, 가로비율, 세로비율, 원본 크기
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

            # 원본 이미지 크기
            orig_h, orig_w = image.shape[:2]
            original_size = (orig_w, orig_h)  # (width, height)

            # 전처리 - YOLOX의 preproc 사용
            # preproc이 직접 ratio를 계산하고 반환함
            img, ratio = preproc(image, input_size=image_size)

            # 디버깅 정보
            logger.info(f"원본 이미지: {orig_w}x{orig_h}, 모델입력: {image_size}, ratio: {ratio}")

            # 텐서로 변환
            img = torch.from_numpy(img).to(self.device).unsqueeze(0).float()

            # YOLOX는 aspect ratio를 유지하므로 같은 ratio 사용
            return img, ratio, original_size

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
            # 디바이스 설정
            self.device = torch.device("cuda" if torch.cuda.is_available() and device_str == "gpu" else "cpu")
            self.model = self.model.to(self.device)

            # 이미지 크기 가져오기 (exp에서 기본값 사용)
            test_size = getattr(self.exp, "test_size", (640, 640))

            # test_size 정규화 - (height, width) 튜플 유지
            if isinstance(test_size, (list, tuple)):
                if len(test_size) == 1:
                    image_size = (test_size[0], test_size[0])  # (H, W)
                else:
                    image_size = tuple(test_size)  # 이미 (H, W)
            else:
                # 단일 정수인 경우 정사각형으로
                image_size = (test_size, test_size)  # (H, W)

            # 데이터 전처리
            img_tensor, ratio, original_size = self.preprocess_data(data, image_size)

            # 모델 추론
            with torch.no_grad():
                prediction_result = self.model(img_tensor)

            # 후처리
            num_classes = getattr(self.exp, "num_classes", len(COCO_CLASSES))

            original_predictions = postprocess(
                prediction=prediction_result,
                num_classes=num_classes,
                conf_thre=self.conf,
                nms_thre=self.iou,
            )[0]

            # 결과 처리
            if original_predictions is None:
                return {
                    "predictions": [],
                    "image_info": {
                        "original_size": {"width": original_size[0], "height": original_size[1]},
                        "model_input_size": {"width": image_size[1], "height": image_size[0]},
                    },
                }

            output = original_predictions.cpu()
            bboxes = output[:, 0:4].clone()  # 모델 출력 좌표

            # 디버깅: 변환 전 bbox와 ratio 확인
            if len(bboxes) > 0:
                logger.info(f"변환 전 bbox 샘플 (640x640 좌표): {bboxes[0].tolist()}")
                logger.info(f"적용할 ratio: {ratio}")
                logger.info(f"원본 이미지 크기: {original_size}")

            # bbox 좌표 역변환 (640x640 → 원본 이미지 좌표계)
            # preproc이 적용한 변환의 역과정:
            # - 패딩은 좌상단 정렬이므로 무시됨
            # - ratio로 나누면 원본 좌표로 복원
            if len(bboxes) > 0:
                if ratio == 0:
                    logger.error(f"ERROR: ratio가 0입니다! 원본: {original_size}, 모델입력: {image_size}")
                    # ratio가 0이면 변환 불가능
                    return {
                        "predictions": [],
                        "image_info": {
                            "original_size": {"width": original_size[0], "height": original_size[1]},
                            "model_input_size": {"width": image_size[1], "height": image_size[0]},
                        },
                        "error": "Invalid ratio calculation",
                    }
                elif ratio < 0.001:
                    logger.warning(f"WARNING: ratio가 매우 작습니다: {ratio}")

                bboxes /= ratio

                # 변환 후 확인
                if len(bboxes) > 0:
                    logger.info(f"변환 후 bbox 샘플 (원본 좌표): {bboxes[0].tolist()}")

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            # 결과를 리스트로 변환
            predictions = []
            for bbox, score, cls_id in zip(bboxes, scores, cls):
                cls_idx = int(cls_id.item())

                x1, y1, x2, y2 = bbox.tolist()

                # 유효한 bbox인지 확인
                if x2 <= x1 or y2 <= y1:
                    continue

                # 클래스 라벨 결정
                if hasattr(self.exp, "dataset") and hasattr(self.exp.dataset, "class_names"):
                    # exp에 커스텀 클래스가 정의되어 있으면 사용
                    class_names = self.exp.dataset.class_names
                    label = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
                elif cls_idx < len(COCO_CLASSES):
                    # COCO 클래스 범위 내면 COCO_CLASSES 사용
                    label = COCO_CLASSES[cls_idx]
                else:
                    # 그 외의 경우 클래스 인덱스 표시
                    label = f"class_{cls_idx}"

                # NaN/Infinity 체크 (JSON 변환 오류 방지)
                score_value = float(score.item())
                if not all(np.isfinite(v) for v in [score_value, x1, y1, x2, y2]):
                    logger.warning(
                        f"Skipping detection with non-finite values: score={score_value}, "
                        f"bbox=[{x1}, {y1}, {x2}, {y2}]"
                    )
                    continue

                predictions.append(
                    {
                        "score": score_value,
                        "label": label,
                        "box": [float(x1), float(y1), float(x2), float(y2)],  # 명시적으로 float 변환
                    }
                )

            # 신뢰도 기준으로 정렬 및 상위 결과만 반환
            predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)[:50]  # 최대 50개

            return {
                "predictions": predictions,
                "image_info": {
                    "original_size": {"width": original_size[0], "height": original_size[1]},
                    "model_input_size": {"width": image_size[1], "height": image_size[0]},
                },
            }

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
