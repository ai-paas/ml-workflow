import gc
from functools import lru_cache
from io import BytesIO
from typing import IO

import numpy as np
import torch
from logging_inference import logger
from model_manager.base import BaseModelManager
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForSemanticSegmentation,
    DetrForSegmentation,
)

FILE_INPUT_TYPE = "이미지"
ACCELERATOR = "cpu, gpu"

categories = {
    "ImageClassification": {"input_shape": (1, 3, 224, 224), "auto_loader": AutoModelForImageClassification},
    "ObjectDetection": {
        "input_shape": (1, 3, 640, 640),
        "auto_loader": AutoModelForObjectDetection,
    },
    "SemanticSegmentation": {
        "input_shape": (1, 3, 224, 224),
        "auto_loader": AutoModelForSemanticSegmentation,
    },
    "DetrForSegmentation": {
        "input_shape": (1, 3, 224, 224),
        "auto_loader": DetrForSegmentation,
    },
}


class PytorchModelManager(BaseModelManager):
    def __init__(self):
        """
        ex)
        ```
        super().__init__()
        ```
        """
        super().__init__()

        # 카테고리별 처리 로직 매핑
        self.post_processors = {
            "ImageClassification": self._process_image_classification,
            "ObjectDetection": self._process_object_detection,
            "SemanticSegmentation": self._process_semantic_segmentation,
            "DetrForSegmentation": self._process_semantic_segmentation,
        }

    def load_config(self, local_path):
        """
        Load the configuration of a Hugging Face model.
        :param local_path: Local path to the model.
        :return: Model configuration.
        """
        config = AutoConfig.from_pretrained(local_path)
        return config

    def get_category(self, config):
        """
        Get the category of the model based on its configuration.
        :param config: Model configuration.
        :return: Category of the model. if not found, return None.
        """
        if config.architectures:
            for category in categories.keys():
                # task 이름으로 끝나는 모델 타입 확인
                if any(arch for arch in config.architectures if arch.endswith(category)):
                    return category

        return None

    # 모델 실제 호출 : huggingface
    def _load_model(self, category, local_path):
        """
        Load a Hugging Face model based on its configuration.
        :param config: Model configuration.
        :param local_path: Local path to the model.
        :return: Loaded model.
        """
        # config 내 architectures 확인
        try:
            model_class = categories[category]["auto_loader"]
            logger.info(f"모델 타입 '{category}' -> {model_class} 로딩.")
            return model_class.from_pretrained(local_path)
        except Exception as e:
            logger.warning(f"모델 타입 '{category}' 로딩 실패 -> AutoModel 로딩: {e}")
            return AutoModel.from_pretrained(local_path)

    def load_model(self, model_name: str, run_id: str):
        """
        AutoModel 등을 활용한 load_model 로직 구현
        """

        self.model_name = model_name
        self.run_id = run_id

        local_path = self._load_artifacts(run_id, model_name)
        # 추가 필요
        config = self.load_config(local_path)
        self.category = self.get_category(config)
        self.model = self._load_model(self.category, local_path)

        self.pre_processor = AutoImageProcessor.from_pretrained(local_path)

        return self.model

    def preprocess_data(self, data: str | bytes | IO[bytes], device: torch.device):
        """
        AutoImageProcessor 등을 활용한 전처리 로직 구현
        현재 테스트 대상 모델은 이미지 입력이기 때문에 이미지 전처리를 수행
        :param data: 이미지 데이터 (바이너리 또는 파일 객체)
        :param device: 추론 장치
        :return: 전처리된 데이터
        """
        try:
            if isinstance(data, bytes):
                image = Image.open(BytesIO(data))
            else:
                image = Image.open(data)

            # RGB로 변환 (RGBA나 다른 형식일 경우)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 이미지 프로세서에 전달
            inputs = self.pre_processor(images=image, return_tensors="pt")
            return {k: v.to(device) for k, v in inputs.items()}
        except Exception as e:
            logger.error(f"이미지 전처리 중 오류 발생: {e}")
            raise ValueError(f"이미지 전처리 실패: {str(e)}")

    def predict(self, data: str | bytes | IO[bytes], device_str: str):
        """
        AutoModel 등을 활용한 predict 로직 구현
        :param data: 입력 데이터
        :param device_str: 추론 장치 문자열 ("cpu" 또는 "gpu")
        :return: 예측 결과
        """
        try:
            if not self.model or not self.pre_processor or not self.category:
                logger.error(f"ModelManager not ready: \n  model={self.model},\n  processor={self.pre_processor}")
                raise ValueError("모델 추론이 준비되지 않았습니다")

            logger.info(
                f"start inference\n"
                f"  device={device_str}\n"
                f"  model={type(self.model)}\n"
                f"  processor={type(self.pre_processor)}\n"
                f"  data_type={type(data)}"
            )

            # 모델 추론 장치 선택
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() and device_str == "gpu" else "cpu")
            model = self.model.to(device)
            # 데이터 전처리
            inputs = self.preprocess_data(data, device)

            with torch.no_grad():
                # 모델 추론
                outputs = model(**inputs)

                if self.category not in self.post_processors:
                    raise ValueError(f"지원하지 않는 모델 카테고리입니다: {self.category}")

                return self.post_processors[self.category](inputs, outputs)
        except Exception as e:
            logger.error(f"추론 중 오류 발생: {e}")
            raise e
        finally:
            torch.cuda.empty_cache()
            self._clear_model()

    def _process_image_classification(self, inputs, outputs):  # 입력 형태를 통일하기 위해 input 파라미터 유지
        """
        이미지 분류 결과 처리
        :param inputs: 입력 데이터
        :param outputs: 모델 출력
        :return: 예측 결과
        """
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        label = self.model.config.id2label.get(int(predicted_class_idx), f"unknown_{int(predicted_class_idx)}")
        return predicted_class_idx, label

    def _process_object_detection(self, inputs, outputs):
        """
        객체 검출 결과 처리
        :param inputs: 입력 데이터
        :param outputs: 모델 출력
        :return: 예측 결과
        """
        conf_threshold = 0.7  # 신뢰도 임계값
        target_sizes = torch.tensor([inputs["pixel_values"].shape[-2:]])
        results = self.pre_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]

        processed_results = [
            {
                "score": score.item(),
                "label": self.model.config.id2label.get(int(label.item()), f"unknown_{int(label.item())}"),
                "box": box.tolist(),
            }
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
        ]

        # 상위 5개 결과만 반환
        processed_results = sorted(processed_results, key=lambda x: x["score"], reverse=True)[:5]
        logger.info(f"ObjectDetection result: {processed_results}")
        return processed_results

    def _get_segment_info(self, pred: np.ndarray, conf: np.ndarray, class_id: int, conf_threshold: float = 0.7) -> dict:
        """
        클래스별 세그먼트 정보를 계산
        :param pred: 예측 결과 배열
        :param conf: 신뢰도 배열
        :param class_id: 클래스 ID
        :param conf_threshold: 신뢰도 임계값 (기본값: 0.7)
        :return: 세그먼트 정보. 유효한 세그먼트가 없는 경우 None을 반환합니다.
            반환되는 딕셔너리는 다음 키를 포함합니다:
            - class_id (int): 클래스 ID
            - class_name (str): 클래스 이름
            - confidence (float): 평균 신뢰도
            - pixels (int): 유효한 픽셀 수
            - center (list): 중심점 좌표 [x, y]
        """
        mask = pred == class_id
        pixels = np.sum(mask)

        if pixels == 0:
            return None

        # 신뢰도가 임계값 이상인 픽셀만 사용
        conf_mask = conf >= conf_threshold
        valid_mask = mask & conf_mask
        valid_pixels = np.sum(valid_mask)

        if valid_pixels == 0:
            return None

        y_indices, x_indices = np.where(valid_mask)
        return {
            "class_id": int(class_id),
            "class_name": self.model.config.id2label.get(int(class_id), f"unknown_{int(class_id)}"),
            "confidence": float(np.mean(conf[valid_mask])),
            "pixels": int(valid_pixels),
            "center": [int(np.mean(x_indices)), int(np.mean(y_indices))],
        }

    def _process_semantic_segmentation(self, inputs, outputs):
        """
        시맨틱 세그멘테이션 결과 처리
        :param inputs: 모델 입력 데이터
        :param outputs: 모델 출력
        :return: 세그멘테이션 결과 리스트
        """
        # 모델 출력을 원본 이미지 크기로 업샘플링
        upsampled = torch.nn.functional.interpolate(
            outputs.logits,
            size=inputs["pixel_values"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # 예측 결과와 확률값 계산
        probs = torch.nn.functional.softmax(upsampled, dim=1)
        pred = upsampled.argmax(dim=1)[0].cpu().numpy()
        conf = probs.max(dim=1)[0][0].cpu().numpy()

        conf_threshold = 0.7  # 신뢰도 임계값
        # 배경을 제외한 각 클래스별 정보 계산
        segments = [
            segment
            for segment in (
                self._get_segment_info(pred, conf, int(class_id), conf_threshold)
                for class_id in np.unique(pred)
                if class_id != 0  # 배경 제외
            )
            if segment is not None
        ]

        # 신뢰도 기준으로 정렬
        segments.sort(key=lambda x: x["confidence"], reverse=True)
        segments = segments[: 5 if len(segments) > 5 else len(segments)]

        return segments

    def _clear_model(self):
        """
        메모리 정리
        """
        if self.model is not None:
            del self.model
        self.model = None
        self.pre_processor = None
        self.category = None
        self.model_name = None
        self.run_id = None
        gc.collect()


@lru_cache
def get_model_manager():
    model_manager = PytorchModelManager()
    return model_manager


model_manager = get_model_manager()
