import os
from functools import lru_cache
from io import BytesIO
from typing import IO

import numpy as np
import onnxruntime as ort
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


class OnnxModelManager(BaseModelManager):
    def __init__(self):
        """
        ex)
        ```
        super().__init__()
        ```
        """
        # set dummy shape -> load model -> export onnx -> preprocess -> predict
        super().__init__()
        self.onnx_base = "/tmp/onnx"  # 컨테이너 내 경로, 외부 경로 사용 시 컨테이너 내 경로로 변경

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
        if category:
            model_class = categories[category]["auto_loader"]
            logger.info(f"모델 타입 '{category}' -> {model_class} 로딩.")
            model = model_class.from_pretrained(local_path)
            return model

        return AutoModel.from_pretrained(local_path)

    def export_onnx(
        self, model: torch.nn.Module, save_dir: str, model_name: str, dummy_shape: tuple[int, int, int, int]
    ):
        """
        onnx 모델 내보내기
        ort 세션에서 생성된 onnx 경로 활용하여 추론
        :param model: 모델
        :param save_dir: 저장 경로
        :param model_name: 모델 이름
        :param dummy_shape: 더미 입력 데이터 형상 (batch_size, channels, height, width)
        :return: onnx 모델 경로
        """
        if not dummy_shape:
            raise ValueError("dummy_shape 설정 필요")

        # 저장 경로 생성
        onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
        target_dir = os.path.dirname(onnx_path)
        os.makedirs(target_dir, exist_ok=True)

        # 더미 입력 데이터 생성
        dummy_input = torch.randn(dummy_shape)

        # onnx 모델 내보내기
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}},
            opset_version=16,
        )
        return onnx_path

    def load_model(self, model_name: str, run_id: str):
        """
        AutoModel 등을 활용한 load_model 로직 구현
        """
        self.model_name = model_name
        self.run_id = run_id

        local_path = self._load_artifacts(run_id, model_name)
        config = self.load_config(local_path)
        self.category = self.get_category(config)
        self.model = self._load_model(self.category, local_path)

        self.pre_processor = AutoImageProcessor.from_pretrained(local_path)

        # onnx 모델 내보내기
        # dummy_shape 설정 필요
        # 추론 시 모델 경로 활용
        try:
            self.onnx_path = self.export_onnx(
                self.model,  # 모델
                self.onnx_base,  # 저장 경로
                model_name,  # 모델 이름
                categories[self.category]["input_shape"],  # 카테고리별 input_shape
            )

            return self.model  # cannot return onnx model
        except Exception as e:
            logger.error(f"onnx 모델 내보내기 중 오류 발생: {e}")
            raise ValueError(f"onnx 모델 내보내기 실패: {str(e)}")

    def preprocess_data(self, data: str | bytes | IO[bytes]):
        """
        AutoImageProcessor 등을 활용한 전처리 로직 구현
        현재 테스트 대상 모델은 이미지 입력이기 때문에 이미지 전처리를 수행
        :param data: 이미지 데이터 (바이너리 또는 파일 객체)
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

            # 원본 이미지 크기 저장
            self.original_size = (image.width, image.height)
            logger.info(f"원본 이미지 크기: {self.original_size}")

            # 이미지 프로세서에 전달
            input_shape = categories[self.category]["input_shape"]
            height, width = input_shape[-2:]  # (batch, channel, height, width)에서 height, width 추출
            inputs = None
            size = {"height": height, "width": width}
            if "shortest_edge" in self.pre_processor.size:
                size = {"shortest_edge": min(height, width), "longest_edge": max(height, width)}
            inputs = self.pre_processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size=size,
            )
            return inputs["pixel_values"].numpy()
        except Exception as e:
            logger.error(f"이미지 전처리 중 오류 발생: {e}")
            raise ValueError(f"이미지 전처리 실패: {str(e)}")

    def _process_image_classification(self, inputs, outputs):
        """
        이미지 분류 결과 처리
        :param inputs: 입력 데이터
        :param outputs: 모델 출력
        :return: 예측 결과
        """
        logits = outputs[0]
        predicted_class_idx = np.argmax(logits, axis=-1)[0]
        label = self.model.config.id2label.get(int(predicted_class_idx), f"unknown_{int(predicted_class_idx)}")
        return int(predicted_class_idx), label

    def _convert_boxes(self, boxes, h, w):
        """
        모델 입력 크기의 박스 좌표를 원본 이미지 크기로 변환
        :param boxes: 모델 입력 크기의 박스 좌표
        :param h: 원본 이미지 높이
        :param w: 원본 이미지 너비
        :return: 원본 이미지 크기의 박스 좌표 [x, y, width, height]
        """
        # 1. config에서 bbox_format 확인
        bbox_format = getattr(self.model.config, "bbox_format", None)
        if bbox_format:
            logger.info(f"Using bbox_format from config: {bbox_format}")
            if bbox_format == "xywh":
                # 모델 입력 크기에서 원본 크기로 스케일 변환
                converted = boxes * torch.tensor([w, h, w, h])
            elif bbox_format == "xyxy":
                converted = torch.stack(
                    [boxes[..., 0] * w, boxes[..., 1] * h, boxes[..., 2] * w, boxes[..., 3] * h], dim=-1
                )
            elif bbox_format == "cxcywh":
                converted = torch.stack(
                    [boxes[..., 0] * w, boxes[..., 1] * h, boxes[..., 2] * w, boxes[..., 3] * h], dim=-1
                )
        else:
            # 2. architecture 기반 처리 (fallback)
            arch = self.model.config.architectures[0].lower()
            logger.info(f"Using bbox format based on architecture: {arch}")

            if "yolo" in arch:
                converted = boxes * torch.tensor([w, h, w, h])
            elif "detr" in arch or "fasterrcnn" in arch or "maskrcnn" in arch:
                converted = torch.stack(
                    [boxes[..., 0] * w, boxes[..., 1] * h, boxes[..., 2] * w, boxes[..., 3] * h], dim=-1
                )
            else:
                # 3. 기본값으로 xywh 사용
                logger.warning(f"Unknown bbox format for {arch}, using default xywh")
                converted = boxes * torch.tensor([w, h, w, h])

        logger.info(f"Original box coordinates: {boxes[0]}")  # 변환 전 좌표
        logger.info(f"Converted box coordinates: {converted[0]}")  # 변환 후 좌표
        return converted

    def _process_object_detection(self, inputs, outputs):
        """
        객체 검출 결과 처리
        :param inputs: 입력 데이터
        :param outputs: 모델 출력
        :return: 예측 결과
        """
        conf_threshold = 0.7  # 신뢰도 임계값

        # ONNX 출력 형태 로깅
        logger.info(f"ONNX outputs shape: {[o.shape for o in outputs]}")
        logger.info(f"Input shape: {inputs.shape}")

        # 모델 타입에 따라 다르게 처리
        if len(outputs) == 1:
            # YOLO 등 통합된 출력을 사용하는 모델
            logits = outputs[0]
            logger.info(f"Logits shape: {logits.shape}")

            # YOLO 출력 형식: [batch, num_boxes, num_classes + 5]
            # 마지막 차원: [x, y, w, h, conf, class_scores...]
            # scores = logits[0, :, 4]  # confidence scores
            boxes = logits[0, :, :4]  # bounding boxes
            # labels = logits[0, :, 5:]  # class scores

            # 클래스 점수에 소프트맥스 적용
            # 각 박스의 클래스 점수에 대해 소프트맥스 적용
            num_classes = logits.shape[-1] - 4  # 마지막 차원에서 4를 뺀 값이 클래스 수
            logger.info(f"YOLO style output detected. Number of classes: {num_classes}")
            class_probs = logits[..., :num_classes]
            boxes = logits[..., -4:]  # 마지막 4개 값이 box coordinates
            probs = torch.nn.functional.softmax(torch.from_numpy(class_probs), dim=-1).numpy()

            # confidence score는 클래스 확률의 최대값
            scores = np.max(probs, axis=-1)

            logger.info(f"Scores shape: {scores.shape}, sample: {scores[:5]}")
            logger.info(f"Boxes shape: {boxes.shape}, sample: {boxes[:5]}")
            logger.info(f"Class probs shape: {class_probs.shape}, sample: {class_probs[:5]}")

            # 신뢰도 임계값 적용
            mask = scores > conf_threshold
            scores = scores[mask]
            boxes = boxes[mask]
            class_probs = class_probs[mask]

            logger.info(
                f"After threshold - Scores: {scores.shape}, Boxes: {boxes.shape}, Class probs: {class_probs.shape}"
            )

            if len(scores) == 0:
                logger.warning("No detections above confidence threshold")
                return []

            # 클래스 예측
            class_ids = np.argmax(class_probs, axis=1)
            logger.info(f"Class IDs: {class_ids}")
        else:
            # DETR 등 분리된 출력을 사용하는 모델
            logits = outputs[0]
            boxes = outputs[1]

            # softmax로 확률 계산
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            scores = np.max(probs, axis=-1)
            class_ids = np.argmax(probs, axis=-1)

            # 신뢰도 임계값 적용
            mask = scores > conf_threshold
            scores = scores[mask]
            boxes = boxes[mask]
            class_ids = class_ids[mask]

            if len(scores) == 0:
                return []

        # 원본 이미지 크기로 스케일링
        w, h = self.original_size
        model_h, model_w = categories[self.category]["input_shape"][-2:]
        scale_w = w / model_w
        scale_h = h / model_h

        # 박스 좌표 스케일링
        boxes = boxes * np.array([scale_w, scale_h, scale_w, scale_h])

        # 결과 변환
        results = [
            {
                "score": float(score),
                "label": self.model.config.id2label.get(int(label), f"unknown_{int(label)}"),
                "box": [float(x) for x in box],
            }
            for score, label, box in zip(scores, class_ids, boxes)
        ]

        # 상위 5개 결과만 반환
        processed_results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]
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
        # 예측 결과와 확률값 계산
        logits = outputs[0]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # softmax
        pred = np.argmax(logits, axis=1)[0]
        conf = np.max(probs, axis=1)[0]

        # 원본 이미지 크기로 업샘플링
        h, w = inputs.shape[-2:]
        # PIL이 지원하는 uint8로 변환하고 올바른 shape으로 변환
        pred = pred.astype(np.uint8)
        pred = np.squeeze(pred)  # (1, 1) -> (h, w)
        pred = np.array(Image.fromarray(pred).resize((w, h), Image.NEAREST))

        conf = conf.astype(np.float32)
        conf = np.squeeze(conf)  # (1, 1) -> (h, w)
        conf = np.array(Image.fromarray(conf).resize((w, h), Image.BILINEAR))

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

    def predict(self, data: str | bytes | IO[bytes], device_str: str):
        """
        AutoModel 등을 활용한 predict 로직 구현
        :param data: 입력 데이터
        :param device_str: 추론 장치 문자열 ("cpu" 또는 "cuda")
        :return: 예측 결과
        """
        ort_session = None
        try:
            if not self.onnx_path or not self.pre_processor or not self.category:
                logger.error(f"ModelManager not ready: \n  model={self.model},\n  processor={self.pre_processor}")
                raise ValueError("모델 추론이 준비되지 않았습니다")

            logger.info(
                f"start inference\n"
                f"  device={device_str}\n"
                f"  model={type(self.model)}\n"
                f"  processor={type(self.pre_processor)}\n"
                f"  data_type={type(data)}"
            )

            # 데이터 전처리
            input_tensor = self.preprocess_data(data)

            # 모델 추론 장치 선택
            providers = (
                ["CUDAExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers() and device_str == "gpu"
                else ["CPUExecutionProvider"]
            )

            # ONNX 런타임 세션 생성
            ort_session = ort.InferenceSession(self.onnx_path, providers=providers)
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name

            # 추론 실행
            outputs = ort_session.run([output_name], {input_name: input_tensor})

            if self.category not in self.post_processors:
                raise ValueError(f"지원하지 않는 모델 카테고리입니다: {self.category}")

            return self.post_processors[self.category](input_tensor, outputs)
        except Exception as e:
            logger.error(f"추론 중 오류 발생: {e}")
            raise e
        finally:
            if ort_session:
                ort_session.set_providers([])
                ort_session = None
            self._clear_model()

    def _clear_model(self):
        """
        메모리 정리
        """
        self.onnx_path = None
        self.pre_processor = None
        self.category = None
        self.model = None
        self.model_name = None
        self.run_id = None


# 모델 인스턴스 캐싱
@lru_cache
def get_model_manager():
    model_manager = OnnxModelManager()
    return model_manager


model_manager = get_model_manager()
