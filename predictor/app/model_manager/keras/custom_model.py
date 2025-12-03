import gc
from functools import lru_cache
from io import BytesIO
from typing import IO

import tensorflow as tf
from app.logging_inference import logger
from app.model_manager.base import BaseModelManager
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    TFAutoModelForImageClassification,
    TFAutoModelForSemanticSegmentation,
)

FILE_INPUT_TYPE = "이미지/테이블"
ACCELERATOR = "cpu, gpu"

categories = {
    "ImageClassification": {"input_shape": (1, 3, 224, 224), "auto_loader": TFAutoModelForImageClassification},
    "SemanticSegmentation": {
        "input_shape": (1, 3, 224, 224),
        "auto_loader": TFAutoModelForSemanticSegmentation,
    },
}


class KerasModelManager(BaseModelManager):
    def __init__(self):
        super().__init__()

        # 카테고리별 후처리 로직 매핑
        self.post_processors = {
            "ImageClassification": self._process_image_classification,
            "SemanticSegmentation": self._process_semantic_segmentation,
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

    def _load_model(self, category, local_path):
        """
        Load a Hugging Face model based on its configuration.
        :param config: Model configuration.
        :param local_path: Local path to the model.
        :return: Loaded model.
        """
        # config 내 architectures 확인
        try:
            if category:
                model_class = categories[category]["auto_loader"]
                logger.info(f"모델 타입 '{category}' -> {model_class} 로딩.")
                model = model_class.from_pretrained(local_path)
                return model
        except Exception as e:
            logger.warning(f"모델 로딩 중 오류 발생: {e}")
            return AutoModel.from_pretrained(local_path)

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
        if not self.model:
            raise ValueError(f"지원하지 않는 모델 입니다: {model_name}")
        self.pre_processor = AutoImageProcessor.from_pretrained(local_path)

        return self.model

    def preprocess_data(self, data: str | bytes | IO[bytes]):
        """
        AutoImageProcessor 등을 활용한 전처리 로직 구현
        현재 테스트 대상 모델은 이미지 입력이기 때문에 이미지 전처리를 수행
        :param data: 이미지 데이터 (바이너리 또는 파일 객체)
        :return: 전처리된 데이터
        """
        try:
            # 이미지 데이터 로드
            if isinstance(data, bytes):
                image = Image.open(BytesIO(data))
            else:
                image = Image.open(data)

            # RGB로 변환 (RGBA나 다른 형식일 경우)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 이미지 프로세서에 전달
            return self.pre_processor(images=image, return_tensors="tf")
        except Exception as e:
            logger.error(f"이미지 전처리 중 오류 발생: {e}")
            raise ValueError(f"이미지 전처리 실패: {str(e)}")

    def _process_image_classification(self, outputs):
        """
        이미지 분류 결과 처리
        :param outputs: 모델 출력
        :return: (클래스 인덱스, 클래스 레이블)
        """
        logits = outputs.logits
        predicted_class_idx = int(tf.argmax(logits, axis=-1).numpy()[0])
        label = self.model.config.id2label.get(predicted_class_idx, f"unknown_{int(predicted_class_idx)}")
        logger.info(f"end inference\n  predicted_class_idx={predicted_class_idx}\n  label={label}")
        return predicted_class_idx, str(label)

    def _process_object_detection(self, outputs):
        """
        객체 검출 결과 처리
        :param outputs: 모델 출력
        :return: 처리된 객체 검출 결과 리스트
        """
        conf_threshold = 0.7
        target_sizes = tf.constant([outputs.logits.shape[-2:]])
        results = self.pre_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]

        processed_results = [
            {
                "score": float(score),
                "label": self.model.config.id2label.get(int(label), f"unknown_{int(label)}"),
                "box": box.numpy().tolist(),
            }
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
        ]

        # 상위 5개 결과만 반환
        processed_results = sorted(processed_results, key=lambda x: x["score"], reverse=True)[:5]
        logger.info(f"ObjectDetection result: {processed_results}")
        return processed_results

    def _get_segment_info(self, pred: tf.Tensor, conf: tf.Tensor, class_id: int, conf_threshold: float = 0.9) -> dict:
        """
        클래스별 세그먼트 정보를 계산
        :param pred: 예측 결과 텐서
        :param conf: 신뢰도 텐서
        :param class_id: 클래스 ID
        :param conf_threshold: 신뢰도 임계값
        :return: 세그먼트 정보 딕셔너리
        """
        mask = pred == class_id
        pixels = tf.reduce_sum(tf.cast(mask, tf.int32))

        if pixels == 0:
            return None

        # 신뢰도가 임계값 이상인 픽셀만 사용
        conf_mask = conf >= conf_threshold
        valid_mask = mask & conf_mask
        valid_pixels = tf.reduce_sum(tf.cast(valid_mask, tf.int32))

        if valid_pixels == 0:
            return None

        # tf.where의 반환값을 올바르게 처리
        indices = tf.where(valid_mask)
        if len(indices.shape) == 2 and indices.shape[1] == 2:
            y_indices = indices[:, 0]
            x_indices = indices[:, 1]
        else:
            # 차원이 맞지 않는 경우 빈 결과 반환
            return None

        return {
            "class_id": int(class_id),
            "class_name": self.model.config.id2label.get(int(class_id), f"unknown_{int(class_id)}"),
            "confidence": float(tf.reduce_mean(tf.boolean_mask(conf, valid_mask))),
            "pixels": int(valid_pixels),
            "center": [int(tf.reduce_mean(x_indices)), int(tf.reduce_mean(y_indices))],
        }

    def _process_semantic_segmentation(self, outputs):
        """
        시맨틱 세그멘테이션 결과 처리
        :param outputs: 모델 출력
        :return: 처리된 세그멘테이션 결과 딕셔너리
        """
        # 모델 출력을 원본 이미지 크기로 업샘플링
        upsampled = tf.image.resize(outputs.logits, outputs.logits.shape[-2:], method=tf.image.ResizeMethod.BILINEAR)
        probs = tf.nn.softmax(upsampled, axis=1)
        pred = tf.argmax(upsampled, axis=1)[0]
        conf = tf.reduce_max(probs, axis=1)[0]

        # pred를 1차원으로 변환
        pred_flat = tf.reshape(pred, [-1])

        # 결과 통계 로깅
        logger.info(
            f"Segmentation result: shape={pred.shape}, "
            f"classes={len(tf.unique(pred_flat)[0])}, "
            f"range=[{tf.reduce_min(pred)}, {tf.reduce_max(pred)}], "
        )

        conf_threshold = 0.7  # 신뢰도 임계값

        # 배경을 제외한 각 클래스별 정보 계산
        segments = [
            segment
            for segment in (
                self._get_segment_info(pred, conf, class_id, conf_threshold)
                for class_id in tf.unique(pred_flat)[0]
                if class_id != 0  # 배경 제외
            )
            if segment is not None
        ]

        # 신뢰도 기준으로 정렬
        segments.sort(key=lambda x: x["confidence"], reverse=True)
        segments = segments[: 5 if len(segments) > 5 else len(segments)]

        return {"image_size": list(pred.shape), "segments": segments}

    def predict(self, data: str | bytes | IO[bytes], device_str: str):
        """
        모델 추론 및 결과 처리
        :param data: 입력 데이터
        :param device_str: 추론 장치 문자열 ("cpu" 또는 "gpu")
        :return: 예측 결과
        """
        try:
            if not self.model or not self.pre_processor:
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
            gpu_devices = tf.config.list_physical_devices("GPU")
            device = "/GPU:0" if gpu_devices and device_str == "gpu" else "/CPU:0"
            # 데이터 전처리
            inputs = self.preprocess_data(data)

            with tf.device(device):
                # 모델 추론
                outputs = self.model(**inputs)

                # 결과 처리
                if self.category not in self.post_processors:
                    raise ValueError(f"지원하지 않는 모델 카테고리입니다: {self.category}")

                return self.post_processors.get(self.category, self._process_object_detection)(outputs)
        except Exception as e:
            logger.error(f"추론 중 오류 발생: {e}")
            raise e
        finally:
            tf.keras.backend.clear_session()
            self._clear_model()

    def _clear_model(self):
        """
        메모리 정리
        """
        self.model = None
        self.pre_processor = None
        self.category = None
        self.model_name = None
        self.run_id = None
        gc.collect()


@lru_cache
def get_model_manager():
    model_manager = KerasModelManager()
    return model_manager


model_manager = get_model_manager()
