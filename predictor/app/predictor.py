import argparse
import base64
import json
import os
from io import BytesIO
from typing import Dict, Union

import kserve
import mlflow
import numpy as np
import torch

# Model Manager Factory 및 관련 import
from app.model_manager.base import BaseModelManager
from app.model_manager.keras.custom_model import KerasModelManager
from app.model_manager.onnx.custom_model import OnnxModelManager
from app.model_manager.pytorch.custom_model import PytorchModelManager
from app.model_manager.yolox.custom_model import YoloxModelManager
from kserve import InferInput, InferOutput, InferResponse, Model, ModelServer, logging
from kserve.model import PredictorConfig
from kserve.utils.utils import generate_uuid
from PIL import Image, ImageDraw
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


class ModelManagerFactory:
    """
    프레임워크별 모델 매니저를 생성하는 Factory 클래스
    """

    _managers = {
        "pytorch": PytorchModelManager,
        "keras": KerasModelManager,
        "onnx": OnnxModelManager,
        "yolox": YoloxModelManager,
    }

    @classmethod
    def create_model_manager(cls, framework: str) -> BaseModelManager:
        """
        프레임워크 타입에 따라 적절한 모델 매니저를 생성

        Args:
            framework: 프레임워크 타입 ("pytorch", "keras", "onnx")

        Returns:
            BaseModelManager: 생성된 모델 매니저 인스턴스

        Raises:
            ValueError: 지원하지 않는 프레임워크인 경우
        """
        if framework not in cls._managers:
            supported_frameworks = ", ".join(cls._managers.keys())
            raise ValueError(
                f"지원하지 않는 프레임워크입니다: {framework}. \
                지원되는 프레임워크: {supported_frameworks}"
            )

        manager_class = cls._managers[framework]
        return manager_class()

    @classmethod
    def get_supported_frameworks(cls) -> list:
        """지원되는 프레임워크 목록 반환"""
        return list(cls._managers.keys())


def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[
        :, None, None
    ]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image


class InferenceModel(Model):
    def __init__(
        self,
        name: str,
        model_uri: str,
        mlflow_tracking_uri: str,
        mlflow_s3_endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        mlflow_experiment_name: str,
        predictor_host: str,
        predictor_protocol: str,
        predictor_use_ssl: bool,
        framework: str = "pytorch",  # 기본값으로 pytorch 설정
        run_id: str = None,
    ):
        super().__init__(name, PredictorConfig(predictor_host, predictor_protocol, predictor_use_ssl))
        self.name = name
        self.model_uri = model_uri
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.framework = framework
        self.run_id = run_id

        logging.logger.info(
            f"""model_uri = {model_uri},
                mlflow_tracking_uri = {mlflow_tracking_uri},
                mlflow_experiment_name = {mlflow_experiment_name},
                framework = {framework},
                model_name = {name},
                run_id = {run_id}
                """
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.logger.info(f"torch device={self.device}")

        # MLflow 환경 설정
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

        # Model Manager Factory를 통한 모델 매니저 생성
        self.model_manager = ModelManagerFactory.create_model_manager(framework)

        self.load()

    def load(self):
        """모델 로드 및 초기화"""
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name=self.mlflow_experiment_name)

            # Model Manager를 통한 모델 로드
            if self.run_id:
                self.model_manager.load_model(self.model_uri, self.run_id)
                logging.logger.info(f"Model loaded successfully using {self.framework} framework")
            else:
                # 기존 MLflow transformers 로드 방식 (하위 호환성)
                model_artifacts = mlflow.transformers.load_model(self.model_uri)
                self.model = model_artifacts.model.to(self.device)
                self.processor = model_artifacts.image_processor
                self.tokenizer = model_artifacts.tokenizer
                logging.logger.info("Model loaded using legacy MLflow transformers method")

            self.ready = True

        except Exception as e:
            logging.logger.error(f"Model loading failed: {e}")
            raise e

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        """모델 추론 실행"""
        try:
            input_bytes = payload.inputs[0].data
            data = input_bytes[0]

            # 이미지 데이터 추출
            image_bytes = base64.b64decode(data["image"])

            # 텍스트 리스트 추출
            texts = data.get("text", [])

            logging.logger.info(f"input texts = {texts}")

            # Model Manager를 통한 추론 (새로운 방식)
            if hasattr(self, "model_manager") and self.model_manager:
                device_str = "gpu" if torch.cuda.is_available() else "cpu"
                result = self.model_manager.predict(data=image_bytes, device_str=device_str)

                # result는 dict 형태: {"predictions": [...], "image_info": {...}}
                if isinstance(result, dict):
                    predictions = result.get("predictions", result)

                    # 예측 결과를 JSON 직렬화 가능한 형태로 변환
                    if hasattr(predictions, "tolist"):
                        predictions = predictions.tolist()
                    elif isinstance(predictions, (np.ndarray, torch.Tensor)):
                        predictions = predictions.tolist()

                    # image_info와 함께 결과 반환
                    response_data = {"predictions": predictions, "image_info": result.get("image_info", {})}
                else:
                    # 하위 호환성: result가 dict가 아닌 경우
                    if hasattr(result, "tolist"):
                        predictions = result.tolist()
                    elif isinstance(result, (np.ndarray, torch.Tensor)):
                        predictions = result.tolist()
                    else:
                        predictions = result

                    response_data = {"predictions": predictions}

                return InferResponse(
                    response_id=generate_uuid(),
                    model_name=self.name,
                    infer_outputs=[
                        InferOutput(name="OUTPUT_0", datatype="BYTES", shape=[1], data=[json.dumps(response_data)])
                    ],
                )

            # 기존 방식 (하위 호환성)
            else:
                return self._legacy_predict(data, image_bytes, texts)

        except Exception as e:
            logging.logger.error(f"Prediction failed: {e}")
            raise e

    def _legacy_predict(self, data, image_bytes, texts):
        """기존 추론 방식 (하위 호환성)"""
        image = Image.open(BytesIO(image_bytes))

        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        logging.logger.info(f"processed input = {inputs}")

        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))

        unnormalized_image = get_preprocessed_image(inputs.pixel_values.cpu())

        # Convert outputs to COCO API
        target_sizes = torch.tensor([unnormalized_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.2
        )

        i = 0
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            logging.logger.info(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        # 이미지 시각화
        visualized_image = unnormalized_image.copy()
        draw = ImageDraw.Draw(visualized_image)

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = tuple(box)
            draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
            draw.text(xy=(x1, y1), text=text[label])

        # PIL Image를 바이트로 변환
        img_byte_arr = BytesIO()
        visualized_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

        return InferResponse(
            response_id=generate_uuid(),
            model_name=self.name,
            infer_outputs=[InferOutput(name="OUTPUT_0", datatype="BYTES", shape=[1], data=[img_base64])],
        )


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])

# 기존 인자들
parser.add_argument("--model_uri", type=str, required=True, help="URI of the MLflow model")
parser.add_argument("--mlflow_tracking_uri", type=str, required=True, help="MLflow tracking server URI")
parser.add_argument("--mlflow_experiment_name", type=str, required=True, help="MLflow experiment name")
parser.add_argument("--mlflow_s3_endpoint_url", type=str, required=True, help="MLflow s3 endpoint url")
parser.add_argument("--aws_access_key_id", type=str, required=True, help="MLflow s3 username")
parser.add_argument("--aws_secret_access_key", type=str, required=True, help="MLflow s3 password")

# 새로운 인자들 (Model Manager Factory용)
parser.add_argument(
    "--framework",
    type=str,
    default="pytorch",
    choices=ModelManagerFactory.get_supported_frameworks(),
    help="Framework type for model inference",
)
parser.add_argument("--run_id", type=str, help="MLflow run ID")

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)

    model = InferenceModel(
        args.model_name,
        predictor_host=args.predictor_host,
        predictor_protocol=args.predictor_protocol,
        predictor_use_ssl=args.predictor_use_ssl,
        model_uri=args.model_uri,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_s3_endpoint_url=args.mlflow_s3_endpoint_url,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        framework=args.framework,
        run_id=args.run_id,
    )

    ModelServer().start([model])
