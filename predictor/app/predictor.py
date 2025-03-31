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
from kserve import InferInput, InferOutput, InferResponse, Model, ModelServer, logging
from kserve.model import PredictorConfig
from kserve.utils.utils import generate_uuid
from PIL import Image, ImageDraw
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


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
    ):
        super().__init__(name, PredictorConfig(predictor_host, predictor_protocol, predictor_use_ssl))
        self.name = name
        self.model_uri = model_uri
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        logging.logger.info(
            f"""model_uri =
                            {model_uri},
                            mlflow_tracking_uri =
                            {mlflow_tracking_uri},
                            mlflow_experiment_name =
                            {mlflow_experiment_name}
                            """
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.logger.info(f"torch device={self.device}")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        self.load()

    def load(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=self.mlflow_experiment_name)
        model_artifacts = mlflow.transformers.load_model(self.model_uri)
        self.model = model_artifacts.model.to(self.device)
        self.processor = model_artifacts.image_processor
        self.tokenizer = model_artifacts.tokenizer

        # logging.logger.info(f"model = {self.model}")
        # logging.logger.info(f"processor = {self.processor}")
        # logging.logger.info(f"tokenizer = {self.tokenizer}")
        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        input_bytes = payload.inputs[0].data

        # bytes를 JSON으로 디코딩
        data = input_bytes[0]

        # 이미지 데이터 추출
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(BytesIO(image_bytes))

        # 텍스트 리스트 추출 - ['a cat', 'remote control'] 형태
        texts = data["text"]

        logging.logger.info(f"input texts = {texts}")

        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        logging.logger.info(f"processed input = {inputs}")
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))

        unnormalized_image = get_preprocessed_image(inputs.pixel_values.cpu())

        # Convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([unnormalized_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.2
        )
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            logging.logger.info(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        # TODO: 이미지 그려서 내보낼지, 값을 내보내지 client측에서 이미지 그릴지 결정 필요.
        visualized_image = unnormalized_image.copy()

        draw = ImageDraw.Draw(visualized_image)

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = tuple(box)
            draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
            draw.text(xy=(x1, y1), text=text[label])

        # PIL Image를 바이트로 변환
        img_byte_arr = BytesIO()
        visualized_image.save(img_byte_arr, format="PNG")  # 또는 'JPEG'
        img_byte_arr = img_byte_arr.getvalue()
        # Base64로 인코딩
        img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

        return InferResponse(
            response_id=generate_uuid(),
            model_name=self.name,
            infer_outputs=[InferOutput(name="OUTPUT_0", datatype="BYTES", shape=[1], data=[img_base64])],
        )

    # TODO:
    # def postprocess(
    #         self, infer_response: Union[Dict, InferResponse],
    #         headers: Dict[str, str] = None
    # ) -> Union[Dict, InferResponse]:

    #     # TODO: mocking data. 실 데이터 처리 및 응답으로 변경필요
    #     results = [1, 2, 3]
    #     return InferResponse(
    #         model_name=self.name,
    #         infer_outputs=[
    #             InferOutput(name="OUTPUT_0", datatype="INT64", shape=[len(results)], data=results)
    #         ],
    #         response_id=infer_response.id
    #     )


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])

# 필요한 인자들을 추가로 정의
parser.add_argument("--model_uri", type=str, required=True, help="URI of the MLflow model")
parser.add_argument("--mlflow_tracking_uri", type=str, required=True, help="MLflow tracking server URI")
parser.add_argument("--mlflow_experiment_name", type=str, required=True, help="MLflow experiment name")
parser.add_argument("--mlflow_s3_endpoint_url", type=str, required=True, help="MLflow s3 endpoint url")
parser.add_argument("--aws_access_key_id", type=str, required=True, help="MLflow s3 username")
parser.add_argument("--aws_secret_access_key", type=str, required=True, help="MLflow s3 password")

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)  # Configure kserve and uvicorn logger
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
    )

    ModelServer().start([model])
