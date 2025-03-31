import argparse
import ast
import io
import json
import logging
import os
import traceback
import uuid
from pathlib import Path

import albumentations
import mlflow
import numpy as np
import requests
import torch
from datasets import Dataset, load_dataset
from mlflow import MlflowClient
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    CLIPTokenizer,
    Owlv2ForObjectDetection,
    Owlv2ImageProcessor,
    Owlv2Processor,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

# 현재 파일의 절대 경로 얻기
current_path = Path(__file__).absolute().parent


class TrainModel:
    def __init__(
        self,
        train_name: str,
        model_name: str,
        model_uri: str,
        mlflow_tracking_uri: str,
        mlflow_s3_endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        mlflow_experiment_name: str,
        dataset_artifact_uri: str,
        restapi_url: str,
        restapi_username: str,
        restapi_password: str,
    ):
        self.train_name = train_name
        self.model_name = model_name
        self.model_uri = model_uri
        self.dataset_artifact_uri = dataset_artifact_uri
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.restapi_url = restapi_url
        self.restapi_username = restapi_username
        self.restapi_password = restapi_password
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

        self.load()

    def load(self):
        self.client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=self.mlflow_experiment_name)
        self.model_artifacts = mlflow.transformers.load_model(self.model_uri)
        self.dataset_artifacts = mlflow.artifacts.download_artifacts(artifact_uri=self.dataset_artifact_uri)
        self.transform = albumentations.Compose(
            [
                albumentations.Resize(480, 480),
                albumentations.HorizontalFlip(p=1.0),
                albumentations.RandomBrightnessContrast(p=1.0),
            ],
            bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
        )
        self.image_processor = self.model_artifacts.image_processor.image_processor
        self.model = self.model_artifacts.model

        # # mocking data
        self.id2label = {0: "head", 1: "helmet", 2: "person"}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # login to restapi
        self.restapi_token = get_token_from_restapi(
            url=self.restapi_url, username=self.restapi_username, password=self.restapi_password
        )

        # self.model.config.id2label = self.id2label
        # self.model.config.label2id = self.label2id

    # TODO: 코드 동작하지 않음. 추후 개발 필요.
    def preprocess(self):
        # 이미지 변환 처리
        self.train_dataset = convert_dataset_images(Dataset.from_csv(f"{self.dataset_artifacts}/train.csv"))
        self.test_dataset = convert_dataset_images(Dataset.from_csv(f"{self.dataset_artifacts}/test.csv"))
        # print(f"train sample = {self.train_dataset[0]}")
        # print(f"test sample = {self.test_dataset[0]}")
        self.train_dataset_transformed = self.train_dataset.with_transform(self.transform_aug_ann)
        self.test_dataset_transformed = self.test_dataset.with_transform(self.transform_aug_ann)
        print(f"transformed train  ={self.train_dataset_transformed}")
        print(f"transformed test = {self.test_dataset_transformed}")

    # TODO: 코드 동작하지 않음. 추후 개발 필요.
    def train(self):
        # TODO: hyperparameter 외부에서 설정하도록 수정 필요.
        training_args = TrainingArguments(
            output_dir=current_path / self.train_name,
            per_device_train_batch_size=8,
            num_train_epochs=1,
            max_steps=1000,
            # fp16=True,
            save_steps=10,
            logging_steps=30,
            learning_rate=1e-5,
            weight_decay=1e-4,
            save_total_limit=2,
            remove_unused_columns=False,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.collate_fn,
            train_dataset=self.train_dataset_transformed,
            eval_dataset=self.test_dataset_transformed,
            tokenizer=self.image_processor,
        )
        try:
            with mlflow.start_run(run_name=self.train_name) as run:
                # 첫 번째 배치 테스트
                test_dataloader = self.trainer.get_train_dataloader()
                print(f"test_dataloader = {test_dataloader}")
                test_batch = next(iter(test_dataloader))
                print("\nTest batch structure:")
                for k, v in test_batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"{k}: type={type(v)}")

                self.trainer.train()
                print(f"run_id = {run.info.run_id}")
        except Exception as e:
            traceback.print_exc()
            print(f"error occured when training and evaluating : {e}")

    def postprocess(self):
        train_model_name = f"{self.model_name}-fine-tuned"

        # TODO: model 학습 테스트 성공되는것 확인되면 활성화
        # self.model_artifacts.model = self.trainer.model

        with mlflow.start_run(run_name=train_model_name) as run:
            train_model_name = train_model_name.replace("/", "-")
            mlflow.transformers.log_model(
                transformers_model=self.model_artifacts,
                artifact_path=train_model_name,
                registered_model_name=train_model_name,
            )
            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri()
            model_version = self.client.get_latest_versions(name=train_model_name, stages=["None"])[0].version
            train_model_uri = f"models:/{train_model_name}/{model_version}"
        print(
            f"run_id: {run_id}, artifact_uri: {artifact_uri},\
                model_version: {model_version}, train_model_uri: {train_model_uri}"
        )
        # 어떻게 RDB Meatadata를 업데이트할지 구상필요.
        insert_metadata(
            run_id=run_id,
            artifact_uri=artifact_uri,
            model_version=model_version,
            model_uri=train_model_uri,
            train_model_name=train_model_name,
            restapi_url=self.restapi_url,
            restapi_token=self.restapi_token,
        )

    def transform_aug_ann(self, examples):
        """
        데이터셋 변환 함수
        """
        images = examples["image"]
        bboxes = []
        categories = []
        areas = []
        iscrowds = []
        image_ids = []

        # 각 예제에 대한 객체 정보 처리
        for idx, objects in enumerate(examples["objects"]):
            if len(objects["bbox"]) > 0:  # 객체가 있는 경우에만 처리
                category = [self.label2id[label] for label in objects["category"]]
                bboxes.append(objects["bbox"])
                categories.append(category)
                areas.append(objects["area"])
                iscrowds.append([0] * len(objects["bbox"]))
                image_ids.append([idx] * len(objects["bbox"]))
        # categories를 숫자 레이블로 변환
        # 이미지 처리
        processed = self.image_processor(images=images, return_tensors="pt")

        # 타겟 데이터 구성
        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "class_labels": torch.tensor(categories, dtype=torch.long),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowds, dtype=torch.int64),
            "image_id": torch.tensor(image_ids, dtype=torch.int64),
        }
        print(f"tranform_aug_ann = {target}")
        return {"pixel_values": processed["pixel_values"], "labels": target}

    # def formatted_anns(self, image_id, category, area, bbox):
    #     annotations = []
    #     for i in range(0, len(category)):
    #         new_ann = {
    #             "image_id": image_id,
    #             "category_id": category[i],
    #             "isCrowd": 0,
    #             "area": area[i],
    #             "bbox": list(bbox[i]),
    #         }
    #         annotations.append(new_ann)

    #     return annotations

    def collate_fn(self, batch):
        """
        배치 데이터를 처리하는 함수
        """
        pixel_values = []
        labels = []

        for item in batch:
            pixel_values.append(item["pixel_values"])
            labels.append(item["labels"])

        # 이미지 배치 처리
        pixel_values = torch.stack(pixel_values)

        # 라벨 배치 처리
        batched_labels = {
            "boxes": torch.stack([label["boxes"] for label in labels]),
            "labels": torch.stack([label["labels"] for label in labels]),
            "area": torch.stack([label["area"] for label in labels]),
            "iscrowd": torch.stack([label["iscrowd"] for label in labels]),
            "image_id": torch.stack([label["image_id"] for label in labels]),
        }

        return {"pixel_values": pixel_values, "labels": batched_labels}


def insert_metadata(
    run_id: str,
    artifact_uri: str,
    model_version: str,
    model_uri: str,
    train_model_name: str,
    restapi_url: str,
    restapi_token: str,
):
    data = {
        "name": train_model_name,
        "description": train_model_name,
        "model_provider_id": 3,
        "model_type_id": 4,
        "model_format_id": 1,
        "model_registry_schema": json.dumps(
            {
                "run_id": run_id,
                "artifact_path": artifact_uri,
                "versions": model_version,
                "model_uri": model_uri,
            }
        ),
    }
    api_endpoint = f"{restapi_url}/api/v1/models"
    headers = {"Authorization": f"Bearer {restapi_token}"}
    response = requests.post(api_endpoint, headers=headers, data=data)
    return response.json()


def get_token_from_restapi(url: str, username: str, password: str) -> str:
    response = requests.post(f"{url}/api/v1/authentications/token", data={"username": username, "password": password})
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        logger.error("cannot login to rest api.")
        return ""


def convert_array_string_to_dict(input_string):
    # 함수 내부에서 globals()를 수정하여 array를 추가
    globals()["array"] = np.array
    """
    NumPy array 문자열을 파이썬 딕셔너리로 변환

    Args:
        input_string (str): 변환할 문자열
    Returns:
        dict: 변환된 딕셔너리
    """
    # 문자열을 실행하여 딕셔너리 생성
    array_dict = eval(input_string)

    # 각 값을 파이썬 리스트로 변환
    result = {}
    for key, value in array_dict.items():
        if isinstance(value, np.ndarray):
            # object dtype인 경우 재귀적으로 처리
            if value.dtype == object:
                result[key] = [item.tolist() if isinstance(item, np.ndarray) else item for item in value]
            else:
                result[key] = value.tolist()
        else:
            result[key] = value

    return result


def convert_dataset_images(dataset):
    """
    일반 for문을 사용하여 데이터셋의 이미지를 변환합니다.
    """
    converted_data = []

    for item in tqdm(dataset, desc="Converting images"):
        try:
            # 현재 아이템의 모든 필드 복사
            converted_item = dict(item)

            # 이미지 변환
            img_dict = ast.literal_eval(item["image"])
            img_bytes = eval(img_dict["bytes"]) if isinstance(img_dict["bytes"], str) else img_dict["bytes"]
            converted_item["image"] = Image.open(io.BytesIO(img_bytes))
            converted_item["objects"] = convert_array_string_to_dict(item["objects"])

            converted_data.append(converted_item)
        except Exception as e:
            print(f"Error: {e}")
            converted_data.append(item)  # 에러 발생 시 원본 데이터 유지

    # 새로운 데이터셋 생성
    return Dataset.from_list(converted_data)


def main():
    parser = argparse.ArgumentParser(description="Model Training")
    """
            train_name: str,
            model_name: str,
            model_uri: str,
            mlflow_tracking_uri: str,
            mlflow_experiment_name: str,
            dataset_artifact_uri: str,
    """
    parser.add_argument("--train_name", type=str, required=True, help="Name of the train_execution")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--model_uri", type=str, required=True, help="URI of the MLflow model")
    parser.add_argument("--mlflow_tracking_uri", type=str, required=True, help="MLflow tracking server URI")
    parser.add_argument("--mlflow_experiment_name", type=str, required=True, help="MLflow experiment name")
    parser.add_argument("--mlflow_s3_endpoint_url", type=str, required=True, help="MLflow s3 endpoint url")
    parser.add_argument("--aws_access_key_id", type=str, required=True, help="MLflow s3 username")
    parser.add_argument("--aws_secret_access_key", type=str, required=True, help="MLflow s3 password")
    parser.add_argument("--dataset_artifact_uri", type=str, required=True, help="URI of the MLflow dataset artifact")

    parser.add_argument("--restapi_url", type=str, required=True, help="RESTAPI URL")
    parser.add_argument("--restapi_username", type=str, required=True, help="RESTAPI User ID")
    parser.add_argument("--restapi_password", type=str, required=True, help="RESTAPI User Password")

    # 인자 파싱
    args = parser.parse_args()
    model = TrainModel(
        train_name=args.train_name,
        model_name=args.model_name,
        model_uri=args.model_uri,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_s3_endpoint_url=args.mlflow_s3_endpoint_url,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        dataset_artifact_uri=args.dataset_artifact_uri,
        restapi_url=args.restapi_url,
        restapi_username=args.restapi_username,
        restapi_password=args.restapi_password,
    )

    # TODO: 코드 동작하지 않음. 추후 개발 필요.
    # model.preprocess()
    # model.train()

    model.postprocess()


if __name__ == "__main__":
    main()
