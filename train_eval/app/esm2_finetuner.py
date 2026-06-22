#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESM2 (facebook/esm2_t6_8M_UR50D) 파인튜닝.

HuggingFace Trainer + PEFT(LoRA) 로 TCR-Epitope 결합 이진 분류를 학습한다.
backend/notebook/esm2-fine-tuning.ipynb 시나리오를 시스템에 녹인 것.

이 모듈은 train_eval.main() 의 `--model_kind esm2` 분기에서만 lazy import 된다.
(그래서 transformers/peft 는 ESM2 경로에서만 로드된다.)
"""

import zipfile
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from app._common import download_dataset_dir, get_token_from_restapi, setup_mlflow, update_experiment
from datasets import Dataset, DatasetDict
from loguru import logger
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EsmForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

MODEL_ID = "facebook/esm2_t6_8M_UR50D"
MAX_LENGTH = 80
REQUIRED_COLUMNS = ("epitope", "cdr3b", "label")

current_path = Path(__file__).absolute().parent


class MlflowCallback(TrainerCallback):
    """Trainer 로그를 metrics_polling 이 읽는 키로 MLflow 에 송출.

    - 학습 loss → `train/total_loss`, epoch → `train/epoch` (YOLOX 와 동일 키)
    - 평가 메트릭 → `eval_loss`/`eval_accuracy`/`eval_precision`/`eval_recall`/`eval_auc_roc`/`eval_auc_pr`
    YOLOX 전용 `val/best_ap` 는 송출하지 않는다(ESM2 의 average_precision 은 NULL 로 남는다).
    """

    EVAL_KEYS = (
        "eval_loss",
        "eval_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_auc_roc",
        "eval_auc_pr",
    )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        if "epoch" in logs:
            mlflow.log_metric("train/epoch", float(logs["epoch"]), step=step)
        if "loss" in logs:
            mlflow.log_metric("train/total_loss", float(logs["loss"]), step=step)
        for key in self.EVAL_KEYS:
            if key in logs and logs[key] is not None:
                mlflow.log_metric(key, float(logs[key]), step=step)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    # softmax (positive class 확률)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    pos_probs = probs[:, 1]
    preds = logits.argmax(axis=-1)

    has_both_classes = len(set(labels.tolist())) > 1
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels, pos_probs)) if has_both_classes else 0.0,
        "auc_pr": float(average_precision_score(labels, pos_probs)) if has_both_classes else 0.0,
    }


def _resolve_esm_save_args(save_period: str, epochs: int) -> dict:
    """save_period 의미를 ESM2(HF Trainer) save 인자로 환산 (YOLOX 와 호환)."""
    if save_period == "-1":
        return {"save_strategy": "no", "load_best_model_at_end": False}
    n = max(1, int(save_period))
    return {
        "save_strategy": "epoch",
        "save_total_limit": max(1, epochs // n),
        "load_best_model_at_end": True,
    }


class EsmFineTuner:
    """ESM2 LoRA 파인튜너. train_eval.CustomTrainModel 과 동일한 preprocess/train 인터페이스."""

    def __init__(
        self,
        train_name: str,
        experiment_id: int,
        mlflow_tracking_uri: str,
        mlflow_experiment_name: str,
        restapi_url: str,
        restapi_username: str,
        restapi_password: str,
        batch_size: str,
        epochs: str,
        save_period: str,
        weight_decay: str,
        learning_rate: str,
        model_artifact_path: str,
        args,
    ):
        self.train_name = train_name
        self.experiment_id = experiment_id
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.restapi_url = restapi_url
        self.restapi_username = restapi_username
        self.restapi_password = restapi_password
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_period = save_period
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        # 카탈로그 ESM2 base 모델의 MLflow 아티팩트 경로(있으면 여기서 base 로드, 없으면 HF Hub)
        self.model_artifact_path = model_artifact_path
        self.args = args

        self.output_dir = current_path / "outputs" / "esm2"
        self.train_csv: Optional[Path] = None
        self.val_csv: Optional[Path] = None
        self.base_dir: Optional[str] = None

    @classmethod
    def from_args(cls, args) -> "EsmFineTuner":
        return cls(
            train_name=args.train_name,
            experiment_id=args.experiment_id,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            restapi_url=args.restapi_url,
            restapi_username=args.restapi_username,
            restapi_password=args.restapi_password,
            batch_size=args.batch_size,
            epochs=args.epochs,
            save_period=args.save_period,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            model_artifact_path=getattr(args, "model_artifact_path", "") or "",
            args=args,
        )

    def _resolve_base_path(self) -> str:
        """base 모델 로드 경로. MLflow 에서 받은 디렉토리가 있으면 그것을, 없으면 HF Hub(MODEL_ID)."""
        if self.base_dir:
            return self.base_dir
        return MODEL_ID

    @staticmethod
    def _find_csv(root: Path, names) -> Optional[Path]:
        for name in names:
            for candidate in (root / name, *root.glob(f"*/{name}")):
                if candidate.is_file():
                    return candidate
        return None

    def preprocess(self):
        """MLflow base 모델 다운로드(선택) + 데이터셋 다운로드 → zip 해제 → train/val CSV 확정."""
        setup_mlflow(self.args)

        # base 모델: MLflow 에 등록된 카탈로그 base 아티팩트를 우선 사용(없으면 HF Hub fallback)
        if self.model_artifact_path:
            try:
                self.base_dir = mlflow.artifacts.download_artifacts(artifact_uri=self.model_artifact_path)
                logger.info(f"MLflow base 모델 다운로드: {self.model_artifact_path} → {self.base_dir}")
            except Exception as e:
                logger.warning(f"MLflow base 다운로드 실패, HF Hub 로 fallback 합니다: {e}")
                self.base_dir = None

        download_dir = download_dataset_dir(self.args)
        zips = list(Path(download_dir).glob("*.zip"))
        if not zips:
            raise FileNotFoundError("데이터셋 zip 을 찾을 수 없습니다.")

        extract_dir = Path(download_dir) / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips[0], "r") as zf:
            zf.extractall(extract_dir)

        self.train_csv = self._find_csv(extract_dir, ("train.csv", "train.sample.csv"))
        self.val_csv = self._find_csv(extract_dir, ("val.csv", "val.sample.csv"))
        if self.train_csv is None:
            any_csv = sorted(extract_dir.glob("**/*.csv"))
            if not any_csv:
                raise FileNotFoundError("CSV 파일을 찾을 수 없습니다.")
            self.train_csv = any_csv[0]
        logger.info(f"train CSV: {self.train_csv} / val CSV: {self.val_csv}")

    def _build_datasets(self, tokenizer) -> DatasetDict:
        df = pd.read_csv(self.train_csv)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

        if self.val_csv is not None:
            train_df = df
            val_df = pd.read_csv(self.val_csv)
        else:
            train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

        cols = list(REQUIRED_COLUMNS)
        ds = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df[cols].reset_index(drop=True)),
                "val": Dataset.from_pandas(val_df[cols].reset_index(drop=True)),
            }
        )

        def tokenize(batch):
            texts = [str(e) + str(c) for e, c in zip(batch["epitope"], batch["cdr3b"])]
            return tokenizer(texts, truncation=True, max_length=MAX_LENGTH)

        ds = ds.map(tokenize, batched=True, remove_columns=["epitope", "cdr3b"])
        ds = ds.rename_column("label", "labels")
        return ds

    def train(self):
        """ESM2 + LoRA 학습 후 어댑터를 MLflow 'adapter' 아티팩트로 등록."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        base_path = self._resolve_base_path()
        logger.info(f"ESM2 base 로드 경로: {base_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        ds = self._build_datasets(tokenizer)

        model = EsmForSequenceClassification.from_pretrained(base_path, num_labels=2)
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            bias="none",
            r=8,
            lora_alpha=16,
            lora_dropout=0.2,
            target_modules=["query", "key", "value"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, peft_config)
        try:
            model.base_model.model.classifier.modules_to_save.default.dropout.p = 0.25
        except AttributeError:
            logger.warning("classifier dropout 패치를 적용하지 못했습니다(구조 상이). 무시하고 진행합니다.")

        epochs_int = int(self.epochs)
        save_args = _resolve_esm_save_args(self.save_period, epochs_int)
        training_args = TrainingArguments(
            seed=42,
            fp16=torch.cuda.is_available(),
            output_dir=str(self.output_dir),
            eval_strategy="epoch",
            logging_strategy="epoch",
            num_train_epochs=epochs_int,
            learning_rate=float(self.learning_rate),
            per_device_train_batch_size=int(self.batch_size),
            per_device_eval_batch_size=max(int(self.batch_size) * 4, 64),
            gradient_accumulation_steps=4,
            weight_decay=float(self.weight_decay),
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            metric_for_best_model="auc_roc",
            greater_is_better=True,
            report_to=[],
            **save_args,
        )

        callbacks = [MlflowCallback()]
        if save_args.get("load_best_model_at_end"):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=max(1, int(epochs_int * 0.2)),
                    early_stopping_threshold=0.01,
                )
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["val"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=_compute_metrics,
            callbacks=callbacks,
        )

        token = get_token_from_restapi(self.restapi_url, self.restapi_username, self.restapi_password)
        with mlflow.start_run(run_name=self.train_name) as run:
            # run_id 를 백엔드 experiment 에 즉시 전달 (metrics_polling 이 이를 기다린다)
            update_experiment(
                restapi_url=self.restapi_url,
                restapi_token=token,
                experiment_id=self.experiment_id,
                status="RUNNING",
                mlflow_run_id=run.info.run_id,
            )

            trainer.train()

            # PEFT 어댑터 디렉토리 저장 후 'adapter' 아티팩트로 업로드
            adapter_dir = self.output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            mlflow.log_artifacts(local_path=str(adapter_dir), artifact_path="adapter")
            logger.info(f"ESM2 LoRA 어댑터 등록 완료: {adapter_dir} → mlflow artifact 'adapter'")

    def postprocess(self):
        """현재 단계에서는 별도 후처리 없음 (모델 등록은 register_model 컴포넌트가 수행)."""
        return None
