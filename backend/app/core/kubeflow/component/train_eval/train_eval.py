from config.settings import get_settings
from kfp import dsl

settings = get_settings()


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "kubernetes==28.1.0",
    ],
)
def container_train_eval_component(
    model_id: int,
    experiment_id: int,
    train_name: str,
    model_artifact_path: str,
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
    gpu_limit: str,
    batch_size: str,
    epochs: str,
    save_period: str,
    weight_decay: str,
    lr0: str,
    lrf: str,
    namespace: str,
    train_image_url: str,
) -> str:
    import json
    import logging
    import time

    from kubernetes import client
    from kubernetes import config as k8s_config

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Kubernetes 설정
        k8s_config.load_incluster_config()

        logger.info(f"Creating training job for model_id: {model_id}, experiment_id: {experiment_id}")

        # 리소스 설정 (Pod YAML 형식과 유사)
        resources = client.V1ResourceRequirements(
            requests={
                "nvidia.com/gpu": gpu_limit,
            },
            limits={
                "nvidia.com/gpu": gpu_limit,
            },
        )

        logger.info(f"GPU resources added: {gpu_limit} GPU(s) (fixed to: {gpu_limit})")

        # Job 생성
        job_name = f"train-eval-{model_id}-{experiment_id}-{int(time.time())}"

        # 컨테이너 args 구성
        container_args = [
            "--model_id",
            str(model_id),
            "--experiment_id",
            str(experiment_id),
            "--train_name",
            train_name,
            "--model_artifact_path",
            model_artifact_path,
            "--model_uri",
            model_uri,
            "--mlflow_tracking_uri",
            mlflow_tracking_uri,
            "--mlflow_experiment_name",
            mlflow_experiment_name,
            "--mlflow_s3_endpoint_url",
            mlflow_s3_endpoint_url,
            "--aws_access_key_id",
            aws_access_key_id,
            "--aws_secret_access_key",
            aws_secret_access_key,
            "--dataset_artifact_uri",
            dataset_artifact_uri,
            "--restapi_url",
            restapi_url,
            "--restapi_username",
            restapi_username,
            "--restapi_password",
            restapi_password,
            "--gpu_limit",
            gpu_limit,
            "--batch_size",
            batch_size,
            "--epochs",
            epochs,
            "--save_period",
            save_period,
            "--weight_decay",
            weight_decay,
            "--lr0",
            lr0,
            "--lrf",
            lrf,
        ]

        job = client.V1Job(
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=namespace,
                labels={
                    "model-id": str(model_id),
                    "experiment-id": str(experiment_id),
                    "app": "train-eval",
                },
            ),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "model-id": str(model_id),
                            "experiment-id": str(experiment_id),
                            "app": "train-eval",
                        },
                        annotations={
                            "sidecar.istio.io/inject": "false",
                        },
                    ),
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        containers=[
                            client.V1Container(
                                name="train-eval",
                                image=train_image_url,
                                command=["python", "-m", "app.train_eval"],
                                args=container_args,
                                resources=resources,
                            )
                        ],
                    ),
                ),
                backoff_limit=3,
                ttl_seconds_after_finished=1800,  # Job 완료 후 30분 뒤 자동 삭제
            ),
        )

        # Job 생성
        batch_v1 = client.BatchV1Api()
        batch_v1.create_namespaced_job(namespace=namespace, body=job)
        logger.info(f"Created Job: {job_name}")

        # Job 완료 대기 (최대 2시간)
        max_wait = 7200  # 120분
        wait_interval = 15  # 15초 간격
        elapsed = 0
        job_completed = False

        while elapsed < max_wait:
            try:
                job_status = batch_v1.read_namespaced_job_status(name=job_name, namespace=namespace)

                if job_status.status.succeeded:
                    logger.info(f"Job {job_name} completed successfully")
                    job_completed = True
                    break
                elif job_status.status.failed:
                    logger.error(f"Job {job_name} failed")
                    raise RuntimeError(f"Job {job_name} failed")

                time.sleep(wait_interval)
                elapsed += wait_interval

            except client.exceptions.ApiException as e:
                # Job이 삭제된 경우 (404) - ttlSecondsAfterFinished에 의해 자동 삭제된 것으로 간주
                if e.status == 404:
                    logger.info(
                        f"Job {job_name} not found (likely deleted after completion by ttlSecondsAfterFinished)"
                    )
                    job_completed = True
                    break
                else:
                    logger.warning(f"Error checking job status: {e.status} - {e.reason}")
                    time.sleep(wait_interval)
                    elapsed += wait_interval
            except Exception as e:
                logger.warning(f"Error checking job status: {e}")
                time.sleep(wait_interval)
                elapsed += wait_interval

        if not job_completed:
            raise RuntimeError(f"Job {job_name} did not complete within {max_wait} seconds")

        return json.dumps({"job_name": job_name, "status": "completed"})

    except Exception as e:
        logger.error(f"Failed to create training job: {str(e)}")
        return json.dumps({"status": "failed", "error": str(e)})


# local test info


# container_train_eval_component(
#     mlflow_tracking_uri="https://aipaas-mlflow.surromind.ai",
#     train_name="train-google-owlv2",
#     model_uri="models:/google-owlv2-base-patch16-ensemble/2",
#     model_name="google/owlv2-base-patch16-ensemble",
#     dataset_artifact_uri="mlflow-artifacts:/3/c6b9909566a349b1b0d11a97e796281d/artifacts/owlv2-dataset",
#     mlflow_experiment_name="ml_workflow_dev",
#     rest_api_url="http://0.0.0.0:8000/api/v1"
# )
