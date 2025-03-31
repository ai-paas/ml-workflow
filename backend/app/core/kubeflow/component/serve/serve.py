import uuid

from kfp import dsl, local

# local.init(runner=local.SubprocessRunner())


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "mlflow==2.17.0",
        "transformers==4.45.2",
        "kserve==0.11.2",
        "torch==2.4.1",
        "torchvision==0.19.1",
    ],
)
def serving_component(
    inference_service_name: str,
    mlflow_tracking_uri: str,
    mlflow_s3_endpoint_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    mlflow_experiment_name: str,
    model_name: str,
    model_uri: str = None,
    s3_storage_uri: str = None,
    kserve_gpu: bool = False,
) -> str:
    import logging

    from kserve import (
        KServeClient,
        V1beta1InferenceService,
        V1beta1InferenceServiceSpec,
        V1beta1ModelFormat,
        V1beta1ModelSpec,
        V1beta1PredictorSpec,
        constants,
    )
    from kubernetes import client, config
    from transformers import Owlv2ForObjectDetection

    logger = logging.getLogger(__name__)

    # config.load_kube_config()
    config.load_incluster_config()
    # Inference Service 이름 및 네임스페이스 설정
    kserve_client = KServeClient()

    # TODO: mocking data. 외부에서 설정할수 있도록 수정 필요.
    namespace = "kubeflow-user-example-com"

    # TODO: s3_storage 지원시 활용 필요
    # env_vars = [
    #     client.V1EnvVar(name="AWS_ACCESS_KEY_ID", value_from=client.V1EnvVarSource(
    #         secret_key_ref=client.V1SecretKeySelector(name="minio-credentials", key="AWS_ACCESS_KEY_ID")
    #     )),
    #     client.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value_from=client.V1EnvVarSource(
    #         secret_key_ref=client.V1SecretKeySelector(name="minio-credentials", key="AWS_SECRET_ACCESS_KEY")
    #     )),
    #     client.V1EnvVar(name="AWS_ENDPOINT_URL", value="http://your-minio-server:9000")
    # ]
    # model_spec = V1beta1ModelSpec(
    #     model_format=V1beta1ModelFormat(name="tensorflow"),
    #     protocol_version="v2",
    #     storage_uri=model_storage_uri,
    # )
    logger.info(f"Use GPU = {kserve_gpu}")
    predictor_spec = V1beta1PredictorSpec(
        # TODO: s3_storage 지원시 활용 필요
        # model=model_spec,
        min_replicas=1,
        containers=[
            client.V1Container(
                name="kserve-container",
                image="aipaas-harbor.surromind.ai/ml-workflow/inference:latest",  # TensorFlow 모델 서빙을 위한 이미지
                # env=env_vars
                args=[
                    f"--model_name={model_name}",
                    f"--model_uri={model_uri}",
                    f"--mlflow_tracking_uri={mlflow_tracking_uri}",
                    f"--mlflow_experiment_name={mlflow_experiment_name}",
                    f"--mlflow_s3_endpoint_url={mlflow_s3_endpoint_url}",
                    f"--aws_access_key_id={aws_access_key_id}",
                    f"--aws_secret_access_key={aws_secret_access_key}",
                ],
                resources=client.V1ResourceRequirements(
                    requests=(
                        {"memory": "2Gi", "cpu": "200m", "nvidia.com/gpu": "1"}  # GPU 리소스 제한
                        if kserve_gpu
                        else {
                            "memory": "2Gi",
                            "cpu": "200m",
                        }
                    ),
                    limits=(
                        {"memory": "4Gi", "cpu": "500m", "nvidia.com/gpu": "1"}  # GPU 리소스 제한
                        if kserve_gpu
                        else {
                            "memory": "4Gi",
                            "cpu": "500m",
                        }
                    ),
                ),
            )
        ],
    )

    inference_service_spec = V1beta1InferenceServiceSpec(predictor=predictor_spec)

    inference_service = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=inference_service_name,
            namespace=namespace,
        ),
        spec=inference_service_spec,
    )
    # Inference Service 생성 (배포)
    kserve_client.create(inference_service)

    return "Serve completed."


# local test

# serving_component(
#     mlflow_tracking_uri="https://aipaas-mlflow.surromind.ai",
#     model_uri="models:/google-owlv2-base-patch16-ensemble/2",
#     mlflow_experiment_name="ml_workflow_dev",
#     model_storage_uri="s3://mlflow/3/be03517338db4d38ad5157b9ff7c152b/artifacts/google-owlv2-base-patch16-ensemble",
#     inference_service_name=f"mlworkflow-{uuid.uuid4()}",
# )
