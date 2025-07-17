from kfp import dsl, local

# @dsl.component(base_image="python:3.10-slim",
#                    target_image=f"aipaas-harbor.surromind.ai/ml-workflow/train:{version}",
#                    )
# local.init(runner=local.SubprocessRunxner())
# local.init(runner=local.DockerRunner())


@dsl.container_component
def container_train_eval_component(
    train_name: str,
    model_name: str,
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
):
    return dsl.ContainerSpec(
        # TODO: harbor URL 변수로 관리 필요
        image="aipaas-harbor.surromind.ai/ml-workflow/train:latest",
        # TODO: Kubernetes는 보안과 유연성을 위해 기본적으로 컨테이너의 ENTRYPOINT를 무시하고 override함.
        command=["python", "-m", "app.train_eval"],
        args=[
            f"--train_name={train_name}",
            f"--model_name={model_name}",
            f"--model_artifact_path={model_artifact_path}",
            f"--model_uri={model_uri}",
            f"--mlflow_tracking_uri={mlflow_tracking_uri}",
            f"--mlflow_experiment_name={mlflow_experiment_name}",
            f"--mlflow_s3_endpoint_url={mlflow_s3_endpoint_url}",
            f"--aws_access_key_id={aws_access_key_id}",
            f"--aws_secret_access_key={aws_secret_access_key}",
            f"--dataset_artifact_uri={dataset_artifact_uri}",
            f"--restapi_url={restapi_url}",
            f"--restapi_username={restapi_username}",
            f"--restapi_password={restapi_password}",
        ],
        # args=[train_name, model_name, model_uri, mlflow_tracking_uri,
        #       mlflow_experiment_name, dataset_artifact_uri]
    )


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
