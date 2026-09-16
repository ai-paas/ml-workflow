import abc
import logging

logger = logging.getLogger(__name__)


class StorageClient(abc.ABC):
    """오브젝트 스토리지 추상 클라이언트

    데이터셋 파일의 업로드/다운로드/삭제를 위한 공통 인터페이스.
    배포 환경에 따라 S3 호환 스토리지 또는 Hub-Connect API를 사용하는
    구현체가 팩토리 함수를 통해 선택된다.
    """

    @abc.abstractmethod
    def upload_file(self, local_path: str, object_key: str) -> str:
        """로컬 파일을 오브젝트 스토리지에 업로드

        Args:
            local_path: 업로드할 로컬 파일 경로
            object_key: 저장할 오브젝트 키 (예: "datasets/{name}/{filename}")

        Returns:
            업로드된 오브젝트 키
        """
        ...

    @abc.abstractmethod
    def download_file(self, object_key: str, local_dir: str) -> str:
        """오브젝트를 로컬 디렉터리에 다운로드

        Args:
            object_key: 다운로드할 오브젝트 키
            local_dir: 다운로드 대상 로컬 디렉터리

        Returns:
            다운로드된 로컬 파일 경로
        """
        ...

    @abc.abstractmethod
    def delete_object(self, object_key: str) -> bool:
        """단일 오브젝트 삭제

        Args:
            object_key: 삭제할 오브젝트 키

        Returns:
            삭제 성공 여부
        """
        ...

    @abc.abstractmethod
    def delete_folder(self, prefix: str) -> bool:
        """특정 prefix 하위의 모든 오브젝트 삭제

        Args:
            prefix: 삭제할 폴더 경로 (예: "datasets/{name}/")

        Returns:
            삭제 성공 여부
        """
        ...
