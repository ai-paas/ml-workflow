#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from loguru import logger

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

# 현재 파일의 절대 경로 얻기
current_path = Path(__file__).absolute().parent
parent_path = current_path.parent


class YOLOXTrainTest:
    """YOLOX tools/train.py를 직접 실행하는 학습 클래스"""

    def __init__(
        self,
        exp_file: str,  # 실험 파일 경로
        batch_size: int = 64,
        devices: int = 1,
        fp16: bool = False,
        cache: Optional[str] = None,
        resume: bool = False,
        ckpt: Optional[str] = None,
        start_epoch: Optional[int] = None,
        occupy: bool = False,
        logger_type: str = "tensorboard",
        model_name: Optional[str] = None,
        **kwargs,
    ):
        """
        YOLOX tools/train.py를 직접 실행하는 학습 클래스 초기화

        Args:
            exp_file: YOLOX 실험 파일 경로
            batch_size: 배치 크기
            devices: 사용할 GPU 수
            fp16: FP16 사용 여부
            cache: 캐시 사용 여부 (None, "ram", "disk")
            resume: 재개 여부
            ckpt: 체크포인트 경로
            start_epoch: 시작 에폭
            occupy: GPU 점유 여부
            logger_type: 로거 타입
            model_name: 모델명
        """

        self.exp_file = exp_file
        self.batch_size = batch_size
        self.devices = devices
        self.fp16 = fp16
        self.cache = cache
        self.resume = resume
        self.ckpt = ckpt
        self.start_epoch = start_epoch
        self.occupy = occupy
        self.logger_type = logger_type
        self.model_name = model_name

        # 기본 설정
        self.data_dir = parent_path / "data"
        self.output_dir = parent_path / "YOLOX_outputs"

    def download_voc_dataset(self):
        """VOC 데이터셋 다운로드"""
        logger.info("VOC 데이터셋 다운로드를 시작합니다...")

        # 데이터 디렉토리 생성
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # VOC 데이터셋 URL들
        voc_urls = {
            "VOC2007_trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "VOC2007_test": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
            "VOC2012_trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        }

        for name, url in voc_urls.items():
            tar_file = self.data_dir / f"{name}.tar"

            # 이미 다운로드되었는지 확인
            if tar_file.exists():
                logger.info(f"{name} 이미 다운로드됨: {tar_file}")
                continue

            try:
                logger.info(f"{name} 다운로드 시작: {url}")

                # wget 명령으로 다운로드
                result = subprocess.run(
                    ["wget", "-O", str(tar_file), url], cwd=str(self.data_dir), capture_output=True, text=True
                )

                if result.returncode == 0:
                    logger.info(f"{name} 다운로드 완료")

                    # 압축 해제
                    logger.info(f"{name} 압축 해제 시작")
                    result = subprocess.run(
                        ["tar", "-xvf", str(tar_file)], cwd=str(self.data_dir), capture_output=True, text=True
                    )

                    if result.returncode == 0:
                        logger.info(f"{name} 압축 해제 완료")
                        # 압축 파일 제거 (옵션)
                        # tar_file.unlink()
                    else:
                        logger.error(f"{name} 압축 해제 실패: {result.stderr}")
                        raise RuntimeError(f"압축 해제 실패: {result.stderr}")
                else:
                    logger.error(f"{name} 다운로드 실패: {result.stderr}")
                    raise RuntimeError(f"다운로드 실패: {result.stderr}")

            except Exception as e:
                logger.error(f"{name} 처리 중 오류: {e}")
                raise

        logger.info("VOC 데이터셋 다운로드 완료")

    def preprocess(self):
        """데이터 전처리"""
        logger.info("데이터 전처리를 시작합니다...")

        # # VOC 데이터셋 다운로드
        # self.download_voc_dataset()
        # os.environ["YOLOX_DATADIR"] = str(self.data_dir)

        # 데이터 구조 확인
        voc_dirs = ["VOC2007", "VOC2012"]
        for voc_dir in voc_dirs:
            voc_path = self.data_dir / "VOCdevkit" / voc_dir
            if voc_path.exists():
                logger.info(f"데이터 구조 확인: {voc_path}")
                for subdir in ["Annotations", "ImageSets", "JPEGImages"]:
                    subdir_path = voc_path / subdir
                    if subdir_path.exists():
                        logger.info(f"  {subdir}: {len(list(subdir_path.iterdir()))} 파일")
                    else:
                        logger.warning(f"  {subdir}: 디렉토리가 존재하지 않음")

        logger.info("데이터 전처리 완료")

    def train(self):
        """YOLOX tools/train.py를 직접 실행하여 모델 학습"""
        logger.info("YOLOX tools/train.py를 사용한 모델 학습을 시작합니다...")

        # YOLOX tools/train.py 명령 구성
        cmd = [
            sys.executable,
            "-m",
            "yolox.tools.train",
            "-f",
            self.exp_file,
            "-d",
            str(self.devices),
            "-b",
            str(self.batch_size),
        ]

        # 선택적 인자들 추가
        if self.fp16:
            cmd.append("--fp16")

        if self.cache:
            cmd.extend(["--cache", self.cache])

        if self.resume:
            cmd.append("--resume")

        if self.ckpt:
            cmd.extend(["-c", self.ckpt])

        if self.start_epoch is not None:
            cmd.extend(["-e", str(self.start_epoch)])

        if self.occupy:
            cmd.append("-o")

        # 로거 타입 설정
        cmd.extend(["-l", self.logger_type])

        # 실험명 설정
        if self.model_name:
            cmd.extend(["-n", self.model_name])

        logger.info(f"실행 명령: {' '.join(cmd)}")

        try:
            # 환경 변수 설정
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.devices))

            # 데이터 디렉토리 환경 변수 설정 (YOLOX가 참조할 수 있도록)
            env["YOLOX_DATADIR"] = str(self.data_dir)

            # YOLOX 학습 실행
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
                cwd=str(current_path),
            )

            # 실시간 로그 출력
            for line in iter(process.stdout.readline, ""):
                if line:
                    line = line.rstrip()
                    logger.info(line)

            process.wait()

            if process.returncode == 0:
                logger.info("YOLOX 학습이 성공적으로 완료되었습니다!")
            else:
                raise RuntimeError(f"YOLOX 학습이 실패했습니다. 종료 코드: {process.returncode}")

        except Exception as e:
            logger.error(f"학습 중 오류 발생: {e}")
            raise

    def postprocess(self):
        """후처리 작업"""
        logger.info("후처리를 시작합니다...")

        try:
            # 학습 결과 파일 확인
            yolox_outputs = Path("YOLOX_outputs")
            if yolox_outputs.exists():
                # 최신 실험 폴더 찾기
                exp_folders = [f for f in yolox_outputs.iterdir() if f.is_dir()]
                if exp_folders:
                    latest_exp = max(exp_folders, key=lambda x: x.stat().st_mtime)
                    logger.info(f"최신 실험 폴더: {latest_exp}")

                    # 모델 파일 확인
                    model_files = list(latest_exp.glob("*.pth"))
                    for model_file in model_files:
                        logger.info(f"모델 파일 생성됨: {model_file}")

                    # 로그 파일 확인
                    log_files = list(latest_exp.glob("*.log"))
                    for log_file in log_files:
                        logger.info(f"로그 파일 생성됨: {log_file}")

                    # TensorBoard 로그 확인
                    tb_dirs = list(latest_exp.glob("tensorboard*"))
                    for tb_dir in tb_dirs:
                        if tb_dir.is_dir():
                            logger.info(f"TensorBoard 로그 생성됨: {tb_dir}")

            logger.info("후처리 완료")

        except Exception as e:
            logger.error(f"후처리 중 오류 발생: {e}")
            raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="YOLOX tools/train.py를 직접 실행하는 모델 학습")

    # YOLOX 설정 (tools/train.py와 동일한 옵션)
    parser.add_argument("-f", "--exp_file", type=str, required=True, help="YOLOX 실험 파일 경로")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="배치 크기")
    parser.add_argument("-d", "--devices", type=int, default=1, help="사용할 GPU 장치 수")
    parser.add_argument("--fp16", action="store_true", help="FP16 사용 여부")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="캐시 사용 여부 (ram/disk)")
    parser.add_argument("--resume", action="store_true", help="재개 여부")
    parser.add_argument("-c", "--ckpt", type=str, help="체크포인트 경로")
    parser.add_argument("-e", "--start_epoch", type=int, help="시작 에폭")
    parser.add_argument("-o", "--occupy", action="store_true", help="GPU 점유 여부")
    parser.add_argument("-l", "--logger", type=str, default="tensorboard", help="로거 타입")
    parser.add_argument("-n", "--name", type=str, help="모델명")

    # 인자 파싱
    args = parser.parse_args()

    # 모델 초기화
    model = YOLOXTrainTest(
        exp_file=args.exp_file,
        batch_size=args.batch_size,
        devices=args.devices,
        fp16=args.fp16,
        cache=args.cache,
        resume=args.resume,
        ckpt=args.ckpt,
        start_epoch=args.start_epoch,
        occupy=args.occupy,
        logger_type=args.logger,
        model_name=args.name,
    )

    try:
        # 데이터 전처리 (VOC 데이터셋 다운로드)
        model.preprocess()

        # 학습 실행
        model.train()

        # 후처리
        model.postprocess()

        logger.info("학습 완료!")

    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
