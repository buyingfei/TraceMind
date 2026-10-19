"""对象存储服务（骨架占位）。"""

from app.core.exceptions import NotImplementedCapabilityError


def upload_file(local_path: str, object_key: str) -> str:
    """后续接入阿里云 OSS，返回可访问 URL。"""
    raise NotImplementedCapabilityError("oss_service.upload_file")
