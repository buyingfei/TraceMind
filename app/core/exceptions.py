"""业务异常定义（骨架占位）。"""


class DiagnoseAgentError(Exception):
    """应用通用异常。"""

    def __init__(self, message: str, code: str = "DIAGNOSE_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class NotImplementedCapabilityError(DiagnoseAgentError):
    """尚未实现的能力。"""

    def __init__(self, capability: str) -> None:
        super().__init__(
            message=f"能力尚未实现: {capability}",
            code="NOT_IMPLEMENTED",
        )
