"""不依赖外部 Milvus 的单元测试。"""

from app.services.embedding_service import _local_hash_embedding


def test_local_hash_embedding_dim_and_norm() -> None:
    vec = _local_hash_embedding("连锁自动入库", 32)
    assert len(vec) == 32
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6
