-- TraceMind 代码索引元数据（小表全量；不含 code_chunk_meta 行数据）
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Records of `code_repo` (1 rows)
-- ----------------------------
INSERT INTO `code_repo` (`id`, `repo_code`, `repo_name`, `local_root`, `status`, `created_at`, `updated_at`) VALUES (1, 'wintogether-bff', '赢在一起 BFF', 'D:\\Codes\\wintogether-bff', 1, '2026-10-20 11:08:43', '2026-10-20 11:08:43');

-- ----------------------------
-- Records of `code_index_version` (3 rows)
-- ----------------------------
INSERT INTO `code_index_version` (`id`, `repo_id`, `branch`, `image_tag`, `git_commit`, `index_version`, `embedding_model`, `embedding_dim`, `milvus_collection`, `status`, `file_count`, `chunk_count`, `is_current`, `remark`, `started_at`, `ready_at`, `created_at`) VALUES (1, 1, 'dev_3.7.0_T21新品2', 'wt-bff:3.7.0-a4f5e532e5', 'a4f5e532e557ee99083f3bebe0a0675e3837272d', 'idx_20261020_110843', 'batiai/qwen3-embedding:0.6b', 1024, 'wt_code_chunks', 'ready', 1, 3, 0, '本地初始化：按当前 Git 分支生成镜像号', '2026-10-20 11:08:43', '2026-10-20 13:37:18', '2026-10-20 11:08:43');
INSERT INTO `code_index_version` (`id`, `repo_id`, `branch`, `image_tag`, `git_commit`, `index_version`, `embedding_model`, `embedding_dim`, `milvus_collection`, `status`, `file_count`, `chunk_count`, `is_current`, `remark`, `started_at`, `ready_at`, `created_at`) VALUES (2, 1, 'dev_3.7.0_T21新品2', 'wt-bff:3.7.0-a4f5e532e5', 'a4f5e532e557ee99083f3bebe0a0675e3837272d', 'idx_20261020_135036', 'batiai/qwen3-embedding:0.6b', 1024, 'wt_code_chunks', 'failed', 18, 114, 0, 'full repo index | aborted: smoke-test process exited', '2026-10-20 13:50:37', NULL, '2026-10-20 13:50:36');
INSERT INTO `code_index_version` (`id`, `repo_id`, `branch`, `image_tag`, `git_commit`, `index_version`, `embedding_model`, `embedding_dim`, `milvus_collection`, `status`, `file_count`, `chunk_count`, `is_current`, `remark`, `started_at`, `ready_at`, `created_at`) VALUES (3, 1, 'dev_3.7.0_T21新品2', 'wt-bff:3.7.0-a4f5e532e5', 'a4f5e532e557ee99083f3bebe0a0675e3837272d', 'idx_20261020_135631', 'batiai/qwen3-embedding:0.6b', 1024, 'wt_code_chunks', 'ready', 3719, 19170, 1, 'resume 2026-10-20T15:11:56', '2026-10-20 13:56:32', '2026-10-20 16:23:05', '2026-10-20 13:56:31');

SET FOREIGN_KEY_CHECKS = 1;
