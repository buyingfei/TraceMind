-- TraceMind 本系统 MySQL schema（由当前库 SHOW CREATE TABLE 导出）
-- 库名以 .env 中 MYSQL_DATABASE 为准（默认 diagnose-agent）
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for `chat_conversation`
-- ----------------------------
DROP TABLE IF EXISTS `chat_conversation`;
CREATE TABLE `chat_conversation` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `title` varchar(256) NOT NULL DEFAULT '新对话' COMMENT '会话标题',
  `status` varchar(32) NOT NULL DEFAULT 'active' COMMENT 'active/archived/deleted',
  `last_message_at` datetime DEFAULT NULL COMMENT '最后消息时间，列表排序',
  `message_count` int NOT NULL DEFAULT '0' COMMENT '消息条数冗余',
  `context_repo` varchar(64) NOT NULL DEFAULT '' COMMENT '可选绑定仓库',
  `context_image_tag` varchar(128) NOT NULL DEFAULT '' COMMENT '可选绑定镜像号',
  `context_index_version` varchar(64) NOT NULL DEFAULT '' COMMENT '可选绑定索引版本',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_chat_conv_status_last` (`status`,`last_message_at`)
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='对话会话';

-- ----------------------------
-- Table structure for `chat_message`
-- ----------------------------
DROP TABLE IF EXISTS `chat_message`;
CREATE TABLE `chat_message` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `conversation_id` bigint NOT NULL COMMENT 'chat_conversation.id',
  `role` varchar(32) NOT NULL COMMENT 'user/assistant/system',
  `content` mediumtext NOT NULL COMMENT '文本内容',
  `content_type` varchar(32) NOT NULL DEFAULT 'text' COMMENT 'text/mixed',
  `seq` int NOT NULL DEFAULT '0' COMMENT '会话内序号',
  `client_msg_id` varchar(64) NOT NULL DEFAULT '' COMMENT '前端幂等键',
  `status` varchar(32) NOT NULL DEFAULT 'normal' COMMENT 'normal/failed/deleted',
  `error_message` varchar(512) NOT NULL DEFAULT '' COMMENT '失败说明',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_chat_msg_conv_client` (`conversation_id`,`client_msg_id`),
  KEY `idx_chat_msg_conv_seq` (`conversation_id`,`seq`)
) ENGINE=InnoDB AUTO_INCREMENT=55 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='对话消息';

-- ----------------------------
-- Table structure for `chat_message_asset`
-- ----------------------------
DROP TABLE IF EXISTS `chat_message_asset`;
CREATE TABLE `chat_message_asset` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `message_id` bigint NOT NULL COMMENT 'chat_message.id',
  `asset_type` varchar(32) NOT NULL DEFAULT 'image' COMMENT '附件类型',
  `relative_path` varchar(512) NOT NULL COMMENT '相对 data/chat_uploads 路径',
  `mime_type` varchar(128) NOT NULL DEFAULT '' COMMENT 'MIME',
  `size_bytes` int NOT NULL DEFAULT '0',
  `width` int NOT NULL DEFAULT '0',
  `height` int NOT NULL DEFAULT '0',
  `sort_no` int NOT NULL DEFAULT '0' COMMENT '同消息多图顺序',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_chat_asset_msg_sort` (`message_id`,`sort_no`)
) ENGINE=InnoDB AUTO_INCREMENT=14 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='消息本地附件元数据';

-- ----------------------------
-- Table structure for `code_repo`
-- ----------------------------
DROP TABLE IF EXISTS `code_repo`;
CREATE TABLE `code_repo` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `repo_code` varchar(64) NOT NULL COMMENT '仓库编码',
  `repo_name` varchar(128) NOT NULL COMMENT '展示名',
  `local_root` varchar(512) NOT NULL COMMENT '本地根目录',
  `status` smallint NOT NULL COMMENT '1启用 0停用',
  `created_at` datetime NOT NULL DEFAULT (now()),
  `updated_at` datetime NOT NULL DEFAULT (now()),
  PRIMARY KEY (`id`),
  UNIQUE KEY `repo_code` (`repo_code`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Table structure for `code_index_version`
-- ----------------------------
DROP TABLE IF EXISTS `code_index_version`;
CREATE TABLE `code_index_version` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `repo_id` bigint NOT NULL COMMENT 'code_repo.id',
  `branch` varchar(128) NOT NULL COMMENT 'Git 分支',
  `image_tag` varchar(128) NOT NULL COMMENT '镜像号',
  `git_commit` varchar(40) NOT NULL COMMENT 'commit sha',
  `index_version` varchar(64) NOT NULL COMMENT '索引批次号',
  `embedding_model` varchar(128) NOT NULL COMMENT 'Embedding 模型',
  `embedding_dim` int NOT NULL COMMENT '向量维度',
  `milvus_collection` varchar(128) NOT NULL COMMENT 'Milvus collection',
  `status` varchar(32) NOT NULL COMMENT 'building/ready/deprecated/failed',
  `file_count` int NOT NULL,
  `chunk_count` int NOT NULL,
  `is_current` smallint NOT NULL COMMENT '是否当前问答版本',
  `remark` varchar(512) NOT NULL,
  `started_at` datetime DEFAULT NULL,
  `ready_at` datetime DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT (now()),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_repo_image_index` (`repo_id`,`image_tag`,`index_version`),
  KEY `ix_code_index_version_repo_id` (`repo_id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Table structure for `code_chunk_meta`
-- ----------------------------
DROP TABLE IF EXISTS `code_chunk_meta`;
CREATE TABLE `code_chunk_meta` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `index_version_id` bigint NOT NULL COMMENT 'code_index_version.id',
  `milvus_id` varchar(64) NOT NULL COMMENT '与完整 Milvus id 一致',
  `path` varchar(1024) NOT NULL COMMENT '相对仓库路径',
  `language` varchar(32) NOT NULL,
  `node_type` varchar(64) NOT NULL COMMENT 'Tree-sitter 节点类型',
  `symbol` varchar(512) NOT NULL COMMENT '类名/方法名',
  `start_line` int NOT NULL,
  `end_line` int NOT NULL,
  `content_hash` varchar(64) NOT NULL COMMENT 'text md5',
  `char_len` int NOT NULL,
  `created_at` datetime NOT NULL DEFAULT (now()),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_version_milvus_id` (`index_version_id`,`milvus_id`),
  KEY `idx_chunk_meta_version_path` (`index_version_id`,`path`(255)),
  KEY `ix_code_chunk_meta_index_version_id` (`index_version_id`)
) ENGINE=InnoDB AUTO_INCREMENT=19315 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SET FOREIGN_KEY_CHECKS = 1;
