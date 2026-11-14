# Help Desk RAG

Help Desk RAG 是一個知識庫問答系統。把文件放進 `documents/` 資料夾，系統把文件切塊並建立索引。使用者在網頁上提問，系統找出相關段落，交給 LLM 模型產生回答。

## 兩個程序

整個系統是一個 Python 套件 `helpdesk`，跑成兩個程序。


| 程序               | 做什麼                                                                        |
| ------------------ | ----------------------------------------------------------------------------- |
| `helpdesk-api`     | 提供網頁、對話 API、檢索端點`POST /api/v1/retrieve`，以及回答的產生與引用驗證 |
| `ingestion-worker` | 每天定時掃描`documents/`，解析、切塊、算 E5 向量，寫進 PostgreSQL 與 Qdrant   |

兩個程序共用一個 PostgreSQL 與一個 Qdrant。PostgreSQL 存文件、chunk 全文、工作佇列與對話，資料表都在 `helpdesk` schema。Qdrant 存 chunk 的向量，collection 是 `helpdesk_chunks_v1`。

## 啟動系統

需要能執行 Linux 容器的 Docker，例如 Docker Desktop、OrbStack 或 Colima，以及 Docker Compose 2.24 以上。指令一律是 `docker compose`。

1. 複製設定範例。

   ```bash
   cp .env.example .env
   ```
2. 在 `.env` 填入 `GROQ_API_KEY`。
3. 在 `.env` 填入 `POSTGRES_PASSWORD` 與 `QDRANT_API_KEY`。兩個都用長的隨機值，例如 `openssl rand -hex 32` 的輸出。
4. 把 `DATABASE_URL` 裡的密碼換成 `POSTGRES_PASSWORD` 的值。帳號與資料庫名稱也要和 `POSTGRES_USER`、`POSTGRES_DB` 相同。
5. 啟動所有容器。

   ```bash
   docker compose up -d
   ```
6. 確認 `helpdesk-api` 已經就緒。第一次啟動要下載 E5 模型，大約需要一到兩分鐘。

   ```bash
   curl http://127.0.0.1:8140/health/ready
   ```

   回應是 200，而且 `postgres`、`qdrant`、`embedding_model` 都是 `true`。
7. 用瀏覽器開啟 `http://127.0.0.1:8140/`。

`docker compose up -d` 啟動以下服務。這個 compose 專案叫 `helpdesk`，網路是自己的 `helpdesk_default`。


| 服務               | 說明                                                                      |
| ------------------ | ------------------------------------------------------------------------- |
| `postgres`         | PostgreSQL 16，資料在 volume`helpdesk_postgres-data`                      |
| `qdrant`           | Qdrant 1.18.3，資料在 volume`helpdesk_qdrant-data`，需要 `QDRANT_API_KEY` |
| `migrate`          | 執行`alembic upgrade head` 後結束                                         |
| `helpdesk-api`     | 只開在`127.0.0.1:8140`                                                    |
| `ingestion-worker` | 以唯讀方式掛載`./documents`                                               |

`postgres` 與 `qdrant` 不對主機開 port。要查資料庫時，進容器執行 `psql`。

```bash
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

`helpdesk-api` 與 `ingestion-worker` 用 http 連 Qdrant，所以 qdrant-client 啟動時會印出 `Api key is used with an insecure connection.`。Qdrant 只在 `helpdesk_default` 網路上，金鑰不會離開這台電腦的 Docker 網路，所以這個警告可以忽略。若要讓 Qdrant 對外，先替它設定 TLS。

## 建立文件索引

### 放入文件

把文件放進根目錄的 `documents/`，可以有子資料夾。支援 PDF、DOCX、Markdown 與純文字檔。`documents/` 內的檔案不進 git。

`ingestion-worker` 每天在 `SCHEDULE_HOUR` 與 `SCHEDULE_MINUTE` 指定的時間掃描一次，時區是 `SCHEDULE_TIMEZONE`。預設是台北時間凌晨 2 點。如果啟動時今天的排程時間已過，而最後一次完整掃描早於這個時間，啟動後會先補掃一次。單檔 `ingest` 不算完整掃描。

檔案沒有變就不重做。內容或 metadata 改了，下一次掃描會建立新版本，舊版本的向量會刪掉。檔案要連續兩次掃描都不見才會從索引刪除。一次掃描不見超過 25% 的文件，或資料夾變成空的，整次掃描會被拒絕，避免掛載失敗時誤刪。

### 加上 metadata（選填）

metadata 是 YAML 檔，放在文件旁邊。

- `<檔名>.metadata.yaml` 只套用到一個檔案，例如 `Resume.pdf.metadata.yaml`。
- 資料夾裡的 `.ingestion.yaml` 套用到資料夾內所有檔案。越靠近文件的設定優先。

```yaml
title: 請假管理辦法
document_type: policy
owner_department: HR
document_owner: hr@example.com
status: published
confidentiality: internal
language: zh-TW
external_llm_allowed: true
effective_at: 2026-01-01
expires_at: 2027-01-01
acl:
  - principal_type: group
    principal_id: staff
  - principal_type: user
    principal_id: intern@example.com
    effect: deny
```

ACL 關閉時每個欄位都可以不寫。沒有 `title` 時用檔名當標題。

### 手動操作 worker

`helpdesk-worker` 指令可以在容器裡手動執行。輸出是 JSON。

```bash
docker compose run --rm ingestion-worker helpdesk-worker scan
docker compose run --rm ingestion-worker helpdesk-worker ingest 'BookRAG_繁體中文完整翻譯.pdf'
docker compose run --rm ingestion-worker helpdesk-worker run --until-idle
docker compose run --rm ingestion-worker helpdesk-worker items --status quarantined
docker compose run --rm ingestion-worker helpdesk-worker retry <job_id>
```


| 指令               | 做什麼                                                                         |
| ------------------ | ------------------------------------------------------------------------------ |
| `scan`             | 完整掃描`documents/`，替變更的檔案排 job。`--allow-mass-deletion` 略過刪除護欄 |
| `ingest <路徑>`    | 只替一個檔案排 job，其他檔案不會被當成消失                                     |
| `run --until-idle` | 處理佇列裡的 job，處理完就結束                                                 |
| `run`              | 長駐模式，也就是容器預設的指令。處理 job，並執行每日掃描                       |
| `items`            | 列出每個檔案的狀態與最近一次 job，可用`--status`、`--offset`、`--limit`        |
| `retry <job_id>`   | 重新排一個失敗的 job                                                           |

`scan`、`ingest`、`items`、`retry` 只排 job 或查詢。實際的解析與向量計算由 `run` 執行。

## 設定兩個開關

兩個開關在 `.env` 設定，程式預設都是 `true`。改完後重建容器才會生效。

```bash
docker compose up -d --force-recreate helpdesk-api ingestion-worker
```

### `ACCESS_CONTROL_ENABLED`


| 值      | 檢索                                                                                                                          | 文件匯入                                                                                                           | 使用者身分                                    |
| ------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| `false` | 不比對 ACL、文件狀態、生效日與到期日。所有文件都可以送到 Groq                                                                 | metadata 選填                                                                                                      | 每個瀏覽器用匿名 ID                           |
| `true`  | Qdrant 先依 ACL 過濾，再用 PostgreSQL 的最新 metadata 複驗。拒絕優先於允許。`external_llm_allowed: false` 的文件不會送到 Groq | `title`、`document_type`、`owner_department`、`document_owner` 必填。缺少時檔案狀態是 `metadata_missing`，不進索引 | 讀`x-dev-user-id` 與 `x-dev-group-ids` header |

兩種模式都會保存 ACL 資料。切換開關後，下一次掃描會重新判斷每份文件，不需要手動重建索引。沒有任何允許規則的文件視為公開。

### `SENSITIVE_DATA_SCAN_ENABLED`


| 值      | 行為                                                                                                                                                                      |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `false` | 不掃描，直接送到 Groq                                                                                                                                                     |
| `true`  | 送到 Groq 之前，掃描問題、改寫用的歷史與證據。規則包含金鑰、身分證字號、信用卡號、銀行帳號與電子郵件。命中任何一條就擋下，不呼叫模型，回合結果是`outbound_policy_blocked` |

規則在 `src/helpdesk/infrastructure/outbound_scan/policy.yaml`。

## 執行測試

測試需要 Python 3.11 以上與 uv。整合測試另外需要 Docker，會用 testcontainers 啟動暫時的 PostgreSQL 與 Qdrant。

```bash
uv sync --frozen --extra dev
uv run --frozen python -m ruff check .
uv run --frozen python -m ruff format --check .
uv run --frozen python -m mypy src tests
uv run --frozen python -m pytest
uv run --frozen python -m pytest -m integration
```

主機上不安裝 torch，單元與整合測試用固定的雜湊向量代替 E5。真正的 E5 只在 Docker 映像裡執行。

## 已知限制

- 加密的 PDF 一律拒絕，檔案狀態是 `quarantined`，錯誤代碼 `invalid_document`。掃描檔沒有文字層時狀態是 `ocr_required`，系統不做 OCR。
- ACL 開啟時還沒有正式登入。瀏覽器不會送 `x-dev-user-id`，所以網頁在 ACL 開啟時無法使用，只能用帶 header 的 API 呼叫測試。`APP_ENV=production` 加上 ACL 開啟時 `helpdesk-api` 拒絕啟動。
- retrivel 沒有重排序。
