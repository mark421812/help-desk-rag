"""Help Desk RAG。

`helpdesk-api` 與 `ingestion-worker` 兩個程序共用這個套件。分層由內到外是 `domain`、
`application`、`infrastructure`、`presentation`；內層不認識外層，`tests/unit/test_architecture.py`
守住這條規則。
"""
