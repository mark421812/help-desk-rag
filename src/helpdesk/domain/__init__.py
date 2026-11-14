"""Domain 型別：只用標準函式庫。

每個型別都是 frozen dataclass，在 `__post_init__` 檢查 invariant，建立得出來就代表合法。
狀態改變一律回傳新物件，呼叫端不必擔心共享的物件被別人改掉。
"""
