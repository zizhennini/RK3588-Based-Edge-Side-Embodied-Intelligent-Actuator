# tests — 单元测试

| 文件 | 测试内容 |
|------|---------|
| `test_locator.py` | ColorLocator: 红色定位、无目标过滤、小目标过滤 |
| `test_ik.py` | ArmController._ik(): 6 关节角输出、角度范围 |
| `test_vlm.py` | VLM 框架接口: 模拟引擎 + 结果解析 |

## 运行

```bash
python tests/test_ik.py
python tests/test_locator.py
python tests/test_vlm.py
```
