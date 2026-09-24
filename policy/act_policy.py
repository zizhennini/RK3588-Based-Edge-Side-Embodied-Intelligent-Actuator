"""ACT 策略模型板端推理 -- ONNX Runtime CPU

加载分模块 ONNX (vision_encoder + transformer)，
输出 action chunk (100步动作序列)，按步执行。

预估性能 (RK3588 A76):
  - 推理延迟: 70-120ms
  - 内存占用: ~350-450MB
  - 控制频率: 1-5Hz (chunk 内插值可达 20-50Hz)

推理流程:
  Step 1: vision_encoder.onnx
    image (1, 3, 480, 640) float32 [0,1]
    → vision_features (1, 512, 15, 20)

  Step 2: transformer.onnx
    vision_features + state (1, 6) + query_embed (100, 512)
    → actions (1, 100, 6)
"""
import time
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from hardware.interfaces import PolicyModule, Observation, Action

logger = logging.getLogger(__name__)


class ACTPolicy(PolicyModule):
    """ACT 策略推理 -- ONNX Runtime CPU，Action Chunking

    每次 predict() 返回单步动作。内部维护 action buffer：
      - buffer 非空：弹出下一步动作
      - buffer 空：执行完整推理，填充 chunk_size 步，弹出第一步
    """

    def __init__(self) -> None:
        self._vision_session: Optional[object] = None
        self._transformer_session: Optional[object] = None
        self._config: dict = {}
        self._query_embed: Optional[np.ndarray] = None  # (chunk_size, 512)
        self._norm_stats: dict = {}
        self._action_buffer: list = []
        self._buffer_lock = threading.Lock()
        self._chunk_size: int = 100
        self._n_action_steps: int = 100
        self._action_dim: int = 6
        self._dim_model: int = 512
        self._last_infer_ms: float = 0.0
        self._model_dir: Optional[Path] = None
        self._setup_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Module 生命周期
    # ------------------------------------------------------------------ #
    def setup(self, config: dict) -> None:
        """加载 ONNX 模型和配置。

        config keys:
            model_dir: str       # 模型目录 (含 vision_encoder.onnx / transformer.onnx / act_config.json)
            n_action_steps: int  # 每次推理后执行的步数 (默认 100)
            intra_op_threads: int# ONNX Runtime 线程数 (默认 2, 建议绑 A76 核 4-5)
            inter_op_threads: int# ONNX Runtime 跨算子线程 (默认 1)
            chunk_size: int      # 覆盖 act_config 中的 chunk_size (可选)
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            self._setup_error = f"onnxruntime 未安装: {e}"
            logger.error(self._setup_error)
            return

        model_dir = Path(config.get("model_dir", "models/act"))
        self._model_dir = model_dir

        # --- 加载 act_config.json ---
        config_path = model_dir / "act_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.info(f"ACT 配置已加载: {config_path}")
            except Exception as e:
                logger.warning(f"act_config.json 解析失败 ({e})，使用默认配置")
                self._config = {}
        else:
            logger.warning(f"未找到 {config_path}，使用默认配置")
            self._config = {}

        # --- 超参 ---
        self._chunk_size = int(config.get(
            "chunk_size", self._config.get("chunk_size", 100)))
        self._n_action_steps = int(config.get(
            "n_action_steps", self._config.get("n_action_steps", self._chunk_size)))
        self._n_action_steps = max(1, min(self._n_action_steps, self._chunk_size))
        self._action_dim = int(self._config.get("action_dim", 6))
        self._dim_model = int(self._config.get("dim_model", 512))

        # --- ONNX Runtime session options ---
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = int(config.get("intra_op_threads", 2))
        sess_opts.inter_op_num_threads = int(config.get("inter_op_threads", 1))
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.log_severity_level = 2  # 只输出 warning/error

        providers = ["CPUExecutionProvider"]

        # --- 加载 vision encoder ---
        vision_path = model_dir / self._config.get(
            "vision_encoder_path", "vision_encoder.onnx")
        if not vision_path.exists():
            self._setup_error = f"vision encoder 不存在: {vision_path}"
            logger.error(self._setup_error)
            return
        try:
            t0 = time.perf_counter()
            self._vision_session = ort.InferenceSession(
                str(vision_path), sess_options=sess_opts, providers=providers)
            logger.info(
                f"vision_encoder 加载耗时 {(time.perf_counter() - t0) * 1000:.0f}ms"
            )
        except Exception as e:
            self._setup_error = f"vision encoder 加载失败: {e}"
            logger.exception(self._setup_error)
            self._vision_session = None
            return

        # --- 加载 transformer ---
        transformer_path = model_dir / self._config.get(
            "transformer_path", "transformer.onnx")
        if not transformer_path.exists():
            self._setup_error = f"transformer 不存在: {transformer_path}"
            logger.error(self._setup_error)
            self._vision_session = None
            return
        try:
            t0 = time.perf_counter()
            self._transformer_session = ort.InferenceSession(
                str(transformer_path), sess_options=sess_opts, providers=providers)
            logger.info(
                f"transformer 加载耗时 {(time.perf_counter() - t0) * 1000:.0f}ms"
            )
        except Exception as e:
            self._setup_error = f"transformer 加载失败: {e}"
            logger.exception(self._setup_error)
            self._transformer_session = None
            self._vision_session = None
            return

        # --- query embedding (固定 learned queries) ---
        query_path = model_dir / self._config.get(
            "query_embed_path", "query_embed.npy")
        if query_path.exists():
            try:
                self._query_embed = np.load(str(query_path)).astype(np.float32)
                if self._query_embed.ndim == 2 and \
                        self._query_embed.shape != (self._chunk_size, self._dim_model):
                    logger.warning(
                        f"query_embed 形状 {self._query_embed.shape} 与预期 "
                        f"({self._chunk_size}, {self._dim_model}) 不一致，将按实际形状使用"
                    )
                    self._chunk_size = self._query_embed.shape[0]
                    self._dim_model = self._query_embed.shape[1]
                    self._n_action_steps = min(self._n_action_steps, self._chunk_size)
            except Exception as e:
                logger.warning(f"query_embed.npy 加载失败 ({e})，使用零初始化")
                self._query_embed = np.zeros(
                    (self._chunk_size, self._dim_model), dtype=np.float32)
        else:
            logger.warning(
                f"未找到 {query_path}，使用零初始化 query_embed "
                "(实际部署应从 checkpoint 提取 learned queries)"
            )
            self._query_embed = np.zeros(
                (self._chunk_size, self._dim_model), dtype=np.float32)

        # --- 归一化统计量 ---
        self._norm_stats = self._config.get("norm_stats", {}) or {}

        # --- 记录 session I/O 信息，便于调试 ---
        try:
            v_in = [(i.name, i.shape, i.type) for i in self._vision_session.get_inputs()]
            v_out = [(o.name, o.shape, o.type) for o in self._vision_session.get_outputs()]
            t_in = [(i.name, i.shape, i.type) for i in self._transformer_session.get_inputs()]
            t_out = [(o.name, o.shape, o.type) for o in self._transformer_session.get_outputs()]
            logger.debug(f"vision_encoder  inputs={v_in} outputs={v_out}")
            logger.debug(f"transformer    inputs={t_in} outputs={t_out}")
        except Exception:
            pass

        self._setup_error = None
        logger.info(
            f"ACT Policy 已加载: {model_dir} "
            f"(chunk_size={self._chunk_size}, n_action_steps={self._n_action_steps}, "
            f"intra_op_threads={sess_opts.intra_op_num_threads})"
        )

    def start(self) -> None:
        """无后台线程；action chunk 推理在 predict() 内按需触发。"""
        if self.is_available:
            logger.info("ACTPolicy 已就绪")
        else:
            logger.warning(f"ACTPolicy 未就绪: {self._setup_error or '模型未加载'}")

    def stop(self) -> None:
        with self._buffer_lock:
            self._action_buffer = []
        self._vision_session = None
        self._transformer_session = None
        self._query_embed = None
        logger.info("ACTPolicy 已停止并释放 session")

    @property
    def is_available(self) -> bool:
        return (self._vision_session is not None
                and self._transformer_session is not None
                and self._query_embed is not None)

    def on_failure(self) -> str:
        """ACT 不可用时降级到 GGCNN 抓取管线。"""
        return "fallback"

    # ------------------------------------------------------------------ #
    # PolicyModule 接口
    # ------------------------------------------------------------------ #
    def predict(self, obs: Observation) -> Action:
        """返回单步动作。buffer 空时触发完整推理填充 chunk。"""
        if not self.is_available:
            raise RuntimeError(
                f"ACTPolicy 不可用: {self._setup_error or '模型未加载'}")

        with self._buffer_lock:
            if not self._action_buffer:
                actions = self._infer_chunk(obs)  # (chunk_size, action_dim)
                n = min(self._n_action_steps, actions.shape[0])
                self._action_buffer = [actions[i] for i in range(n)]
                logger.debug(
                    f"ACT chunk 已填充: {n} 步 (推理 {self._last_infer_ms:.1f}ms)")

            if self._action_buffer:
                action_array = self._action_buffer.pop(0)
            else:
                # 极端异常：推理返回空 chunk
                logger.warning("ACT 推理返回空 chunk，保持当前状态")
                return Action(
                    positions=np.asarray(obs.state, dtype=np.float32).copy(),
                    gripper=0.0,
                    execution_time=0.0,
                )

        # 夹爪开合度：ACT 输出第 6 维即夹爪 (rad 或 [0,1]，取决于训练数据)
        positions = np.asarray(action_array[:self._action_dim], dtype=np.float32)
        gripper_raw = float(action_array[self._action_dim - 1]) \
            if len(action_array) >= self._action_dim else 0.0
        gripper = self._map_gripper(gripper_raw)

        return Action(
            positions=positions,
            gripper=gripper,
            execution_time=0.02,  # 50Hz 控制频率
        )

    # ------------------------------------------------------------------ #
    # 推理内部实现
    # ------------------------------------------------------------------ #
    def _infer_chunk(self, obs: Observation) -> np.ndarray:
        """完整推理: image + state → (chunk_size, action_dim) 反归一化动作序列。"""
        t0 = time.perf_counter()

        # 1. 图像预处理: (H,W,3) uint8 → (1,3,H,W) float32 [0,1]
        image = self._preprocess_image(obs.rgb)

        # 2. 状态归一化 → (1, action_dim)
        state_vec = self._normalize_state(obs.state)
        state = np.expand_dims(state_vec.astype(np.float32), 0)

        # 3. Vision Encoder 推理
        try:
            vision_input_name = self._vision_session.get_inputs()[0].name
            vision_features = self._vision_session.run(
                None, {vision_input_name: image})[0]
        except Exception as e:
            logger.error(f"vision_encoder 推理失败: {e}")
            raise

        # 4. Transformer 推理
        try:
            inputs = self._build_transformer_inputs(vision_features, state)
            actions = self._transformer_session.run(None, inputs)[0]
        except Exception as e:
            logger.error(f"transformer 推理失败: {e}")
            raise

        # (1, chunk, dim) → (chunk, dim)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions.squeeze(0)
        elif actions.ndim != 2:
            raise RuntimeError(f"transformer 输出维度异常: {actions.shape}")

        # 5. 反归一化
        actions = self._denormalize_actions(actions)

        # NaN / Inf 保护
        if not np.isfinite(actions).all():
            bad = int((~np.isfinite(actions)).sum())
            logger.warning(f"ACT 输出包含 {bad} 个非有限值，已替换为当前状态")
            fallback = np.tile(
                np.asarray(obs.state, dtype=np.float32)[:self._action_dim],
                (actions.shape[0], 1))
            actions = np.where(np.isfinite(actions), actions, fallback)

        elapsed = (time.perf_counter() - t0) * 1000.0
        self._last_infer_ms = elapsed
        if elapsed > 150.0:
            logger.warning(f"ACT 推理延迟过高: {elapsed:.1f}ms (预期 70-120ms)")
        else:
            logger.debug(f"ACT 推理: {elapsed:.1f}ms, 输出 {actions.shape}")

        return actions

    def _preprocess_image(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 → (1, 3, H, W) float32 [0,1]，可选 resize。"""
        if rgb is None:
            raise ValueError("Observation.rgb 为 None")

        img = rgb
        # 目标尺寸（可配置，默认训练分辨率 480x640）
        target_hw = tuple(self._config.get("image_size", [480, 640]))
        if img.shape[:2] != target_hw:
            img = self._resize_image(img, target_hw)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[..., :3]

        img = img.astype(np.float32) / 255.0
        # 可选 ImageNet 归一化（训练使用 torchvision 时常见）
        mean = self._config.get("image_mean")
        std = self._config.get("image_std")
        if mean is not None and std is not None:
            mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
            std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
            std_arr = np.where(np.abs(std_arr) < 1e-6, 1.0, std_arr)
            img = (img - mean_arr) / std_arr

        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC → CHW
        return np.expand_dims(img, 0).astype(np.float32)    # → (1,3,H,W)

    @staticmethod
    def _resize_image(img: np.ndarray, target_hw: tuple) -> np.ndarray:
        """优先使用 cv2；不可用时退化为最近邻。"""
        h, w = int(target_hw[0]), int(target_hw[1])
        try:
            import cv2  # type: ignore
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass
        # 最近邻回退（无额外依赖）
        src_h, src_w = img.shape[:2]
        row_idx = (np.arange(h) * src_h / max(h, 1)).astype(np.int64)
        col_idx = (np.arange(w) * src_w / max(w, 1)).astype(np.int64)
        row_idx = np.clip(row_idx, 0, src_h - 1)
        col_idx = np.clip(col_idx, 0, src_w - 1)
        return img[np.ix_(row_idx, col_idx)]

    def _build_transformer_inputs(self, vision_features: np.ndarray,
                                  state: np.ndarray) -> dict:
        """按输入名语义匹配填充 transformer 输入字典。"""
        query = np.asarray(self._query_embed, dtype=np.float32)
        inputs: dict = {}
        matched = {"vision": False, "state": False, "query": False}

        for inp in self._transformer_session.get_inputs():
            name = inp.name.lower()
            value = None
            if any(k in name for k in ("vision", "feature", "image_feat", "cnn")):
                value, matched["vision"] = vision_features, True
            elif "state" in name or "qpos" in name or "proprio" in name:
                value, matched["state"] = state, True
            elif "query" in name or "embed" in name:
                value, matched["query"] = query, True

            if value is None:
                raise RuntimeError(
                    f"transformer 输入 '{inp.name}' 无法识别，"
                    f"请检查 ONNX 导出命名 (期望包含 vision/state/query 关键字)")
            inputs[inp.name] = value.astype(np.float32, copy=False)

        missing = [k for k, v in matched.items() if not v]
        if missing:
            logger.warning(f"transformer 输入未匹配: {missing}（已填充: {list(inputs)}）")

        return inputs

    # ------------------------------------------------------------------ #
    # 归一化 / 反归一化
    # ------------------------------------------------------------------ #
    def _get_norm_arrays(self, key: str) -> tuple:
        stats = self._norm_stats.get(key, {}) or {}
        mean = np.asarray(
            stats.get("mean", [0.0] * self._action_dim), dtype=np.float32)
        std = np.asarray(
            stats.get("std", [1.0] * self._action_dim), dtype=np.float32)
        if mean.size == 1:
            mean = np.full(self._action_dim, float(mean), dtype=np.float32)
        if std.size == 1:
            std = np.full(self._action_dim, float(std), dtype=np.float32)
        if mean.size < self._action_dim:
            mean = np.pad(mean, (0, self._action_dim - mean.size))
        if std.size < self._action_dim:
            std = np.pad(std, (0, self._action_dim - std.size), constant_values=1.0)
        mean = mean[:self._action_dim]
        std = std[:self._action_dim]
        std = np.where(np.abs(std) < 1e-6, 1.0, std)
        return mean, std

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """状态归一化 (减均值除标准差)。"""
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.size < self._action_dim:
            s = np.pad(s, (0, self._action_dim - s.size))
        s = s[:self._action_dim]
        mean, std = self._get_norm_arrays("state")
        return (s - mean) / std

    def _denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """动作反归一化。"""
        mean, std = self._get_norm_arrays("action")
        return actions * std + mean

    @staticmethod
    def _map_gripper(value: float) -> float:
        """夹爪值 → [0=全闭, 1=全开]。超出 [0,1] 时按弧度线性映射后裁剪。"""
        if not np.isfinite(value):
            return 0.0
        v = float(value)
        if v > 1.0:
            # 训练数据夹爪单位可能为 rad (0 ~ ~0.9)，直接裁剪到 [0,1]
            v = min(v, 1.0)
        elif v < 0.0:
            v = 0.0
        return v

    # ------------------------------------------------------------------ #
    # 外部辅助 API
    # ------------------------------------------------------------------ #
    def reset_buffer(self) -> None:
        """清空动作缓冲（任务切换 / 中断时调用）。"""
        with self._buffer_lock:
            n = len(self._action_buffer)
            self._action_buffer = []
        if n:
            logger.info(f"ACT action buffer 已清空 (丢弃 {n} 步)")

    @property
    def buffer_remaining(self) -> int:
        """剩余缓冲动作数。"""
        with self._buffer_lock:
            return len(self._action_buffer)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def n_action_steps(self) -> int:
        return self._n_action_steps

    @property
    def last_infer_ms(self) -> float:
        """最近一次完整推理耗时 (ms)。"""
        return self._last_infer_ms

    @property
    def setup_error(self) -> Optional[str]:
        return self._setup_error
