# VLM 引擎测试
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vla.vlm import VLMBase, VLMResult


class FakeVLM(VLMBase):
    def load(self, model_path):
        pass
    def infer(self, image_path, prompt=None):
        return VLMResult(color="红色", object="方块", raw='{"color":"红色","object":"方块"}')
    def unload(self):
        pass


def test_vlm_result():
    vlm = FakeVLM()
    vlm.load("")
    result = vlm.infer("")
    assert result.color == "红色"
    assert result.object == "方块"
    print("VLM 框架测试通过")


if __name__ == "__main__":
    test_vlm_result()
