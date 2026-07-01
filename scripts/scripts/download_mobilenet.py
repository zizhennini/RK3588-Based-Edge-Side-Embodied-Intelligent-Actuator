"""模型下载脚本 — 下载 MobileNet SSD 预训练模型"""
import os
import sys


def download_mobilenet_ssd(target_dir: str = "./models/MobileNetSSD"):
    import urllib.request

    os.makedirs(target_dir, exist_ok=True)

    prototxt_urls = [
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    ]
    caffemodel_urls = [
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
    ]

    prototxt_path = os.path.join(target_dir, "MobileNetSSD_deploy.prototxt")
    caffemodel_path = os.path.join(target_dir, "MobileNetSSD_deploy.caffemodel")

    def _download(urls, path, desc):
        import socket
        socket.setdefaulttimeout(120)
        for url in urls:
            try:
                print(f"下载 {desc} ...")
                urllib.request.urlretrieve(url, path)
                print(f"  ✓ {path}")
                return True
            except Exception as e:
                print(f"  ✗ 失败: {e}")
                continue
        return False

    if not os.path.exists(prototxt_path):
        _download(prototxt_urls, prototxt_path, "prototxt")
    else:
        print(f"  ✓ {prototxt_path} (已存在)")

    if not os.path.exists(caffemodel_path):
        ok = _download(caffemodel_urls, caffemodel_path, "caffemodel (23 MB)")
        if not ok:
            print()
            print("手动下载方法：")
            print("  wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel")
            print(f"  mv mobilenet_iter_73000.caffemodel {caffemodel_path}")
            print()
            print("或使用代理：")
            print("  wget -e use_proxy=on -e http_proxy=http://127.0.0.1:7890 \\")
            print("       https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel")
            print(f"  mv mobilenet_iter_73000.caffemodel {caffemodel_path}")
    else:
        print(f"  ✓ {caffemodel_path} (已存在)")


if __name__ == "__main__":
    download_mobilenet_ssd()
