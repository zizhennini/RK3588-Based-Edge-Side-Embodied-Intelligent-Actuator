#include <libobsensor/hpp/Pipeline.hpp>
#include <libobsensor/hpp/Frame.hpp>
#include <libobsensor/hpp/Error.hpp>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) try {
    ob::Pipeline pipe;
    auto config = std::make_shared<ob::Config>();
    config->enableVideoStream(OB_STREAM_DEPTH);
    pipe.start(config);

    auto frameset = pipe.waitForFrames(5000);
    if (!frameset) { std::cerr << "timeout" << std::endl; return 1; }

    auto depthFrame = frameset->depthFrame();
    if (!depthFrame) { std::cerr << "no depth frame" << std::endl; return 1; }

    int w = depthFrame->width(), h = depthFrame->height();
    float scale = depthFrame->getValueScale();
    auto data = (uint16_t *)depthFrame->data();

    // 保存深度图: float32, 单位米
    {
        std::ofstream f("_depth.f32", std::ios::binary);
        for (int i = 0; i < w * h; i++) {
            float z = data[i] * scale / 1000.0f;
            f.write((const char *)&z, sizeof(float));
        }
    }

    // 保存宽高信息
    {
        std::ofstream f("_depth.info");
        f << w << " " << h << std::endl;
    }

    std::cout << "depth: " << w << "x" << h << std::endl;

    pipe.stop();
    return 0;
} catch (ob::Error &e) {
    std::cerr << "Error: " << e.getMessage() << std::endl;
    return 1;
}
