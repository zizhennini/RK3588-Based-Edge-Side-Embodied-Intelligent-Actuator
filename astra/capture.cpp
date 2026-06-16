#include <libobsensor/hpp/Pipeline.hpp>
#include <libobsensor/hpp/Frame.hpp>
#include <libobsensor/hpp/Error.hpp>
#include <fstream>
#include <iostream>
#include <signal.h>

volatile sig_atomic_t running = 1;
void handle_signal(int) { running = 0; }

int main() try {
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    ob::Pipeline pipe;
    auto config = std::make_shared<ob::Config>();
    config->enableVideoStream(OB_STREAM_DEPTH);
    pipe.start(config);

    while (running) {
        auto frameset = pipe.waitForFrames(1000);
        if (!frameset) continue;
        auto depthFrame = frameset->depthFrame();
        if (!depthFrame) continue;

        int w = depthFrame->width(), h = depthFrame->height();
        float scale = depthFrame->getValueScale();
        auto data = (uint16_t *)depthFrame->data();

        std::ofstream f("_depth.f32", std::ios::binary);
        for (int i = 0; i < w * h; i++) {
            float z = data[i] * scale / 1000.0f;
            f.write((const char *)&z, sizeof(float));
        }
        f.close();

        std::ofstream info("_depth.info");
        info << w << " " << h << std::endl;
    }

    pipe.stop();
    return 0;
} catch (ob::Error &e) {
    std::cerr << "Error: " << e.getMessage() << std::endl;
    return 1;
}
