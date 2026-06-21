// Headless smoke-test for VideoSource: opens a source, reads a few frames,
// prints their sizes. No GUI. Not part of the app build.
#include "ot/video_source.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: probe_decode <video> [height]\n"); return 2; }
    try {
        int h = (argc >= 3) ? std::atoi(argv[2]) : 0;
        ot::VideoSource src(argv[1], h);
        std::printf("native %dx%d -> working %dx%d @ %.2f fps\n",
                    src.native_width(), src.native_height(),
                    src.width(), src.height(), src.fps());
        cv::Mat f;
        int n = 0;
        while (n < 5 && src.read(f)) {
            std::printf("  frame %lld: %dx%d ch=%d\n",
                        (long long)src.frame_index(), f.cols, f.rows, f.channels());
            ++n;
        }
        std::printf("decoded %d frames OK\n", n);
        return n > 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
