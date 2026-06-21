#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace ot {

// Captures mouse clicks on a highgui window. Coordinates are mapped from the
// (possibly resized) window back to the displayed image's pixel space, so they
// match the working-resolution frame the detector/tracker operate on.
class Selector {
public:
    // Registers a mouse callback on `window`. `image_size` is the displayed
    // frame size (working resolution).
    Selector(const std::string& window, cv::Size image_size);

    // If a click occurred since the last poll, writes it to `pt` and returns true.
    bool poll(cv::Point2f& pt);

private:
    static void on_mouse(int event, int x, int y, int flags, void* userdata);
    cv::Size    image_size_;
    bool        has_click_ = false;
    cv::Point2f click_{0, 0};
};

}  // namespace ot
