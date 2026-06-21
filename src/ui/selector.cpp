#include "ot/selector.hpp"

#include <opencv2/highgui.hpp>

#include <algorithm>

namespace ot {

Selector::Selector(const std::string& window, cv::Size image_size)
    : image_size_(image_size) {
    cv::setMouseCallback(window, &Selector::on_mouse, this);
}

void Selector::on_mouse(int event, int x, int y, int /*flags*/, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* self = static_cast<Selector*>(userdata);
    // highgui reports coordinates in the displayed image's pixel space (it
    // accounts for window resizing), so they map directly to the working frame.
    self->click_.x = std::clamp(static_cast<float>(x), 0.0f,
                                static_cast<float>(self->image_size_.width - 1));
    self->click_.y = std::clamp(static_cast<float>(y), 0.0f,
                                static_cast<float>(self->image_size_.height - 1));
    self->has_click_ = true;
}

bool Selector::poll(cv::Point2f& pt) {
    if (!has_click_) return false;
    pt = click_;
    has_click_ = false;
    return true;
}

}  // namespace ot
