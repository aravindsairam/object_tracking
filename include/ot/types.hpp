#pragma once

#include <opencv2/core.hpp>
#include <algorithm>
#include <vector>

namespace ot {

// Axis-aligned box in pixel coordinates of the current working frame.
struct BBox {
    float x = 0, y = 0, w = 0, h = 0;  // top-left + size

    cv::Point2f center() const { return {x + w * 0.5f, y + h * 0.5f}; }
    float area() const { return w * h; }
    cv::Rect to_rect() const {
        return cv::Rect(cvRound(x), cvRound(y), cvRound(w), cvRound(h));
    }

    // Clamp the box to lie within [0,W]x[0,H].
    void clamp(int W, int H) {
        float x2 = std::min(static_cast<float>(W), x + w);
        float y2 = std::min(static_cast<float>(H), y + h);
        x = std::max(0.0f, x);
        y = std::max(0.0f, y);
        w = std::max(0.0f, x2 - x);
        h = std::max(0.0f, y2 - y);
    }
};

// One detection from a Detector: a box, its normalized class id, and score.
struct Detection {
    BBox box;
    int   class_id = -1;   // model-native class id
    float score = 0.0f;
};

// A tracked object: a detection box with a persistent identity across frames.
struct Track {
    BBox  box;
    int   id = -1;        // persistent track id
    float score = 0.0f;
    int   class_id = -1;
};

// L2-normalized appearance feature (ReID), used from M5 onward.
using Embedding = std::vector<float>;

// Lifecycle of the locked target (used from M3 onward).
enum class LockState { Acquiring, Locked, Coasting, Lost };

}  // namespace ot
