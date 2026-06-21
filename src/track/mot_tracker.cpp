#include "ot/mot_tracker.hpp"

#include <Eigen/Dense>
#include <motcpp/trackers/bytetrack.hpp>

namespace ot {

struct MotTracker::Impl {
    motcpp::trackers::ByteTrack tracker;

    explicit Impl(int frame_rate)
        : tracker(
              /*det_thresh   */ 0.1f,
              /*max_age      */ 30,
              /*max_obs      */ 50,
              /*min_hits     */ 2,
              /*iou_threshold*/ 0.3f,
              /*per_class    */ false,   // keep id across class flips (car<->truck)
              /*nr_classes   */ 80,
              /*asso_func    */ "iou",
              /*is_obb       */ false,
              /*min_conf     */ 0.1f,
              /*track_thresh */ 0.4f,
              /*match_thresh */ 0.8f,
              /*track_buffer */ 30,
              /*frame_rate   */ frame_rate) {}
};

MotTracker::MotTracker(int frame_rate) : impl_(std::make_unique<Impl>(frame_rate)) {}
MotTracker::~MotTracker() = default;
MotTracker::MotTracker(MotTracker&&) noexcept = default;
MotTracker& MotTracker::operator=(MotTracker&&) noexcept = default;

std::vector<Track> MotTracker::update(const cv::Mat& frame,
                                      const std::vector<Detection>& dets) {
    // Build the N×6 detection matrix [x1,y1,x2,y2,conf,cls].
    Eigen::MatrixXf in(static_cast<int>(dets.size()), 6);
    for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
        const BBox& b = dets[i].box;
        in(i, 0) = b.x;
        in(i, 1) = b.y;
        in(i, 2) = b.x + b.w;
        in(i, 3) = b.y + b.h;
        in(i, 4) = dets[i].score;
        in(i, 5) = static_cast<float>(dets[i].class_id);
    }

    const Eigen::MatrixXf out = impl_->tracker.update(in, frame);

    // Output rows are [x1,y1,x2,y2,id,conf,cls,det_idx].
    std::vector<Track> tracks;
    tracks.reserve(static_cast<size_t>(out.rows()));
    for (int i = 0; i < out.rows(); ++i) {
        Track t;
        t.box = {out(i, 0), out(i, 1), out(i, 2) - out(i, 0), out(i, 3) - out(i, 1)};
        t.id = static_cast<int>(out(i, 4));
        t.score = out(i, 5);
        t.class_id = static_cast<int>(out(i, 6));
        tracks.push_back(t);
    }
    return tracks;
}

}  // namespace ot
