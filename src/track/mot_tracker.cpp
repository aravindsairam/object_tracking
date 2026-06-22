#include "ot/mot_tracker.hpp"

#include "ot/reid.hpp"

#include <Eigen/Dense>
#include <motcpp/tracker.hpp>
#include <motcpp/trackers/bytetrack.hpp>
#include <motcpp/trackers/botsort.hpp>
#include <motcpp/trackers/ocsort.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ot {

struct MotTracker::Impl {
    std::unique_ptr<motcpp::BaseTracker> tracker;
    std::shared_ptr<ReidEmbedder> reid;
    bool feed_embeddings = false;   // BoT-SORT + with_reid + a non-null embedder

    Impl(const TrackerCfg& cfg, int frame_rate, std::shared_ptr<ReidEmbedder> r)
        : reid(std::move(r)) {
        if (cfg.type == "bytetrack") {
            tracker = std::make_unique<motcpp::trackers::ByteTrack>(
                /*det_thresh   */ 0.1f,
                /*max_age      */ 30,
                /*max_obs      */ 50,
                /*min_hits     */ 2,
                /*iou_threshold*/ cfg.iou_threshold,
                /*per_class    */ false,   // keep id across class flips (car<->truck)
                /*nr_classes   */ 80,
                /*asso_func    */ "iou",
                /*is_obb       */ false,
                /*min_conf     */ cfg.min_conf,
                /*track_thresh */ cfg.track_thresh,
                /*match_thresh */ cfg.match_thresh,
                /*track_buffer */ cfg.track_buffer,
                /*frame_rate   */ frame_rate);
        } else if (cfg.type == "botsort") {
            tracker = std::make_unique<motcpp::trackers::BotSort>(
                /*reid_weights      */ "",      // empty: appearance is supplied externally via embs
                /*use_half          */ false,
                /*use_gpu           */ false,
                /*det_thresh        */ 0.1f,
                /*max_age           */ 30,
                /*max_obs           */ 50,
                /*min_hits          */ 2,
                /*iou_threshold     */ cfg.iou_threshold,
                /*per_class         */ false,
                /*nr_classes        */ 80,
                /*asso_func         */ "iou",
                /*is_obb            */ false,
                /*track_high_thresh */ cfg.track_thresh,
                /*track_low_thresh  */ cfg.min_conf,
                /*new_track_thresh  */ cfg.new_track_thresh,
                /*track_buffer      */ cfg.track_buffer,
                /*match_thresh      */ cfg.match_thresh,
                /*proximity_thresh  */ cfg.proximity_thresh,
                /*appearance_thresh */ cfg.appearance_thresh,
                /*cmc_method        */ cfg.cmc_method,
                /*frame_rate        */ frame_rate,
                /*fuse_first_assoc  */ false,
                /*with_reid         */ cfg.with_reid);
            feed_embeddings = cfg.with_reid && reid != nullptr;
        } else if (cfg.type == "ocsort") {
            // Motion-only: no embeddings, no GMC, no per-detection ReID cost.
            // OC-SORT takes no frame_rate, so its frame-count params (max_age, delta_t)
            // are referenced to 30 fps and scaled here to the source fps — otherwise at
            // 60 fps a raw max_age=30 tolerates only 0.5 s of occlusion (half of what
            // ByteTrack, which scales its buffer internally, gets).
            const float fps_scale = frame_rate > 0 ? frame_rate / 30.0f : 1.0f;
            const int max_age = std::max(1, static_cast<int>(std::lround(cfg.max_age * fps_scale)));
            const int delta_t = std::max(1, static_cast<int>(std::lround(cfg.delta_t * fps_scale)));
            tracker = std::make_unique<motcpp::trackers::OCSort>(
                /*det_thresh   */ cfg.track_thresh,   // high-confidence cut
                /*max_age      */ max_age,
                /*max_obs      */ 50,
                /*min_hits     */ 2,
                /*iou_threshold*/ cfg.iou_threshold,
                /*per_class    */ false,
                /*nr_classes   */ 80,
                /*asso_func    */ "iou",
                /*is_obb       */ false,
                /*min_conf     */ cfg.min_conf,
                /*delta_t      */ delta_t,
                /*inertia      */ cfg.inertia,
                /*use_byte     */ cfg.use_byte,
                /*Q_xy_scaling */ cfg.q_xy_scaling,
                /*Q_s_scaling  */ cfg.q_s_scaling);
        } else {
            throw std::runtime_error("MotTracker: unknown tracker.type '" + cfg.type +
                                     "' (expected 'bytetrack', 'botsort' or 'ocsort')");
        }
    }
};

MotTracker::MotTracker(int frame_rate)
    : MotTracker(TrackerCfg{}, frame_rate, nullptr) {}

MotTracker::MotTracker(const TrackerCfg& cfg, int frame_rate, std::shared_ptr<ReidEmbedder> reid)
    : impl_(std::make_unique<Impl>(cfg, frame_rate, std::move(reid))) {}

MotTracker::~MotTracker() = default;
MotTracker::MotTracker(MotTracker&&) noexcept = default;
MotTracker& MotTracker::operator=(MotTracker&&) noexcept = default;

std::vector<Track> MotTracker::update(const cv::Mat& frame,
                                      const std::vector<Detection>& dets) {
    const int n = static_cast<int>(dets.size());

    // Build the N×6 detection matrix [x1,y1,x2,y2,conf,cls].
    Eigen::MatrixXf in(n, 6);
    for (int i = 0; i < n; ++i) {
        const BBox& b = dets[i].box;
        in(i, 0) = b.x;
        in(i, 1) = b.y;
        in(i, 2) = b.x + b.w;
        in(i, 3) = b.y + b.h;
        in(i, 4) = dets[i].score;
        in(i, 5) = static_cast<float>(dets[i].class_id);
    }

    // BoT-SORT with ReID: attach one appearance vector per detection, row-aligned
    // with `in`, so association can fuse appearance with IoU. ByteTrack ignores it.
    // A degenerate box yields an empty embedding -> a zero row, which BoT-SORT
    // treats as "no appearance" (falls back to IoU for that detection).
    Eigen::MatrixXf embs;
    if (impl_->feed_embeddings && n > 0) {
        std::vector<Embedding> feats(static_cast<size_t>(n));
        int dim = 0;
        for (int i = 0; i < n; ++i) {
            feats[i] = impl_->reid->embed(frame, dets[i].box);
            if (dim == 0 && !feats[i].empty()) dim = static_cast<int>(feats[i].size());
        }
        if (dim > 0) {
            embs = Eigen::MatrixXf::Zero(n, dim);
            for (int i = 0; i < n; ++i) {
                if (static_cast<int>(feats[i].size()) == dim)
                    for (int j = 0; j < dim; ++j) embs(i, j) = feats[i][static_cast<size_t>(j)];
            }
        }
    }

    const Eigen::MatrixXf out = impl_->tracker->update(in, frame, embs);

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
