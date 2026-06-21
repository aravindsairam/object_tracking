#include "ot/lock_manager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace ot {

namespace {
// Per-frame lock tracing for diagnosing wrong/far re-locks. Off unless the
// OT_LOCK_DEBUG env var is set; prints to stderr so it won't disturb the HUD.
bool lock_debug() {
    static const bool on = std::getenv("OT_LOCK_DEBUG") != nullptr;
    return on;
}

bool contains(const BBox& b, const cv::Point2f& p) {
    return p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h;
}
float dist2(const cv::Point2f& a, const cv::Point2f& b) {
    const float dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}
float diag(const BBox& b) { return std::sqrt(b.w * b.w + b.h * b.h); }
constexpr float kVelAlpha = 0.5f;       // EMA factor for velocity
constexpr float kTplAlpha = 0.1f;       // EMA factor for appearance template

// Running-mode of the class votes, with hysteresis: the current class is only
// replaced by one with a STRICTLY larger count, so a tie can't flip the label.
int voted_class(const std::unordered_map<int, int>& votes, int current) {
    int best = current, best_n = -1;
    if (auto it = votes.find(current); it != votes.end()) best_n = it->second;
    for (const auto& [cls, n] : votes)
        if (n > best_n) { best_n = n; best = cls; }
    return best;
}

// new = normalize((1-a)*old + a*cur)
void blend_template(Embedding& tpl, const Embedding& cur, float a) {
    if (cur.empty()) return;
    if (tpl.size() != cur.size()) { tpl = cur; return; }
    float norm = 0.0f;
    for (size_t i = 0; i < tpl.size(); ++i) {
        tpl[i] = (1 - a) * tpl[i] + a * cur[i];
        norm += tpl[i] * tpl[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) for (float& v : tpl) v /= norm;
}
}  // namespace

LockManager::LockManager(double fps, int coast_to_lost, float reacquire_thresh,
                         std::shared_ptr<ReidEmbedder> reid, float verify_thresh,
                         float reacquire_max_frac, float reacquire_margin)
    : fps_(fps > 0 ? fps : 30.0),
      coast_to_lost_(std::max(1, coast_to_lost)),
      reacquire_thresh_(reacquire_thresh),
      verify_thresh_(verify_thresh),
      reacquire_max_frac_(reacquire_max_frac > 0.0f ? reacquire_max_frac : 0.2f),
      reacquire_margin_(std::max(0.0f, reacquire_margin)),
      reid_(std::move(reid)) {}

void LockManager::reset() {
    locked_id_ = -1;
    coast_ = 0;
    vel_per_frame_ = {0, 0};
    template_.clear();
    template_size_ = {0, 0};
    lock_id_ = -1;
    class_votes_.clear();
    cur_ = TargetState{};
}

void LockManager::select(const cv::Point2f& click, const cv::Mat& frame,
                         const std::vector<Track>& tracks) {
    const Track* best = nullptr;
    float best_d2 = std::numeric_limits<float>::max();
    for (const auto& t : tracks) {
        if (contains(t.box, click)) {
            const float d2 = dist2(t.box.center(), click);
            if (d2 < best_d2) { best_d2 = d2; best = &t; }
        }
    }
    if (!best) {
        for (const auto& t : tracks) {
            const float d2 = dist2(t.box.center(), click);
            if (d2 < best_d2) { best_d2 = d2; best = &t; }
        }
        const float max_r = 60.0f;
        if (best && best_d2 > max_r * max_r) best = nullptr;
    }
    if (!best) return;

    locked_id_ = best->id;
    coast_ = 0;
    vel_per_frame_ = {0, 0};
    template_ = reid_ ? reid_->embed(frame, best->box) : Embedding{};
    template_size_ = {best->box.w, best->box.h};
    lock_id_ = next_lock_id_++;          // a fresh, stable id for this lock
    class_votes_.clear();
    class_votes_[best->class_id]++;
    cur_ = TargetState{};
    cur_.state = LockState::Locked;
    cur_.box = best->box;
    cur_.class_id = best->class_id;
    cur_.track_id = best->id;
    cur_.lock_id = lock_id_;
    cur_.confidence = 1.0f;
}

int LockManager::try_reacquire(const cv::Mat& frame, const std::vector<Track>& tracks) const {
    if (template_.empty() || !reid_) return -1;

    const cv::Point2f predicted = cur_.box.center();
    const float tpl_area = std::max(1.0f, template_size_.width * template_size_.height);
    const float tpl_diag = std::sqrt(template_size_.width * template_size_.width +
                                     template_size_.height * template_size_.height);
    const float vmag = std::sqrt(vel_per_frame_.x * vel_per_frame_.x +
                                 vel_per_frame_.y * vel_per_frame_.y);
    // Search window grows with how long we've been lost, but is hard-capped to a
    // fraction of the frame's short side. The old cap was half the frame DIAGONAL,
    // which let a long occlusion re-acquire an object clear across the screen — the
    // confirmed cause of far/wrong re-locks (field logs: dist≈300-500 px grabs).
    const float max_r = reacquire_max_frac_ * std::min(frame.cols, frame.rows);
    const float radius = std::min(max_r, 60.0f + 2.0f * tpl_diag + 2.0f * vmag * coast_);
    const float radius2 = radius * radius;

    int   best_idx = -1;
    float best_score = -1.0f, second_score = -1.0f;
    for (int i = 0; i < static_cast<int>(tracks.size()); ++i) {
        const Track& t = tracks[i];
        if (dist2(t.box.center(), predicted) > radius2) continue;       // position gate
        const float ratio = (t.box.w * t.box.h) / tpl_area;
        if (ratio < 0.4f || ratio > 2.5f) continue;                     // size gate
        const float score = ReidEmbedder::cosine(template_, reid_->embed(frame, t.box));
        if (score > best_score) { second_score = best_score; best_score = score; best_idx = i; }
        else if (score > second_score) { second_score = score; }
    }
    // Require a confident, unambiguous match: above threshold AND clearly ahead
    // of the runner-up (so we don't bind a similar-looking neighbor).
    const bool ok = best_score >= reacquire_thresh_ &&
                    !(second_score >= 0.0f && (best_score - second_score) < reacquire_margin_);
    if (lock_debug() && best_idx >= 0) {
        const float d = std::sqrt(dist2(tracks[best_idx].box.center(), predicted));
        std::fprintf(stderr,
            "[lock] reacquire %s: best=%.3f 2nd=%.3f thr=%.2f radius=%.0f dist=%.0f id=%d coast=%d\n",
            ok ? "ACCEPT" : "reject", best_score, second_score, reacquire_thresh_,
            radius, d, tracks[best_idx].id, coast_);
    }
    if (!ok) return -1;
    return best_idx;
}

TargetState LockManager::update(const cv::Mat& frame, const std::vector<Track>& tracks) {
    if (locked_id_ < 0) { cur_.state = LockState::Acquiring; return cur_; }

    const Track* match = nullptr;
    for (const auto& t : tracks) {
        if (t.id == locked_id_) { match = &t; break; }
    }

    // Verify the id-matched track by appearance before trusting it: ByteTrack
    // reuses a track id across different objects (it keeps a lost track's id alive
    // and re-binds it to whatever detection lands near the predicted box), so a
    // surviving id is NOT proof of identity. If the matched box looks nothing like
    // the stored template, drop it and treat this frame as a miss — that turns a
    // silent id-switch into a coast, and lets re-acquire (or LOST) take over.
    // Reuse this embedding for the template blend below so it costs no extra pass.
    Embedding cur_emb;
    if (match && reid_ && verify_thresh_ > 0.0f && !template_.empty()) {
        cur_emb = reid_->embed(frame, match->box);
        const float sim = ReidEmbedder::cosine(template_, cur_emb);
        if (lock_debug())
            std::fprintf(stderr, "[lock] verify id=%d cos=%.3f thr=%.2f -> %s\n",
                         match->id, sim, verify_thresh_, sim < verify_thresh_ ? "REJECT" : "ok");
        if (sim < verify_thresh_) match = nullptr;
    } else if (match && lock_debug()) {
        std::fprintf(stderr, "[lock] verify SKIPPED id=%d (reid=%d tpl_empty=%d thr=%.2f)\n",
                     match->id, reid_ ? 1 : 0, template_.empty() ? 1 : 0, verify_thresh_);
    }

    // Appearance re-acquire only once the id is truly LOST (id-continuity handles
    // short gaps); re-binding during a brief coast risks grabbing a neighbor.
    bool reacquired = false;
    if (!match && (coast_ + 1) > coast_to_lost_) {
        const int idx = try_reacquire(frame, tracks);
        if (idx >= 0) { match = &tracks[idx]; locked_id_ = match->id; reacquired = true; cur_emb.clear(); }
    }

    if (match) {
        const cv::Point2f c_old = cur_.box.center();
        const cv::Point2f c_new = match->box.center();
        // Skip velocity blending on a re-acquire jump (it's not real motion).
        if (cur_.valid() && !reacquired) {
            vel_per_frame_.x = kVelAlpha * (c_new.x - c_old.x) + (1 - kVelAlpha) * vel_per_frame_.x;
            vel_per_frame_.y = kVelAlpha * (c_new.y - c_old.y) + (1 - kVelAlpha) * vel_per_frame_.y;
        }
        coast_ = 0;
        cur_.state = LockState::Locked;
        cur_.box = match->box;
        class_votes_[match->class_id]++;            // vote, then report the running mode
        cur_.class_id = voted_class(class_votes_, cur_.class_id);
        cur_.track_id = match->id;                  // raw MOT id (may churn)
        cur_.confidence = 1.0f;
        if (reid_) blend_template(template_, cur_emb.empty() ? reid_->embed(frame, match->box) : cur_emb, kTplAlpha);
        template_size_ = {match->box.w, match->box.h};
        if (lock_debug() && reacquired)
            std::fprintf(stderr, "[lock] RE-LOCKED onto id=%d\n", match->id);
    } else {
        ++coast_;
        if (coast_ > coast_to_lost_) {
            if (lock_debug() && cur_.state != LockState::Lost)
                std::fprintf(stderr, "[lock] -> LOST (was id=%d)\n", locked_id_);
            cur_.state = LockState::Lost;
            cur_.confidence = 0.0f;
        } else {
            cur_.state = LockState::Coasting;
            cur_.box.x += vel_per_frame_.x;
            cur_.box.y += vel_per_frame_.y;
            cur_.confidence = 1.0f - static_cast<float>(coast_) / coast_to_lost_;
        }
    }

    // The lock's identity is stable for its whole life — set it on every path
    // (locked / coasting / lost) so the HUD never shows the churning MOT id.
    cur_.lock_id = lock_id_;

    // Keep a coasting/predicted box within the frame so it doesn't drift off.
    cur_.box.clamp(frame.cols, frame.rows);
    cur_.center_px = cur_.box.center();
    cur_.velocity_px_s = {vel_per_frame_.x * static_cast<float>(fps_),
                          vel_per_frame_.y * static_cast<float>(fps_)};
    return cur_;
}

}  // namespace ot
