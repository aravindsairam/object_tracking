#pragma once

#include "ot/reid.hpp"
#include "ot/target_state.hpp"
#include "ot/types.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ot {

// Owns the single locked target and decides its lock state every frame. This is
// the single source of truth for "are we still locked?" — never a tracker's own
// score. For the MVP it binds to a MOT track id; when that id is briefly absent
// it COASTS with a constant-velocity prediction, and declares LOST after a
// timeout (re-acquire by appearance is added in M5).
class LockManager {
public:
    // `fps` reports velocity in px/s; `coast_to_lost` is the consecutive missing
    // frames tolerated before LOST; `reacquire_thresh` is the appearance cosine
    // needed to auto re-lock; `reid` produces appearance signatures.
    // `verify_thresh` is the appearance cosine the id-matched track must clear to
    // be trusted each frame — the guard against ByteTrack re-using the locked id
    // on a different object. A match below it is treated as a miss (coast), not a
    // confident lock. Set <= 0 to disable (pure id-continuity, the old behavior).
    // `reacquire_max_frac` caps the appearance re-acquire search radius to this
    // fraction of min(frame W, H): a lost target cannot teleport across the frame,
    // so re-acquiring something far away is almost always a wrong object.
    // `reacquire_margin` is how far the best candidate's appearance cosine must beat
    // the runner-up's to re-acquire — refuses an ambiguous match when two nearby
    // objects look almost equally like the target (don't bind a coin-flip).
    LockManager(double fps, int coast_to_lost, float reacquire_thresh,
                std::shared_ptr<ReidEmbedder> reid, float verify_thresh = 0.35f,
                float reacquire_max_frac = 0.2f, float reacquire_margin = 0.10f);

    // Locks onto the track under (or nearest to) `click` and stores its
    // appearance template from `frame`. No-op if no track is close.
    void select(const cv::Point2f& click, const cv::Mat& frame,
                const std::vector<Track>& tracks);

    // Clears the lock (back to Acquiring).
    void reset();

    // Advances the state machine: follows the locked id, coasts when briefly
    // missing, and auto re-acquires by appearance+motion while coasting/lost.
    TargetState update(const cv::Mat& frame, const std::vector<Track>& tracks);

    LockState state() const { return cur_.state; }
    bool      has_target() const { return cur_.valid(); }
    int       locked_id() const { return locked_id_; }

private:
    // Searches tracks for the lost target by appearance + position/size gating.
    // Returns the matching track index, or -1.
    int try_reacquire(const cv::Mat& frame, const std::vector<Track>& tracks) const;

    double       fps_;
    int          coast_to_lost_;
    float        reacquire_thresh_;
    float        verify_thresh_;
    float        reacquire_max_frac_;
    float        reacquire_margin_;
    std::shared_ptr<ReidEmbedder> reid_;

    int          locked_id_ = -1;
    int          coast_ = 0;
    cv::Point2f  vel_per_frame_{0, 0};   // smoothed center velocity (px/frame)
    Embedding    template_;               // appearance signature of the target
    cv::Size2f   template_size_{0, 0};
    TargetState  cur_;                    // last reported state

    // Stable identity of the current lock, independent of the churning MOT id.
    int          lock_id_ = -1;           // id of the current lock (kept across re-acquire)
    int          next_lock_id_ = 1;       // source of new lock ids (one per select)
    // Class vote since this lock began: the displayed class is the running mode,
    // so a frame or two of detector misclassification can't flip the label.
    std::unordered_map<int, int> class_votes_;
};

}  // namespace ot
