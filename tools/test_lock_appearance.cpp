// Unit test for appearance verification on the id-matched path of LockManager.
//
// Root cause this guards against: ByteTrack reuses a track id across different
// physical objects. The lock used to trust the id alone (no appearance check on
// the matched path), so when the id was re-bound to a different object the lock
// followed it at full confidence. This test feeds a track that KEPT the locked
// id but sits where a different-looking object is, and asserts the lock refuses
// to adopt it.
//
// Uses a fake embedder (appearance keyed by box x-position) so it needs no
// model, GPU, or video — pure logic.
#include "ot/lock_manager.hpp"
#include "ot/reid.hpp"
#include "ot/types.hpp"

#include <opencv2/core.hpp>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

// Appearance depends only on horizontal position: objects on the left half and
// right half of the frame get orthogonal unit embeddings (cosine 0). So a track
// that jumps from the left object to the right object looks completely different.
class FakeReid : public ot::ReidEmbedder {
public:
    ot::Embedding embed(const cv::Mat&, const ot::BBox& box) override {
        return box.center().x < 400.0f ? ot::Embedding{1.0f, 0.0f}
                                       : ot::Embedding{0.0f, 1.0f};
    }
};

// Everything looks identical to the target, so appearance can never reject a
// candidate — only the distance/size gate can. Used to prove the re-acquire
// search radius is bounded (a far look-alike must not be grabbed).
class SameLookReid : public ot::ReidEmbedder {
public:
    ot::Embedding embed(const cv::Mat&, const ot::BBox&) override {
        return ot::Embedding{1.0f, 0.0f};
    }
};

// Returns a unit vector keyed by horizontal position, so a candidate's cosine to
// the template (object at center.x<200 -> (1,0)) is a chosen value. Used to craft
// ambiguous re-acquire cases: two candidates that score 0.93 and 0.88.
//   center.x  < 200  -> (1,0)      cos 1.00  (the template object)
//   center.x in 200..256 -> (.930,.368)  cos 0.93
//   center.x in 256..272 -> (.880,.475)  cos 0.88
//   center.x >= 272  -> (.500,.866)  cos 0.50
class CraftReid : public ot::ReidEmbedder {
public:
    ot::Embedding embed(const cv::Mat&, const ot::BBox& b) override {
        const float x = b.center().x;
        if (x < 200.0f) return {1.000f, 0.000f};
        if (x < 256.0f) return {0.930f, 0.368f};
        if (x < 272.0f) return {0.880f, 0.475f};
        return {0.500f, 0.866f};
    }
};

int failures = 0;
void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

const char* sname(ot::LockState s) {
    switch (s) {
        case ot::LockState::Acquiring: return "ACQUIRING";
        case ot::LockState::Locked:    return "LOCKED";
        case ot::LockState::Coasting:  return "COASTING";
        case ot::LockState::Lost:      return "LOST";
    }
    return "?";
}

ot::Track track(int id, float x) {
    ot::Track t;
    t.id = id;
    t.box = {x, 300.0f, 100.0f, 150.0f};   // center.x = x + 50
    t.score = 0.9f;
    t.class_id = 0;
    return t;
}

ot::Track track_cls(int id, float x, int cls) {
    ot::Track t = track(id, x);
    t.class_id = cls;
    return t;
}

// Lock onto object A on the left, then feed a track that kept A's id but is now
// where right-side object B is. The lock must NOT report a confident lock on it.
void test_rejects_id_steal_with_mismatched_appearance() {
    std::printf("test: rejects id-steal with mismatched appearance\n");
    cv::Mat frame = cv::Mat::zeros(720, 1280, CV_8UC3);
    ot::LockManager lock(/*fps*/ 30.0, /*coast_to_lost*/ 30,
                         /*reacquire_thresh*/ 0.6f, std::make_shared<FakeReid>());

    ot::Track a = track(7, 100.0f);          // left object, center.x = 150
    lock.select(a.box.center(), frame, {a});
    check(lock.locked_id() == 7, "locked onto object A (id 7)");

    ot::Track impostor = track(7, 600.0f);   // SAME id, right object, center.x = 650
    ot::TargetState ts = lock.update(frame, {impostor});
    std::printf("  -> state=%s box.center.x=%.0f conf=%.2f\n",
                sname(ts.state), ts.box.center().x, ts.confidence);

    check(ts.state != ot::LockState::Locked, "did not LOCK onto the impostor");
    check(ts.box.center().x < 400.0f, "did not jump the box to the impostor");
}

// Same object moving slightly must still lock — guards against over-rejection.
void test_accepts_matching_appearance() {
    std::printf("test: accepts same object across a small move\n");
    cv::Mat frame = cv::Mat::zeros(720, 1280, CV_8UC3);
    ot::LockManager lock(30.0, 30, 0.6f, std::make_shared<FakeReid>());

    ot::Track a = track(7, 100.0f);
    lock.select(a.box.center(), frame, {a});

    ot::Track moved = track(7, 110.0f);      // still the left object
    ot::TargetState ts = lock.update(frame, {moved});
    std::printf("  -> state=%s box.center.x=%.0f conf=%.2f\n",
                sname(ts.state), ts.box.center().x, ts.confidence);

    check(ts.state == ot::LockState::Locked, "stayed LOCKED on the real object");
}

// Drive the lock through a loss, then offer a same-looking candidate far across
// the frame. The lock must NOT re-acquire something hundreds of px away — that's
// the "loses target, then locks a far object" bug from the field logs.
void test_reacquire_rejects_far_lookalike() {
    std::printf("test: re-acquire rejects a far look-alike\n");
    cv::Mat frame = cv::Mat::zeros(720, 1280, CV_8UC3);
    ot::LockManager lock(/*fps*/ 30.0, /*coast_to_lost*/ 3, /*reacquire_thresh*/ 0.6f,
                         std::make_shared<SameLookReid>(), /*verify_thresh*/ 0.35f);

    ot::Track a = track(7, 100.0f);          // center (150, 375)
    lock.select(a.box.center(), frame, {a});

    ot::Track far = track(99, 400.0f);       // center (450, 375) -> 300 px away
    ot::TargetState ts;
    for (int k = 0; k < 8; ++k) ts = lock.update(frame, {far});   // lose, then offer far
    std::printf("  -> state=%s locked_id=%d\n", sname(ts.state), lock.locked_id());

    check(lock.locked_id() != 99, "did not re-acquire the far look-alike");
    check(ts.state == ot::LockState::Lost, "stayed LOST instead of far-locking");
}

// Guard against over-tightening: a same-looking candidate right next to the last
// known position must still re-acquire.
void test_reacquire_accepts_near_lookalike() {
    std::printf("test: re-acquire accepts a near look-alike\n");
    cv::Mat frame = cv::Mat::zeros(720, 1280, CV_8UC3);
    ot::LockManager lock(30.0, 3, 0.6f, std::make_shared<SameLookReid>(), 0.35f);

    ot::Track a = track(7, 100.0f);          // center (150, 375)
    lock.select(a.box.center(), frame, {a});

    ot::Track near = track(99, 140.0f);      // center (190, 375) -> 40 px away
    ot::TargetState ts;
    for (int k = 0; k < 8; ++k) ts = lock.update(frame, {near});
    std::printf("  -> state=%s locked_id=%d\n", sname(ts.state), lock.locked_id());

    check(lock.locked_id() == 99, "re-acquired the nearby look-alike");
    check(ts.state == ot::LockState::Locked, "locked after a valid nearby re-acquire");
}

// Two nearby candidates that look almost equally like the target (0.93 vs 0.88)
// must NOT be re-acquired — the runner-up is too close to be sure which is real.
// This is the field-log case: best=0.918 2nd=0.862 -> grabbed the wrong one.
void test_reacquire_rejects_ambiguous_lookalikes() {
    std::printf("test: re-acquire rejects ambiguous look-alikes (close runner-up)\n");
    cv::Mat frame = cv::Mat::zeros(720, 1280, CV_8UC3);
    ot::LockManager lock(/*fps*/ 30.0, /*coast_to_lost*/ 3, /*reacquire_thresh*/ 0.6f,
                         std::make_shared<CraftReid>(), /*verify_thresh*/ 0.35f,
                         /*reacquire_max_frac*/ 0.2f);   // runner-up margin uses new default

    ot::Track a = track(7, 100.0f);          // center 150 -> template (1,0)
    lock.select(a.box.center(), frame, {a});

    ot::Track c1 = track(101, 200.0f);       // center 250 -> cos 0.93, dist 100
    ot::Track c2 = track(102, 210.0f);       // center 260 -> cos 0.88, dist 110
    ot::TargetState ts;
    for (int k = 0; k < 8; ++k) ts = lock.update(frame, {c1, c2});
    std::printf("  -> state=%s locked_id=%d\n", sname(ts.state), lock.locked_id());

    check(lock.locked_id() != 101 && lock.locked_id() != 102,
          "did not grab either ambiguous look-alike");
    check(ts.state == ot::LockState::Lost, "stayed LOST on an ambiguous match");
}

// But a clear winner (0.93 vs a distant 0.50) must still re-acquire.
void test_reacquire_accepts_clear_winner() {
    std::printf("test: re-acquire accepts a clear winner\n");
    cv::Mat frame = cv::Mat::zeros(720, 1280, CV_8UC3);
    ot::LockManager lock(30.0, 3, 0.6f, std::make_shared<CraftReid>(), 0.35f, 0.2f);

    ot::Track a = track(7, 100.0f);          // center 150 -> (1,0)
    lock.select(a.box.center(), frame, {a});

    ot::Track c1 = track(101, 200.0f);       // center 250 -> cos 0.93, dist 100
    ot::Track other = track(102, 222.0f);    // center 272 -> cos 0.50, dist 122
    ot::TargetState ts;
    for (int k = 0; k < 8; ++k) ts = lock.update(frame, {c1, other});
    std::printf("  -> state=%s locked_id=%d\n", sname(ts.state), lock.locked_id());

    check(lock.locked_id() == 101, "re-acquired the clear winner");
    check(ts.state == ot::LockState::Locked, "locked after a clear-winner re-acquire");
}

// The HUD id must stay the same object across a re-acquire, even though the
// underlying MOT id churns. (Field: one target showed ids 23,47,935,3074,3652.)
void test_lock_id_stable_across_reacquire() {
    std::printf("test: lock id is stable across a re-acquire\n");
    cv::Mat frame = cv::Mat::zeros(720, 1280, CV_8UC3);
    ot::LockManager lock(30.0, 3, 0.6f, std::make_shared<SameLookReid>(), 0.35f, 0.2f, 0.10f);

    ot::Track a = track(7, 100.0f);
    lock.select(a.box.center(), frame, {a});
    ot::TargetState ts = lock.update(frame, {a});
    const int L = ts.lock_id;
    check(L >= 0, "a stable lock id was assigned on select");

    ot::Track nearer = track(99, 140.0f);    // same look, new MOT id, 40 px away
    for (int k = 0; k < 8; ++k) ts = lock.update(frame, {nearer});   // lose 7, re-acquire 99
    std::printf("  -> track_id=%d lock_id=%d (was %d)\n", ts.track_id, ts.lock_id, L);
    check(ts.track_id == 99, "(sanity) underlying MOT id changed to 99");
    check(ts.lock_id == L, "lock id stayed stable across the re-acquire");
}

// A one-frame class misclassification (person -> bicycle) must not change the
// reported class. (Field: person=2476 frames flickering to bicycle=799, moto=238.)
void test_class_is_stable_under_flicker() {
    std::printf("test: class is stable under a one-frame flicker\n");
    cv::Mat frame = cv::Mat::zeros(720, 1280, CV_8UC3);
    ot::LockManager lock(30.0, 30, 0.6f, std::make_shared<SameLookReid>(), 0.35f, 0.2f, 0.10f);

    ot::Track person = track_cls(7, 100.0f, 1);    // 1 = person
    lock.select(person.box.center(), frame, {person});
    ot::TargetState ts;
    for (int k = 0; k < 5; ++k) ts = lock.update(frame, {track_cls(7, 100.0f, 1)});
    ts = lock.update(frame, {track_cls(7, 100.0f, 2)});   // one bicycle-flicker frame
    std::printf("  -> reported class_id=%d (flicker frame said 2)\n", ts.class_id);
    check(ts.class_id == 1, "stayed person through a one-frame bicycle misclassification");
}

}  // namespace

int main() {
    test_rejects_id_steal_with_mismatched_appearance();
    test_accepts_matching_appearance();
    test_reacquire_rejects_far_lookalike();
    test_reacquire_accepts_near_lookalike();
    test_reacquire_rejects_ambiguous_lookalikes();
    test_reacquire_accepts_clear_winner();
    test_lock_id_stable_across_reacquire();
    test_class_is_stable_under_flicker();
    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
