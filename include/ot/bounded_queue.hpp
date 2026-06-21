#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

namespace ot {

// A bounded, thread-safe producer/consumer queue used to pipeline the capture,
// inference and display stages. Blocking push/pop apply backpressure (no frames
// dropped — correct for file playback); push_latest drops the oldest instead
// (lowest-latency, for live sources). close() wakes every waiter so threads can
// shut down cleanly without deadlocking on a full/empty queue.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : cap_(capacity) {}

    // Blocks while full. Returns false if the queue was closed.
    bool push(T value) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return false;
        q_.push_back(std::move(value));
        not_empty_.notify_one();
        return true;
    }

    // Never blocks: if full, drops the oldest item. Returns false if closed.
    bool push_latest(T value) {
        std::unique_lock<std::mutex> lk(m_);
        if (closed_) return false;
        while (q_.size() >= cap_) q_.pop_front();
        q_.push_back(std::move(value));
        not_empty_.notify_one();
        return true;
    }

    // Blocks until an item is available. Returns false once closed AND drained.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return true;
    }

    // Like pop but with a timeout. Returns: 1 = got item, 0 = timed out,
    // -1 = closed and drained. Lets the display/UI thread stay responsive
    // (pump keys) even when no new frames are flowing (e.g. paused).
    int pop_for(T& out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(m_);
        if (!not_empty_.wait_for(lk, timeout, [&] { return !q_.empty() || closed_; }))
            return 0;
        if (q_.empty()) return -1;
        out = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return 1;
    }

    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::mutex                  m_;
    std::condition_variable     not_full_;
    std::condition_variable     not_empty_;
    std::deque<T>               q_;
    std::size_t                 cap_;
    bool                        closed_ = false;
};

}  // namespace ot
