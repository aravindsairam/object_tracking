#pragma once

#include "ot/config.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ot {

// Map tiles for the map panel: memory -> disk -> network, in that order.
//
// get() never blocks: it returns the decoded tile if it is already in memory
// and otherwise queues it for ONE background loader thread (so at most one
// connection to the tile server, newest request first) and returns an empty
// Mat. The caller draws a placeholder and simply asks again next frame.
//
// Downloaded tiles are written to disk and reused forever, so an area is
// fetched once; with MapCfg::offline (or no network) only cached tiles show.
// Failed tiles are not retried for a while, so a dead network costs nothing.
class TileCache {
public:
    explicit TileCache(const MapCfg& cfg);
    ~TileCache();
    TileCache(const TileCache&) = delete;
    TileCache& operator=(const TileCache&) = delete;

    // 256x256 BGR tile, or an empty Mat when it is not in memory. `request`
    // = false only peeks (used to borrow a parent tile as a blurry stand-in).
    cv::Mat get(int z, int x, int y, bool request = true);

    // True while the most recent download attempt failed (offline / blocked).
    bool network_down() const { return network_down_.load(); }

    // Number of tiles loaded so far. Changes when a requested tile arrives, so
    // a caller that is not redrawing every frame knows when to redraw.
    uint64_t loaded() const { return loaded_.load(); }

    // Directory the tiles are cached in ("" = memory only).
    const std::string& cache_dir() const { return dir_; }

    // Tile URL for (z, x, y) from the configured template. Exposed for tests.
    std::string url_for(int z, int x, int y) const;

private:
    using Clock = std::chrono::steady_clock;
    struct Entry {
        cv::Mat  img;
        uint64_t used = 0;
    };

    static uint64_t key(int z, int x, int y) {
        return (static_cast<uint64_t>(z) << 58) | (static_cast<uint64_t>(x) << 29) |
               static_cast<uint64_t>(y);
    }
    std::string path_for(int z, int x, int y) const;
    void worker();
    void store(uint64_t k, const cv::Mat& img);

    MapCfg      cfg_;
    std::string dir_;

    std::mutex                                  m_;
    std::condition_variable                     cv_;
    std::unordered_map<uint64_t, Entry>         mem_;
    std::unordered_map<uint64_t, Clock::time_point> retry_after_;
    std::vector<uint64_t>                       pending_;   // back = newest
    std::unordered_set<uint64_t>                queued_;
    uint64_t                                    tick_ = 0;
    bool                                        stop_ = false;

    std::atomic<uint64_t> loaded_{0};
    std::atomic<bool> network_down_{false};
    std::atomic<bool> abort_{false};
    std::thread       thread_;
};

}  // namespace ot
