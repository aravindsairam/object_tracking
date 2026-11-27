// Background map-tile loader with a disk + memory cache. See tile_cache.hpp.
#include "ot/tile_cache.hpp"

#include <curl/curl.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace ot {

namespace {
constexpr size_t kMaxMemTiles   = 512;              // ~100 MB of decoded tiles
constexpr size_t kMaxPending    = 64;               // older requests are dropped
constexpr size_t kMaxTileBytes  = 4u * 1024 * 1024;
constexpr auto   kRetryFailed   = std::chrono::seconds(30);
constexpr auto   kRetryMissing  = std::chrono::minutes(10);   // server said "no such tile"

void unpack(uint64_t k, int& z, int& x, int& y) {
    z = static_cast<int>(k >> 58);
    x = static_cast<int>((k >> 29) & 0x1fffffff);
    y = static_cast<int>(k & 0x1fffffff);
}

// FNV-1a, to give every tile source its own cache folder.
std::string hash_hex(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (const unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

std::string default_cache_root() {
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) return std::string(x) + "/map_tiles";
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.cache/map_tiles";
    return "";
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
    for (size_t p = s.find(from); p != std::string::npos; p = s.find(from, p + to.size()))
        s.replace(p, from.size(), to);
}

// Decodes tile bytes to a 256x256 BGR image (empty on failure).
cv::Mat decode_tile(const std::vector<uchar>& bytes) {
    if (bytes.empty()) return {};
    cv::Mat img = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (img.empty()) return {};
    if (img.cols != 256 || img.rows != 256)
        cv::resize(img, img, {256, 256}, 0, 0, cv::INTER_AREA);
    return img;
}

size_t on_data(char* ptr, size_t size, size_t nmemb, void* user) {
    auto* buf = static_cast<std::vector<uchar>*>(user);
    const size_t n = size * nmemb;
    if (buf->size() + n > kMaxTileBytes) return 0;   // abort: not a map tile
    buf->insert(buf->end(), ptr, ptr + n);
    return n;
}

// Lets the destructor cut a transfer short instead of waiting out a timeout.
int on_progress(void* user, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    return static_cast<std::atomic<bool>*>(user)->load() ? 1 : 0;
}
}  // namespace

TileCache::TileCache(const MapCfg& cfg) : cfg_(cfg) {
    std::string root = cfg_.cache_dir.empty() ? default_cache_root() : cfg_.cache_dir;
    if (!root.empty()) {
        dir_ = root + "/" + hash_hex(cfg_.tile_url);
        std::error_code ec;
        std::filesystem::create_directories(dir_, ec);
        if (ec) {
            std::fprintf(stderr, "[map] cannot create tile cache '%s' (%s); caching in memory only\n",
                         dir_.c_str(), ec.message().c_str());
            dir_.clear();
        }
    }
    static std::once_flag curl_once;
    std::call_once(curl_once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    thread_ = std::thread([this] { worker(); });
}

TileCache::~TileCache() {
    {
        std::lock_guard<std::mutex> g(m_);
        stop_ = true;
    }
    abort_.store(true);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

std::string TileCache::url_for(int z, int x, int y) const {
    std::string url = cfg_.tile_url;
    replace_all(url, "{z}", std::to_string(z));
    replace_all(url, "{x}", std::to_string(x));
    replace_all(url, "{y}", std::to_string(y));
    return url;
}

std::string TileCache::path_for(int z, int x, int y) const {
    if (dir_.empty()) return "";
    return dir_ + "/" + std::to_string(z) + "/" + std::to_string(x) + "/" +
           std::to_string(y) + ".tile";
}

cv::Mat TileCache::get(int z, int x, int y, bool request) {
    if (z < 0 || z > 22 || x < 0 || y < 0 || x >= (1 << z) || y >= (1 << z)) return {};
    const uint64_t k = key(z, x, y);
    std::lock_guard<std::mutex> g(m_);
    const auto it = mem_.find(k);
    if (it != mem_.end()) {
        it->second.used = ++tick_;
        return it->second.img;
    }
    if (!request || queued_.count(k)) return {};
    const auto r = retry_after_.find(k);
    if (r != retry_after_.end()) {
        if (Clock::now() < r->second) return {};
        retry_after_.erase(r);
    }
    if (pending_.size() >= kMaxPending) {            // drop the stalest request
        queued_.erase(pending_.front());
        pending_.erase(pending_.begin());
    }
    pending_.push_back(k);
    queued_.insert(k);
    cv_.notify_one();
    return {};
}

void TileCache::store(uint64_t k, const cv::Mat& img) {
    // Caller holds m_.
    if (mem_.size() >= kMaxMemTiles) {               // evict the least recently used ~1/8
        std::vector<std::pair<uint64_t, uint64_t>> age;   // (used, key)
        age.reserve(mem_.size());
        for (const auto& kv : mem_) age.emplace_back(kv.second.used, kv.first);
        const size_t drop = kMaxMemTiles / 8;
        std::nth_element(age.begin(), age.begin() + drop, age.end());
        for (size_t i = 0; i < drop; ++i) mem_.erase(age[i].second);
    }
    mem_[k] = Entry{img, ++tick_};
}

void TileCache::worker() {
    CURL* curl = nullptr;   // created on first download; reused (keep-alive)

    for (;;) {
        uint64_t k = 0;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return stop_ || !pending_.empty(); });
            if (stop_) break;
            k = pending_.back();                     // newest first: what is on screen now
            pending_.pop_back();
        }
        int z, x, y;
        unpack(k, z, x, y);

        cv::Mat img;
        auto retry = kRetryFailed;
        const std::string path = path_for(z, x, y);

        // 1) disk
        if (!path.empty()) {
            std::ifstream in(path, std::ios::binary);
            if (in) {
                std::vector<uchar> bytes((std::istreambuf_iterator<char>(in)),
                                         std::istreambuf_iterator<char>());
                img = decode_tile(bytes);
            }
        }

        // 2) network
        if (img.empty() && !cfg_.offline) {
            if (!curl) curl = curl_easy_init();
            if (curl) {
                std::vector<uchar> bytes;
                const std::string url = url_for(z, x, y);
                curl_easy_reset(curl);
                curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                curl_easy_setopt(curl, CURLOPT_USERAGENT, cfg_.user_agent.c_str());
                curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
                curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 3L);
#if LIBCURL_VERSION_NUM >= 0x075500
                curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
                curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
#else
                curl_easy_setopt(curl, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
                curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
#endif
                curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
                curl_easy_setopt(curl, CURLOPT_TIMEOUT, 12L);
                curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, on_data);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, &bytes);
                curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
                curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, on_progress);
                curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &abort_);

                const CURLcode rc = curl_easy_perform(curl);
                long status = 0;
                curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
                if (rc == CURLE_OK && status == 200) {
                    network_down_.store(false);
                    img = decode_tile(bytes);
                    if (!img.empty() && !path.empty()) {
                        // Write-then-rename so a killed run never leaves half a tile.
                        std::error_code ec;
                        std::filesystem::create_directories(
                            std::filesystem::path(path).parent_path(), ec);
                        const std::string tmp = path + ".part";
                        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
                        out.write(reinterpret_cast<const char*>(bytes.data()),
                                  static_cast<std::streamsize>(bytes.size()));
                        out.close();
                        if (out) std::filesystem::rename(tmp, path, ec);
                        else std::filesystem::remove(tmp, ec);
                    }
                } else if (rc == CURLE_OK) {
                    // The server answered: reachable, it just has no such tile
                    // (or refuses us). Either way, do not hammer it.
                    network_down_.store(false);
                    retry = std::chrono::duration_cast<std::chrono::seconds>(kRetryMissing);
                } else if (rc != CURLE_ABORTED_BY_CALLBACK) {
                    if (!network_down_.exchange(true))
                        std::fprintf(stderr, "[map] tile download failed (%s); showing cached tiles only\n",
                                     curl_easy_strerror(rc));
                }
            }
        }

        std::lock_guard<std::mutex> g(m_);
        queued_.erase(k);
        if (!img.empty()) {
            store(k, img);
            loaded_.fetch_add(1);
        } else {
            retry_after_[k] = Clock::now() + retry;
        }
    }

    if (curl) curl_easy_cleanup(curl);
}

}  // namespace ot
