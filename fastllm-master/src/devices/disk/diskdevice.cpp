#include "devices/disk/diskdevice.h"
#include "gguf.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <unordered_map>

namespace fastllm {
#ifdef _WIN32
    typedef int DiskReadResult;
#else
    typedef ssize_t DiskReadResult;
#endif

    DiskDevice::DiskDevice() {
        this->deviceType = "disk";
        this->ops["MergeMOE"] = (BaseOperator*)(new DiskMergeMOE());
    }

    bool DiskDevice::Malloc(void **ret, size_t size) {
        *ret = (void*)new uint8_t[size];
        return true;
    }

    bool DiskDevice::Free(void *ret) {
        delete[] (uint8_t*)ret;
        return true;
    }

    bool DiskDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        if (dst != src && dst != nullptr && src != nullptr) {
            memcpy(dst, src, size);
        }
        return true;
    }

    bool DiskDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        if (dst != src && dst != nullptr && src != nullptr) {
            memcpy(dst, src, size);
        }
        return true;
    }

    static size_t DiskPartCount(const DiskWeightPart &part) {
        size_t count = 1;
        for (int dim : part.dims) {
            count *= dim;
        }
        return count;
    }

    static int DiskMoeLoadThreads() {
        static int threads = []() {
            const char *env = std::getenv("FASTLLM_DISK_MOE_LOAD_THREADS");
            int v = env == nullptr ? 4 : atoi(env);
            return std::max(1, v);
        }();
        return threads;
    }

    class DiskFileCache {
    public:
        ~DiskFileCache() {
            for (auto &it : fds) {
#ifdef _WIN32
                _close(it.second);
#else
                close(it.second);
#endif
            }
        }

        int Get(const std::string &fileName) {
#ifdef _WIN32
            auto it = fds.find(fileName);
            if (it != fds.end()) {
                return it->second;
            }
            int fd = FastllmOpenReadOnlyBinary(fileName);
            if (fd < 0) {
                ErrorInFastLLM("Disk MoE can't open weight file: " + fileName + "\n");
            }
            fds[fileName] = fd;
            return fd;
#else
            std::lock_guard<std::mutex> guard(locker);
            auto it = fds.find(fileName);
            if (it != fds.end()) {
                return it->second;
            }
            int fd = FastllmOpenReadOnlyBinary(fileName);
            if (fd < 0) {
                ErrorInFastLLM("Disk MoE can't open weight file: " + fileName + "\n");
            }
            fds[fileName] = fd;
            return fd;
#endif
        }

    private:
#ifndef _WIN32
        std::mutex locker;
#endif
        std::unordered_map<std::string, int> fds;
    };

    static DiskFileCache &GetDiskFileCache() {
#ifdef _WIN32
        // _read uses the descriptor's file pointer. Keep one descriptor cache per
        // worker thread so parallel Disk MoE loads do not serialize on seek/read.
        thread_local DiskFileCache cache;
#else
        static DiskFileCache cache;
#endif
        return cache;
    }

    static DiskReadResult ReadDiskFileAt(int fd, uint8_t *dst, size_t bytes, uint64_t offset) {
#ifdef _WIN32
        HANDLE handle = (HANDLE)_get_osfhandle(fd);
        if (handle == INVALID_HANDLE_VALUE) {
            return -1;
        }
        // Synchronous ReadFile accepts an OVERLAPPED offset, avoiding a separate seek.
        OVERLAPPED overlapped;
        memset(&overlapped, 0, sizeof(overlapped));
        overlapped.Offset = (DWORD)(offset & 0xFFFFFFFFULL);
        overlapped.OffsetHigh = (DWORD)(offset >> 32);
        DWORD readBytes = 0;
        if (!ReadFile(handle, dst, (DWORD)std::min<size_t>(bytes, INT_MAX), &readBytes, &overlapped)) {
            return -1;
        }
        return (DiskReadResult)readBytes;
#else
        return pread(fd, dst, bytes, offset);
#endif
    }

    static void ReadDiskPartBytes(const DiskWeightPart &part, uint8_t *dst) {
        int fd = GetDiskFileCache().Get(part.fileName);
        uint64_t done = 0;
        while (done < part.bytes) {
            DiskReadResult ret = ReadDiskFileAt(fd, dst + done, part.bytes - done, part.fileOffset + done);
            if (ret < 0) {
                ErrorInFastLLM("Disk MoE read weight failed: " + part.fileName + "\n");
            }
            if (ret == 0) {
                ErrorInFastLLM("Disk MoE read EOF: " + part.fileName + "\n");
            }
            done += ret;
        }
    }

    static float BF16ToFloat(uint16_t v) {
        uint32_t u = (uint32_t)v << 16;
        return *(float*)&u;
    }

    static uint16_t FloatToBF16(float v) {
        return (uint16_t)(*(uint32_t*)&v >> 16);
    }

    static void ConvertDiskPart(uint8_t *dst, DataType dstType,
                                const uint8_t *src, DataType srcType,
                                size_t count) {
        if (dstType == srcType) {
            size_t bytes = 0;
            if (dstType == DataType::FLOAT32) {
                bytes = count * sizeof(float);
            } else if (dstType == DataType::FLOAT16 || dstType == DataType::BFLOAT16) {
                bytes = count * sizeof(uint16_t);
            }
            if (bytes > 0) {
                memcpy(dst, src, bytes);
                return;
            }
        }

        if (dstType == DataType::FLOAT32) {
            float *out = (float*)dst;
            if (srcType == DataType::FLOAT16) {
                const uint16_t *in = (const uint16_t*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = half_to_float(in[i]);
                }
                return;
            }
            if (srcType == DataType::BFLOAT16) {
                const uint16_t *in = (const uint16_t*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = BF16ToFloat(in[i]);
                }
                return;
            }
        } else if (dstType == DataType::FLOAT16) {
            uint16_t *out = (uint16_t*)dst;
            if (srcType == DataType::FLOAT32) {
                const float *in = (const float*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = float_to_half(in[i]);
                }
                return;
            }
            if (srcType == DataType::BFLOAT16) {
                const uint16_t *in = (const uint16_t*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = float_to_half(BF16ToFloat(in[i]));
                }
                return;
            }
        } else if (dstType == DataType::BFLOAT16) {
            uint16_t *out = (uint16_t*)dst;
            if (srcType == DataType::FLOAT32) {
                const float *in = (const float*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = FloatToBF16(in[i]);
                }
                return;
            }
            if (srcType == DataType::FLOAT16) {
                const uint16_t *in = (const uint16_t*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = FloatToBF16(half_to_float(in[i]));
                }
                return;
            }
        }
        ErrorInFastLLM("Disk MoE unsupported weight dtype conversion.\n");
    }

    static Data *LoadDiskWeight(const Data *weight) {
        if (weight == nullptr || !weight->isDiskWeight) {
            return (Data*)weight;
        }
        Data *loaded = new Data(weight->dataType);
        loaded->name = weight->name;
        loaded->isModelWeight = false;
        loaded->weightType = weight->weightType;
        loaded->tpLinearType = weight->tpLinearType;
        loaded->tpPackType = weight->tpPackType;
        loaded->perChannelAxis = weight->perChannelAxis;
        loaded->group = weight->group;
        loaded->groupCnt = weight->groupCnt;
        loaded->blockK = weight->blockK;
        loaded->blockM = weight->blockM;
        loaded->perChannelsConfigs = weight->perChannelsConfigs;
        loaded->scales = weight->scales;
        loaded->mins = weight->mins;
        loaded->zeros = weight->zeros;
        loaded->halfScales = weight->halfScales;
        loaded->isGGUFData = weight->isGGUFData;
        loaded->ggmlType = weight->ggmlType;
        loaded->IsRepacked = weight->IsRepacked;
        if (weight->ggmlTensor != nullptr) {
            loaded->ggmlTensor = (void*)(new ggml_tensor());
            (*(ggml_tensor*)loaded->ggmlTensor) = (*(ggml_tensor*)weight->ggmlTensor);
        }
        loaded->Resize(weight->dims);

        if (weight->dataType == DataType::DATA_GGUF_FORMAT) {
            uint64_t bytes = 0;
            for (auto &part : weight->diskWeightParts) {
                bytes += part.bytes;
            }
            loaded->expansionSize = bytes;
            loaded->expansionBytes = bytes;
            loaded->cpuData = new uint8_t[bytes];
            uint64_t dstOffset = 0;
            for (auto &part : weight->diskWeightParts) {
                ReadDiskPartBytes(part, loaded->cpuData + dstOffset);
                dstOffset += part.bytes;
            }
            // Disk weights are temporary per token; repacking them every decode step
            // costs more than the saved Q4_K_M read bandwidth. Use GGUF blocks as-is.
            loaded->IsRepacked = true;
            return loaded;
        }

        loaded->Allocate(false);

        uint64_t dstOffset = 0;
        std::vector<uint8_t> buffer;
        for (auto &part : weight->diskWeightParts) {
            uint8_t *dst = loaded->cpuData + dstOffset;
            Data partData(weight->dataType, part.dims);
            uint64_t dstBytes = partData.GetBytes();
            if (part.sourceDataType == weight->dataType && part.bytes == dstBytes) {
                ReadDiskPartBytes(part, dst);
            } else {
                buffer.resize(part.bytes);
                ReadDiskPartBytes(part, buffer.data());
                ConvertDiskPart(dst, weight->dataType, buffer.data(), part.sourceDataType, DiskPartCount(part));
            }
            dstOffset += dstBytes;
        }
        return loaded;
    }

    struct LoadedDiskWeight {
        Data *data = nullptr;
        std::shared_ptr<Data> keepAlive;
        bool owned = false;
    };

    static uint64_t DiskMoeCacheBytesLimit() {
        static uint64_t limit = []() {
            const char *env = std::getenv("FASTLLM_DISK_MOE_CACHE_MB");
            if (env == nullptr) {
                return (uint64_t)0;
            }
            long long mb = atoll(env);
            if (mb <= 0) {
                return (uint64_t)0;
            }
            return (uint64_t)mb * 1024ULL * 1024ULL;
        }();
        return limit;
    }

    static bool DiskMoeStatsEnabled() {
        static bool enabled = std::getenv("FASTLLM_DISK_MOE_STATS") != nullptr;
        return enabled;
    }

    static uint64_t DiskMoeStatsEvery() {
        static uint64_t every = []() {
            const char *env = std::getenv("FASTLLM_DISK_MOE_STATS_EVERY");
            if (env == nullptr) {
                return (uint64_t)128;
            }
            long long value = atoll(env);
            return value <= 0 ? (uint64_t)128 : (uint64_t)value;
        }();
        return every;
    }

    static bool DiskMoeProfileEnabled() {
        static bool enabled = std::getenv("FASTLLM_DISK_MOE_PROFILE") != nullptr;
        return enabled;
    }

    static uint64_t DiskMoeProfileEvery() {
        static uint64_t every = []() {
            const char *env = std::getenv("FASTLLM_DISK_MOE_PROFILE_EVERY");
            if (env == nullptr) {
                return (uint64_t)128;
            }
            long long value = atoll(env);
            return value <= 0 ? (uint64_t)128 : (uint64_t)value;
        }();
        return every;
    }

    static double DiskMoeProfileNowMs() {
        using Clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
    }

    static bool DiskMoeEndsWith(const std::string &value, const std::string &suffix) {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    static std::string DiskMoeExpertStatsName(const std::string &weightName) {
        std::string name = weightName;
        const std::vector<std::string> suffixes = {
            ".gateup.weight", ".w1.weight", ".w2.weight", ".w3.weight",
            ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"
        };
        for (auto &suffix : suffixes) {
            if (DiskMoeEndsWith(name, suffix)) {
                name.resize(name.size() - suffix.size());
                break;
            }
        }
        return name;
    }

    class DiskMoeStats {
    public:
        void RecordCall(int topk, int tokens, int routedUniqueExperts, bool hasSharedExpert,
                        int loadRequests, const std::vector<std::string> &experts) {
            if (!DiskMoeStatsEnabled()) {
                return;
            }
            std::lock_guard<std::mutex> guard(locker);
            calls++;
            totalTokens += std::max(0, tokens);
            totalTopk += std::max(0, topk);
            totalRoutedUniqueExperts += std::max(0, routedUniqueExperts);
            sharedExpertCalls += hasSharedExpert ? 1 : 0;
            totalLoadRequests += std::max(0, loadRequests);
            for (auto &expert : experts) {
                expertHits[expert]++;
            }
            MaybeReportLocked();
        }

        void RecordCacheHit(uint64_t bytes) {
            if (!DiskMoeStatsEnabled()) {
                return;
            }
            std::lock_guard<std::mutex> guard(locker);
            cacheHits++;
            cacheHitBytes += bytes;
        }

        void RecordCacheMiss(uint64_t bytes, bool inserted) {
            if (!DiskMoeStatsEnabled()) {
                return;
            }
            std::lock_guard<std::mutex> guard(locker);
            cacheMisses++;
            cacheMissBytes += bytes;
            if (inserted) {
                cacheInsertions++;
                currentCacheBytes += bytes;
                peakCacheBytes = std::max(peakCacheBytes, currentCacheBytes);
            }
        }

        void RecordCacheBypass(uint64_t bytes) {
            if (!DiskMoeStatsEnabled()) {
                return;
            }
            std::lock_guard<std::mutex> guard(locker);
            cacheBypasses++;
            cacheBypassBytes += bytes;
        }

        void RecordCacheEvict(uint64_t bytes) {
            if (!DiskMoeStatsEnabled()) {
                return;
            }
            std::lock_guard<std::mutex> guard(locker);
            cacheEvictions++;
            currentCacheBytes = currentCacheBytes > bytes ? currentCacheBytes - bytes : 0;
        }

    private:
        void MaybeReportLocked() {
            uint64_t every = DiskMoeStatsEvery();
            if (every == 0 || calls == 0 || calls % every != 0) {
                return;
            }
            ReportLocked();
        }

        void ReportLocked() {
            uint64_t cacheLookups = cacheHits + cacheMisses;
            double hitRate = cacheLookups == 0 ? 0.0 : (double)cacheHits * 100.0 / (double)cacheLookups;
            double avgTopk = calls == 0 ? 0.0 : (double)totalTopk / (double)calls;
            double avgRoutedUnique = calls == 0 ? 0.0 : (double)totalRoutedUniqueExperts / (double)calls;
            double avgLoadRequests = calls == 0 ? 0.0 : (double)totalLoadRequests / (double)calls;

            std::vector<std::pair<std::string, uint64_t>> topExperts(expertHits.begin(), expertHits.end());
            std::sort(topExperts.begin(), topExperts.end(),
                      [](const auto &a, const auto &b) {
                          if (a.second != b.second) {
                              return a.second > b.second;
                          }
                          return a.first < b.first;
                      });

            printf("[fastllm-disk-moe-stats] calls=%llu tokens=%llu avgTopk=%.2f avgRoutedUnique=%.2f sharedCalls=%llu avgLoadWeights=%.2f cacheHit=%llu cacheMiss=%llu hitRate=%.2f%% inserted=%llu evicted=%llu cacheMB=%.1f peakCacheMB=%.1f missReadMB=%.1f topExperts=",
                   (unsigned long long)calls,
                   (unsigned long long)totalTokens,
                   avgTopk,
                   avgRoutedUnique,
                   (unsigned long long)sharedExpertCalls,
                   avgLoadRequests,
                   (unsigned long long)cacheHits,
                   (unsigned long long)cacheMisses,
                   hitRate,
                   (unsigned long long)cacheInsertions,
                   (unsigned long long)cacheEvictions,
                   (double)currentCacheBytes / 1024.0 / 1024.0,
                   (double)peakCacheBytes / 1024.0 / 1024.0,
                   (double)cacheMissBytes / 1024.0 / 1024.0);
            int limit = std::min((int)topExperts.size(), 8);
            for (int i = 0; i < limit; i++) {
                printf("%s%s:%llu", i == 0 ? "" : ";",
                       topExperts[i].first.c_str(),
                       (unsigned long long)topExperts[i].second);
            }
            printf("\n");
            fflush(stdout);
        }

        std::mutex locker;
        uint64_t calls = 0;
        uint64_t totalTokens = 0;
        uint64_t totalTopk = 0;
        uint64_t totalRoutedUniqueExperts = 0;
        uint64_t sharedExpertCalls = 0;
        uint64_t totalLoadRequests = 0;
        uint64_t cacheHits = 0;
        uint64_t cacheMisses = 0;
        uint64_t cacheBypasses = 0;
        uint64_t cacheInsertions = 0;
        uint64_t cacheEvictions = 0;
        uint64_t cacheHitBytes = 0;
        uint64_t cacheMissBytes = 0;
        uint64_t cacheBypassBytes = 0;
        uint64_t currentCacheBytes = 0;
        uint64_t peakCacheBytes = 0;
        std::unordered_map<std::string, uint64_t> expertHits;
    };

    static DiskMoeStats &GetDiskMoeStats() {
        static DiskMoeStats stats;
        return stats;
    }

    class DiskMoeProfile {
    public:
        void Record(double selectMs, double loadMs, double prepareMs,
                    double inputMs, double mergeMs, double outputMs, double cleanupMs,
                    int loadRequests) {
            if (!DiskMoeProfileEnabled()) {
                return;
            }
            std::lock_guard<std::mutex> guard(locker);
            calls++;
            totalSelectMs += selectMs;
            totalLoadMs += loadMs;
            totalPrepareMs += prepareMs;
            totalInputMs += inputMs;
            totalMergeMs += mergeMs;
            totalOutputMs += outputMs;
            totalCleanupMs += cleanupMs;
            totalLoadRequests += std::max(0, loadRequests);
            uint64_t every = DiskMoeProfileEvery();
            if (every != 0 && calls % every == 0) {
                ReportLocked();
            }
        }

    private:
        void ReportLocked() {
            double totalMs = totalSelectMs + totalLoadMs + totalPrepareMs +
                             totalInputMs + totalMergeMs + totalOutputMs + totalCleanupMs;
            double callsDouble = calls == 0 ? 1.0 : (double)calls;
            printf("[fastllm-disk-moe-profile] calls=%llu avgLoadWeights=%.2f avgMs=%.3f select=%.3f load=%.3f prepare=%.3f input=%.3f merge=%.3f output=%.3f cleanup=%.3f totalMs=%.3f\n",
                   (unsigned long long)calls,
                   (double)totalLoadRequests / callsDouble,
                   totalMs / callsDouble,
                   totalSelectMs / callsDouble,
                   totalLoadMs / callsDouble,
                   totalPrepareMs / callsDouble,
                   totalInputMs / callsDouble,
                   totalMergeMs / callsDouble,
                   totalOutputMs / callsDouble,
                   totalCleanupMs / callsDouble,
                   totalMs);
            fflush(stdout);
        }

        std::mutex locker;
        uint64_t calls = 0;
        uint64_t totalLoadRequests = 0;
        double totalSelectMs = 0.0;
        double totalLoadMs = 0.0;
        double totalPrepareMs = 0.0;
        double totalInputMs = 0.0;
        double totalMergeMs = 0.0;
        double totalOutputMs = 0.0;
        double totalCleanupMs = 0.0;
    };

    static DiskMoeProfile &GetDiskMoeProfile() {
        static DiskMoeProfile profile;
        return profile;
    }

    static std::string DiskWeightCacheKey(const Data *weight) {
        std::string key = weight->name + "|" + std::to_string((int)weight->dataType) +
                          "|" + std::to_string(weight->ggmlType);
        for (auto &part : weight->diskWeightParts) {
            key += "|" + part.fileName + ":" + std::to_string(part.fileOffset) +
                   ":" + std::to_string(part.bytes) +
                   ":" + std::to_string((int)part.sourceDataType);
        }
        return key;
    }

    class DiskWeightCache {
    public:
        ~DiskWeightCache() {
            Clear();
        }

        LoadedDiskWeight Get(const Data *weight) {
            uint64_t limit = DiskMoeCacheBytesLimit();
            if (limit == 0 || weight == nullptr || !weight->isDiskWeight) {
                LoadedDiskWeight loaded = {LoadDiskWeight(weight), nullptr, weight != nullptr && weight->isDiskWeight};
                if (weight != nullptr && weight->isDiskWeight) {
                    uint64_t bytes = loaded.data == nullptr ? 0 :
                        (loaded.data->expansionBytes != 0 ? loaded.data->expansionBytes : loaded.data->GetBytes());
                    GetDiskMoeStats().RecordCacheBypass(bytes);
                }
                return loaded;
            }
            std::string key = DiskWeightCacheKey(weight);

            {
                std::lock_guard<std::mutex> guard(locker);
                auto it = entries.find(key);
                if (it != entries.end()) {
                    uint64_t bytes = it->second.bytes;
                    std::shared_ptr<Data> data = it->second.data;
                    Touch(it);
                    GetDiskMoeStats().RecordCacheHit(bytes);
                    return {data.get(), data, false};
                }
            }

            std::unique_ptr<Data> loaded(LoadDiskWeight(weight));
            uint64_t bytes = loaded->expansionBytes != 0 ? loaded->expansionBytes : loaded->GetBytes();
            if (bytes == 0 || bytes > limit) {
                GetDiskMoeStats().RecordCacheMiss(bytes, false);
                return {loaded.release(), nullptr, true};
            }

            std::lock_guard<std::mutex> guard(locker);
            auto existing = entries.find(key);
            if (existing != entries.end()) {
                GetDiskMoeStats().RecordCacheMiss(bytes, false);
                uint64_t cachedBytes = existing->second.bytes;
                std::shared_ptr<Data> data = existing->second.data;
                Touch(existing);
                GetDiskMoeStats().RecordCacheHit(cachedBytes);
                return {data.get(), data, false};
            }

            EvictToFit(bytes, limit);
            if (currentBytes + bytes > limit) {
                GetDiskMoeStats().RecordCacheMiss(bytes, false);
                return {loaded.release(), nullptr, true};
            }

            std::shared_ptr<Data> shared(loaded.release());
            lru.push_front(key);
            entries[key] = {shared, bytes, lru.begin()};
            currentBytes += bytes;
            GetDiskMoeStats().RecordCacheMiss(bytes, true);
            return {shared.get(), shared, false};
        }

    private:
        struct Entry {
            std::shared_ptr<Data> data;
            uint64_t bytes;
            std::list<std::string>::iterator lruIt;
        };

        void Touch(std::unordered_map<std::string, Entry>::iterator it) {
            lru.erase(it->second.lruIt);
            lru.push_front(it->first);
            it->second.lruIt = lru.begin();
        }

        void EvictToFit(uint64_t bytes, uint64_t limit) {
            while (!lru.empty() && currentBytes + bytes > limit) {
                std::string key = lru.back();
                lru.pop_back();
                auto it = entries.find(key);
                if (it != entries.end()) {
                    currentBytes -= it->second.bytes;
                    GetDiskMoeStats().RecordCacheEvict(it->second.bytes);
                    entries.erase(it);
                }
            }
        }

        void Clear() {
            std::lock_guard<std::mutex> guard(locker);
            entries.clear();
            lru.clear();
            currentBytes = 0;
        }

        std::mutex locker;
        uint64_t currentBytes = 0;
        std::list<std::string> lru;
        std::unordered_map<std::string, Entry> entries;
    };

    static DiskWeightCache &GetDiskWeightCache() {
        static DiskWeightCache cache;
        return cache;
    }

    static void LoadDiskWeightInto(Data **weights, std::vector<Data*> *tempWeights,
                                   std::vector<uint8_t> *ownedFlags,
                                   std::vector<std::shared_ptr<Data>> *cachedRefs,
                                   int index) {
        LoadedDiskWeight loaded = GetDiskWeightCache().Get(weights[index]);
        (*tempWeights)[index] = loaded.data;
        (*ownedFlags)[index] = loaded.owned ? 1 : 0;
        if (loaded.keepAlive != nullptr) {
            (*cachedRefs)[index] = loaded.keepAlive;
        }
    }

    struct LoadDiskWeightsOp : MultiThreadBaseOp {
        Data **weights;
        std::vector<Data*> *tempWeights;
        std::vector<uint8_t> *ownedFlags;
        std::vector<std::shared_ptr<Data>> *cachedRefs;
        const std::vector<int> *indices;
        int tid, threadCnt;

        LoadDiskWeightsOp(Data **weights, std::vector<Data*> *tempWeights,
                          std::vector<uint8_t> *ownedFlags,
                          std::vector<std::shared_ptr<Data>> *cachedRefs,
                          const std::vector<int> *indices, int tid, int threadCnt) :
            weights(weights), tempWeights(tempWeights), ownedFlags(ownedFlags),
            cachedRefs(cachedRefs), indices(indices), tid(tid), threadCnt(threadCnt) {}

        void Run() {
            for (int i = tid; i < (int)indices->size(); i += threadCnt) {
                int index = (*indices)[i];
                LoadDiskWeightInto(weights, tempWeights, ownedFlags, cachedRefs, index);
            }
        }
    };

    static void ConvertInputToFloat32(const Data &input, Data &output) {
        output.dataType = DataType::FLOAT32;
        output.Resize(input.dims);
        output.Allocate(false);
        int len = input.Count(0);
        float *dst = (float*)output.cpuData;
        if (input.dataType == DataType::FLOAT32) {
            memcpy(dst, input.cpuData, input.GetBytes());
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *src = (uint16_t*)input.cpuData;
            for (int i = 0; i < len; i++) {
                dst[i] = half_to_float(src[i]);
            }
        } else if (input.dataType == DataType::BFLOAT16) {
            uint16_t *src = (uint16_t*)input.cpuData;
            for (int i = 0; i < len; i++) {
                dst[i] = BF16ToFloat(src[i]);
            }
        } else {
            ErrorInFastLLM("Disk MoE only supports FLOAT32/FLOAT16/BFLOAT16 input for quantized weights.\n");
        }
    }

    static void ConvertFloat32ToOutput(const Data &input, Data &output, DataType outputType) {
        output.dataType = outputType;
        output.Resize(input.dims);
        output.Allocate(false);
        int len = input.Count(0);
        float *src = (float*)input.cpuData;
        if (outputType == DataType::FLOAT32) {
            memcpy(output.cpuData, input.cpuData, input.GetBytes());
        } else if (outputType == DataType::FLOAT16) {
            uint16_t *dst = (uint16_t*)output.cpuData;
            for (int i = 0; i < len; i++) {
                dst[i] = float_to_half(src[i]);
            }
        } else if (outputType == DataType::BFLOAT16) {
            uint16_t *dst = (uint16_t*)output.cpuData;
            for (int i = 0; i < len; i++) {
                dst[i] = FloatToBF16(src[i]);
            }
        } else {
            ErrorInFastLLM("Disk MoE only supports FLOAT32/FLOAT16/BFLOAT16 output for quantized weights.\n");
        }
    }

    bool DiskMergeMOE::CanRun(const std::string &opType, const DataDict &datas,
                              const FloatDict &floatParams, const IntDict &intParams) {
        auto weightIt = datas.find("weights");
        if (weightIt == datas.end()) {
            return false;
        }
        Data **weights = (Data**)weightIt->second;
        if (weights == nullptr || weights[2] == nullptr) {
            return false;
        }
        auto biasIt = datas.find("biass");
        if (biasIt != datas.end()) {
            Data **biass = (Data**)biasIt->second;
            if (biass != nullptr && biass[0] != nullptr && biass[0]->dims.size() > 0) {
                return false;
            }
        }
        return weights[2]->isDiskWeight;
    }

    void DiskMergeMOE::Run(const std::string &opType, const DataDict &datas,
                           const FloatDict &floatParams, const IntDict &intParams) {
        bool profileDiskMoe = DiskMoeProfileEnabled();
        double profileLast = profileDiskMoe ? DiskMoeProfileNowMs() : 0.0;
        double profileSelectMs = 0.0;
        double profileLoadMs = 0.0;
        double profilePrepareMs = 0.0;
        double profileInputMs = 0.0;
        double profileMergeMs = 0.0;
        double profileOutputMs = 0.0;
        double profileCleanupMs = 0.0;
        auto profileLap = [&](double &bucket) {
            if (!profileDiskMoe) {
                return;
            }
            double now = DiskMoeProfileNowMs();
            bucket += now - profileLast;
            profileLast = now;
        };

        Data &index = *(datas.find("index")->second);
        Data **weights = (Data**)datas.find("weights")->second;
        int topk = index.dims[1];
        int weightsBatch = intParams.find("weights___batch") != intParams.end() ?
            intParams.find("weights___batch")->second : (topk + 1) * 2;
        if (std::getenv("FASTLLM_DEBUG_DISK_MOE") != nullptr) {
            printf("[fastllm-disk-moe] enter topk=%d weightsBatch=%d w0=%p w2=%p w2Disk=%d w2Type=%d\n",
                   topk, weightsBatch, weights[0], weights[2],
                   weights[2] == nullptr ? -1 : (int)weights[2]->isDiskWeight,
                   weights[2] == nullptr ? -1 : (int)weights[2]->dataType);
            fflush(stdout);
        }

        std::set<int> selectedExperts;
        int32_t *indexData = (int32_t*)index.cpuData;
        int routedExpertCount = std::max(0, weightsBatch / 2 - 1);
        for (int i = 0; i < index.dims[0] * topk; i++) {
            int expertIdx = routedExpertCount <= 0 ? 0 : std::max(0, std::min(indexData[i], routedExpertCount - 1));
            selectedExperts.insert(expertIdx + 1);
        }
        if (weights[0] != nullptr) {
            selectedExperts.insert(0);
        }

        std::vector<Data*> tempWeights(weightsBatch, nullptr);
        std::vector<Data*> ownedWeights;
        std::vector<std::shared_ptr<Data>> cachedWeightRefs;
        for (int i = 0; i < weightsBatch; i++) {
            tempWeights[i] = weights[i];
        }
        std::vector<int> loadIndices;
        for (int expert : selectedExperts) {
            int gate = expert * 2;
            int down = gate + 1;
            if (gate >= weightsBatch || down >= weightsBatch || weights[gate] == nullptr || weights[down] == nullptr) {
                continue;
            }
            if (weights[gate]->isDiskWeight) {
                loadIndices.push_back(gate);
            }
            if (weights[down]->isDiskWeight) {
                loadIndices.push_back(down);
            }
        }
        if (DiskMoeStatsEnabled()) {
            std::vector<std::string> expertNames;
            bool hasSharedExpert = selectedExperts.find(0) != selectedExperts.end();
            for (int expert : selectedExperts) {
                if (expert <= 0) {
                    continue;
                }
                int gate = expert * 2;
                if (gate < weightsBatch && weights[gate] != nullptr) {
                    expertNames.push_back(DiskMoeExpertStatsName(weights[gate]->name));
                }
            }
            int routedUnique = (int)selectedExperts.size() - (hasSharedExpert ? 1 : 0);
            GetDiskMoeStats().RecordCall(topk, index.dims[0], routedUnique, hasSharedExpert,
                                         (int)loadIndices.size(), expertNames);
        }
        profileLap(profileSelectMs);
        if (loadIndices.size() > 0) {
            std::vector<uint8_t> ownedFlags(weightsBatch, 0);
            std::vector<std::shared_ptr<Data>> cachedRefs(weightsBatch);
            auto *pool = GetAlivePool();
            int threadCnt = std::min((int)loadIndices.size(), DiskMoeLoadThreads());
            threadCnt = std::min(threadCnt, (int)pool->threads.size());
            if (threadCnt <= 1) {
                for (int index : loadIndices) {
                    LoadDiskWeightInto(weights, &tempWeights, &ownedFlags, &cachedRefs, index);
                }
            } else {
                std::vector<LoadDiskWeightsOp*> ops;
                for (int i = 0; i < threadCnt; i++) {
                    ops.push_back(new LoadDiskWeightsOp(weights, &tempWeights, &ownedFlags, &cachedRefs, &loadIndices, i, threadCnt));
                    pool->PushOp(i, ops.back());
                }
                for (int i = 0; i < threadCnt; i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            }
            for (int index : loadIndices) {
                if (ownedFlags[index]) {
                    ownedWeights.push_back(tempWeights[index]);
                }
                if (cachedRefs[index] != nullptr) {
                    cachedWeightRefs.push_back(cachedRefs[index]);
                }
            }
        }
        profileLap(profileLoadMs);
        if (std::getenv("FASTLLM_DEBUG_DISK_MOE") != nullptr) {
            printf("[fastllm-disk-moe] loaded count=%d temp0=%p temp2=%p temp2Disk=%d temp2Type=%d temp2Dims=%d:%d,%d\n",
                   (int)loadIndices.size(), tempWeights[0], tempWeights[2],
                   tempWeights[2] == nullptr ? -1 : (int)tempWeights[2]->isDiskWeight,
                   tempWeights[2] == nullptr ? -1 : (int)tempWeights[2]->dataType,
                   tempWeights[2] == nullptr ? -1 : (int)tempWeights[2]->dims.size(),
                   tempWeights[2] == nullptr || tempWeights[2]->dims.size() < 1 ? -1 : tempWeights[2]->dims[0],
                   tempWeights[2] == nullptr || tempWeights[2]->dims.size() < 2 ? -1 : tempWeights[2]->dims[1]);
            fflush(stdout);
        }
        for (int i = 0; i < weightsBatch; i++) {
            if (tempWeights[i] != nullptr && tempWeights[i]->isDiskWeight) {
                tempWeights[i] = nullptr;
            }
        }
        if (tempWeights[2] == nullptr) {
            for (int expert : selectedExperts) {
                int gate = expert * 2;
                if (gate < weightsBatch && tempWeights[gate] != nullptr) {
                    // CpuMergeMOE uses weights[2] only as the representative dtype/shape
                    // when expert 0 is not selected. Avoid loading expert 0 just for that.
                    tempWeights[2] = tempWeights[gate];
                    break;
                }
            }
        }
        if (tempWeights[2] == nullptr) {
            ErrorInFastLLM("Disk MoE failed to load representative expert weight.\n");
        }
        profileLap(profilePrepareMs);

        DataDict diskDatas = datas;
        diskDatas["weights"] = (Data*)tempWeights.data();
        Data promotedInput, promotedOutput;
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        DataType originalOutputType = output.dataType;
        bool promoteInput = tempWeights[2] != nullptr &&
                            (tempWeights[2]->dataType == DataType::BFLOAT16 ||
                             tempWeights[2]->dataType == DataType::FP8_E4M3 ||
                             tempWeights[2]->dataType == DataType::NVFP4) &&
                            input.dataType != DataType::FLOAT32;
        if (promoteInput) {
            ConvertInputToFloat32(input, promotedInput);
            promotedOutput.dataType = DataType::FLOAT32;
            promotedOutput.Resize(output.dims);
            diskDatas["input"] = &promotedInput;
            diskDatas["output"] = &promotedOutput;
        }
        profileLap(profileInputMs);
        CpuMergeMOE::Run(opType, diskDatas, floatParams, intParams);
        profileLap(profileMergeMs);
        if (std::getenv("FASTLLM_DEBUG_DISK_MOE") != nullptr) {
            printf("[fastllm-disk-moe] cpu-merge-done\n");
            fflush(stdout);
        }
        if (promoteInput) {
            ConvertFloat32ToOutput(promotedOutput, output, originalOutputType);
        }
        profileLap(profileOutputMs);

        for (auto *weight : ownedWeights) {
            delete weight;
        }
        profileLap(profileCleanupMs);
        GetDiskMoeProfile().Record(profileSelectMs, profileLoadMs, profilePrepareMs,
                                   profileInputMs, profileMergeMs, profileOutputMs,
                                   profileCleanupMs, (int)loadIndices.size());
    }
}
