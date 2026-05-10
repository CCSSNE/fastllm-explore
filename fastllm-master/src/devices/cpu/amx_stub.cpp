#include <cstdint>

namespace fastllm {
#ifdef _WIN32
    void InitAMX() {}

    bool LinearBFloat16BFloat16_AMX_Kernel(uint16_t *, uint16_t *, float *, float *,
                                           int, int, int, int, int) {
        return false;
    }
#endif
}
