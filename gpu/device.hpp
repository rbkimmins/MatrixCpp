#pragma once

// ==========================================================================
//  Device query
// ==========================================================================
//
// Enough to answer "is there a GPU, and is it worth using" before committing
// to anything. All of it is pure C++17 — the CUDA calls live in backend.cu.

#include <sstream>
#include <string>

#include "detail/backend.hpp"

namespace mgpu {

    using Device = detail::DeviceInfo;

    // Zero when there is no driver, no card, or CUDA is not functioning.
    // Deliberately does not throw: "no GPU" is a state a program should be
    // able to branch on, not an exception it has to catch.
    inline int deviceCount() { return detail::deviceCount(); }
    inline bool available() { return detail::deviceCount() > 0; }

    inline Device device(int i = 0) { return detail::deviceInfo(i); }
    inline void use(int i) { detail::setDevice(i); }

    // Blocks until every queued kernel has finished. Needed before timing
    // anything: launches are asynchronous, so a timer that does not sync is
    // measuring the launch, not the work.
    inline void sync() { detail::deviceSync(); }

    inline std::string versions() { return detail::backendVersions(); }

    // ── Device memory pool ─────────────────────────────────────────────
    //
    // Freed device buffers are cached and handed back to the next allocation
    // of the same size instead of returning to the driver. cudaMalloc costs
    // tens of microseconds no matter how small the request, which is more than
    // an element-wise kernel takes at moderate sizes -- see backend.cu for the
    // measurement that prompted this.
    //
    // This is the DRIVER's stream-ordered pool (cudaMallocAsync), not one of
    // ours. A hand-rolled pool is a trap: cudaFree implicitly synchronises the
    // device, so recycling a buffer by hand can hand it to a new kernel while
    // the old one is still writing it. The driver's pool tracks that ordering
    // properly. backend.cu records the failure that made the point.
    //
    // It manages itself: it retains up to a quarter of the card and trims on
    // an allocation failure, so it can never be the reason you run out of
    // memory. poolRelease() exists for handing the card to another library.

    using PoolStats = detail::PoolStats;
    inline PoolStats poolStats() { return detail::poolStats(); }
    inline void poolRelease() { detail::poolRelease(); }

    // Human-readable summary, for the top of a benchmark run.
    inline std::string describe(int i = 0) {
        if (!available()) return "no CUDA device available";
        Device d = device(i);
        std::ostringstream os;
        os << d.name << "  sm_" << d.major << d.minor << "  " << d.smCount << " SMs  "
           << (d.totalMem >> 20) << " MB (" << (d.freeMem >> 20) << " MB free)  "
           << (long)d.memBandwidthGBs << " GB/s peak\n"
           << versions();
        return os.str();
    }

}  // namespace mgpu
