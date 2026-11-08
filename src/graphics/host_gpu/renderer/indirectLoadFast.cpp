#include "graphics/host_gpu/renderer/indirectLoadFast.h"

#include "common/liveAb.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace Libs::Graphics {

namespace IndirectLoadFast {

int Mode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_PERF_INDIRECT_LOAD_FAST");
		const int   m     = value != nullptr ? std::atoi(value) : 1; // default on (=2: 4.7e10 checks / 0 mismatches)
		if (m != 0) std::fprintf(stderr, "PERF_INDIRECT_LOAD_FAST mode=%d\n", m);
		return m;
	}();
	return mode;
}

bool CensusEnabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_INDIRECT_FORMAT_CENSUS");
		const bool  b     = value != nullptr && value[0] == '1';
		if (b) std::fprintf(stderr, "DIAG_INDIRECT_FORMAT_CENSUS enabled\n");
		return b;
	}();
	return on;
}

int DiagFixedFormat(uint64_t shader_hash, uint32_t channels) {
	// KYTY_DIAG_INDIRECT_FIXED_FORMAT=<format for 1 channel>/<for 2>/<for 3>/<for 4> (decimal, 0 = keep the switch) for the
	// shaders in KYTY_DIAG_INDIRECT_FIXED_FORMAT_HASHES: measurement only, APPROXIMATION (a lane with another format is
	// decoded wrongly) - the GPU time of the decoders.
	struct Config {
		std::string hashes;
		int         formats[4] {};
	};
	static const Config config = [] {
		Config c;
		if (const char* v = std::getenv("KYTY_DIAG_INDIRECT_FIXED_FORMAT_HASHES"); v != nullptr) c.hashes = v;
		if (const char* v = std::getenv("KYTY_DIAG_INDIRECT_FIXED_FORMAT"); v != nullptr) {
			const char* p = v;
			for (int i = 0; i < 4 && *p != 0; ++i) {
				char* end    = nullptr;
				c.formats[i] = static_cast<int>(std::strtol(p, &end, 10));
				p            = (*end == '/') ? end + 1 : end;
			}
		}
		if (!c.hashes.empty()) {
			std::fprintf(stderr, "DIAG_INDIRECT_FIXED_FORMAT %d/%d/%d/%d for %s (measurement only)\n", c.formats[0], c.formats[1],
			             c.formats[2], c.formats[3], c.hashes.c_str());
		}
		return c;
	}();
	if (config.hashes.empty() || channels < 1u || channels > 4u || config.formats[channels - 1u] <= 0) return -1;
	char text[17];
	std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(shader_hash));
	return config.hashes.find(text) != std::string::npos ? config.formats[channels - 1u] : -1;
}

bool CensusFor(uint64_t shader_hash) {
	// KYTY_DIAG_INDIRECT_FORMAT_CENSUS_HASHES: the shaders counted (default: all).
	static const std::string list = [] {
		const char* value = std::getenv("KYTY_DIAG_INDIRECT_FORMAT_CENSUS_HASHES");
		return value != nullptr ? std::string(value) : std::string("all");
	}();
	if (!CensusEnabled()) return false;
	if (list == "all") return true;
	char text[17];
	std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(shader_hash));
	return list.find(text) != std::string::npos;
}

bool Listed(uint64_t shader_hash) {
	// Default: the probe-evaluation shaders of "Deferred Ambient>Eval non-RT Probes" (2.65 ms of GPU time per frame at
	// the torii gate).
	static const std::string list = [] {
		const char* value = std::getenv("KYTY_PERF_INDIRECT_LOAD_FAST_HASHES");
		return value != nullptr ? std::string(value)
		                        : std::string("85312e99a3c69cef,3ab9b653825476ab,8368b02a85651dd8,2b018f33db5ffffe,"
		                                      "40395313615abcc8");
	}();
	if (list == "all") return true;
	char text[17];
	std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(shader_hash));
	return list.find(text) != std::string::npos;
}

} // namespace IndirectLoadFast

namespace SpirvLive {

bool FormattedCaseLive() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_FORMATTED_CASE_LIVE");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

bool Enabled() {
	static const bool on = IndirectLoadFast::Mode() == 3 || FormattedCaseLive();
	return on;
}

uint32_t Flags() {
	uint32_t flags = 0;
	if (IndirectLoadFast::Mode() == 3 && Common::LiveAb::Gate(Common::LiveAb::Switch::IndirectLoadFast, 1u) != 0u) {
		flags |= FlagIndirectLoadFast;
	}
	if (FormattedCaseLive() && Common::LiveAb::Gate(Common::LiveAb::Switch::DiagFormattedCaseLive, 1u) != 0u) {
		flags |= FlagFormattedCase;
	}
	return flags;
}

void Update(CommandScheduler& scheduler, Buffer& fault_buffer, uint32_t word_index) {
	static uint64_t last = ~uint64_t {0};
	if (!Enabled()) return;
	const uint32_t flags = Flags();
	if (flags == last) return;
	last = flags;
	scheduler.EndRendering();
	auto                     command = scheduler.Current().Handle();
	const vk::DeviceSize     offset  = uint64_t {word_index} * sizeof(uint32_t);
	vk::BufferMemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
	before.buffer        = fault_buffer.Handle();
	before.offset        = offset;
	before.size          = sizeof(uint32_t);
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);
	command.updateBuffer(fault_buffer.Handle(), offset, sizeof(uint32_t), &flags);
	auto after          = before;
	after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	after.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	dependency.pBufferMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
	std::fprintf(stderr, "SPIRV_LIVE flags=0x%x\n", flags);
}

} // namespace SpirvLive

} // namespace Libs::Graphics
