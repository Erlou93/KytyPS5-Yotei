#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_REUSEDISK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_REUSEDISK_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// KYTY_PERF_SHADER_REUSE_DISK (test switch of the "zapis tłumaczeń" project, AGENTS 18; 0 = off, 1 = on, 2 = check):
// the background precompiler's translations (the shader reuse store: SPIR-V and program info of each recorded
// permutation) are kept in _PipelineCache/<title>.reuse, so a later start reads them instead of translating and
// compiling the same permutations again (31-38 s on 8 threads every start before). =2: every permutation found on
// disk is also translated and compiled as before; the fresh result is used and compared word for word with the disk's
// (PERF shader_reuse_disk checks= mismatches=; must stay 0).
[[nodiscard]] int ShaderReuseDiskMode();

// CompiledShaderInfo as words (every field; the round trip is checked against operator== in the compute tests).
void SerializeCompiledInfo(const ShaderRecompiler::IR::CompiledShaderInfo& info, std::vector<uint32_t>& out);
[[nodiscard]] bool DeserializeCompiledInfo(std::span<const uint32_t> words, ShaderRecompiler::IR::CompiledShaderInfo& info);

// The file: a header (magic, version, identity) and frames {magic, key, payload words, XXH3 of the payload, payload}.
// The identity covers what makes a translation: the emulator executable itself (its bytes: two uncommitted builds have
// the same git revision), the SPIR-V-changing settings (EnvironmentFingerprint), the GPU and driver, and the layout of
// the serialized structures. Another identity: the file is started again. Payload: SPIR-V words, the serialized info,
// the worker's translate and compile microseconds.
class ReuseDisk {
public:
	ReuseDisk(std::filesystem::path path, uint64_t identity);
	~ReuseDisk();
	ReuseDisk(const ReuseDisk&)            = delete;
	ReuseDisk& operator=(const ReuseDisk&) = delete;

	[[nodiscard]] bool   Contains(uint64_t key) const;
	[[nodiscard]] size_t Entries() const;
	// False when the frame is unreadable or damaged (the caller translates as before).
	bool Load(uint64_t key, std::vector<uint32_t>& spirv, ShaderRecompiler::IR::CompiledShaderInfo& info,
	          double& translate_ms, double& compile_ms);
	// Appends a frame unless the key is already in the file.
	void Store(uint64_t key, std::span<const uint32_t> spirv, const ShaderRecompiler::IR::CompiledShaderInfo& info,
	           double translate_ms, double compile_ms);

	// The identity of this executable (XXH3 of its file), computed once.
	[[nodiscard]] static uint64_t ExecutableHash();

private:
	struct Frame {
		uint64_t offset = 0; // of the payload
		uint32_t words  = 0;
	};
	std::filesystem::path                  m_path;
	mutable std::mutex                     m_mutex;
	std::FILE*                             m_file = nullptr;
	uint64_t                               m_end  = 0;
	std::unordered_map<uint64_t, Frame>    m_index;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_REUSEDISK_H_ */
