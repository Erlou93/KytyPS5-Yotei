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

// The "zapis tłumaczeń" project (AGENTS 18): the background precompiler's translations (the shader reuse store: SPIR-V,
// program info and each program's resource plan) are kept in _PipelineCache/<title>.reuse (zstd frames), so a later start
// reads them instead of translating and compiling the same permutations again (8.11: start 33-35 s on 8 threads -> 3-4 s;
// file ~416 MB; 9,760 permutations and 8,908 plans compared word for word, 0 differences). Returns 1, or 2 with
// KYTY_DIAG_SHADER_REUSE_DISK_CHECK=1 (diagnostic): every permutation found on disk is also translated and compiled as
// before, the fresh result is used and compared word for word (PERF shader_reuse_disk mismatches= plan_mismatches=;
// must stay 0).
[[nodiscard]] int ShaderReuseDiskMode();

// CompiledShaderInfo as words (every field; the round trip is checked against operator== in the compute tests).
void SerializeCompiledInfo(const ShaderRecompiler::IR::CompiledShaderInfo& info, std::vector<uint32_t>& out);
[[nodiscard]] bool DeserializeCompiledInfo(std::span<const uint32_t> words, ShaderRecompiler::IR::CompiledShaderInfo& info);
// A program's resource plan as words (project step Z2): every retained field, its value graph (value_storage) numbered
// in list order and rebuilt in two passes (instructions first, then their arguments; phis through AddPhiOperand), the SRT
// tape built again (BuildSrtTape), evaluation scratch fresh. False when the plan holds what has no saved form (a value of
// an instruction outside its storage, control_phi_predecessors): it is then not saved.
[[nodiscard]] bool SerializePlan(const ShaderRecompiler::IR::ResourcePlan& plan, std::vector<uint32_t>& out);
[[nodiscard]] bool DeserializePlan(std::span<const uint32_t> words, ShaderRecompiler::IR::ResourcePlan& plan);

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
	[[nodiscard]] bool   ContainsPlan(uint64_t program) const;
	[[nodiscard]] size_t Entries() const;
	[[nodiscard]] size_t Plans() const;
	// False when the frame is unreadable or damaged (the caller translates as before).
	bool Load(uint64_t key, std::vector<uint32_t>& spirv, ShaderRecompiler::IR::CompiledShaderInfo& info,
	          double& translate_ms, double& compile_ms);
	// The saved plan of `program` (the frame of whichever permutation carried it); false: none or damaged.
	bool LoadPlan(uint64_t program, ShaderRecompiler::IR::ResourcePlan& plan);
	// Appends a frame unless the key is already in the file; `plan`: the program's plan when this permutation carries it.
	void Store(uint64_t key, uint64_t program, std::span<const uint32_t> spirv,
	           const ShaderRecompiler::IR::CompiledShaderInfo& info, const ShaderRecompiler::IR::ResourcePlan* plan,
	           double translate_ms, double compile_ms);

	// The identity of this executable (XXH3 of its file), computed once.
	[[nodiscard]] static uint64_t ExecutableHash();

private:
	struct Frame {
		uint64_t offset = 0; // of the stored (zstd-compressed) payload
		uint32_t bytes  = 0; // stored bytes
		uint32_t words  = 0; // payload words once decompressed
	};
	// The payload words of `frame` (checked), under m_mutex.
	bool ReadPayload(const Frame& frame, std::vector<uint32_t>& payload);
	// payload: program (2), SPIR-V, info, plan (empty: none), translate us, compile us.
	struct Payload {
		std::span<const uint32_t> spirv, info, plan;
		uint32_t                  translate_us = 0, compile_us = 0;
	};
	static bool SplitPayload(std::span<const uint32_t> payload, Payload& out);
	std::filesystem::path                  m_path;
	mutable std::mutex                     m_mutex;
	std::FILE*                             m_file = nullptr;
	uint64_t                               m_end  = 0;
	std::unordered_map<uint64_t, Frame>    m_index;
	std::unordered_map<uint64_t, Frame>    m_plans; // program -> the frame carrying its plan
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_REUSEDISK_H_ */
