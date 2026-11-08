#include "graphics/host_gpu/renderer/pipeline/reuseDisk.h"

#include <xxhash.h>

#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace Libs::Graphics {

namespace IR = ShaderRecompiler::IR;

int ShaderReuseDiskMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_PERF_SHADER_REUSE_DISK");
		const int   m     = value != nullptr ? std::atoi(value) : 0;
		if (m != 0) std::fprintf(stderr, "PERF_SHADER_REUSE_DISK mode=%d\n", m);
		return m;
	}();
	return mode;
}

namespace {

constexpr uint32_t FileMagic    = 0x5552594bu; // "KYRU"
constexpr uint32_t FileVersion  = 1u;
constexpr uint32_t FrameMagic   = 0x45525546u; // "FURE"
constexpr uint32_t MaxWords     = 64u << 20u;  // 256 MiB per frame: a damaged size field
constexpr uint32_t MaxContainer = 1u << 16u;

class Out {
public:
	explicit Out(std::vector<uint32_t>& words): m_words(words) {}
	void U32(uint32_t value) { m_words.push_back(value); }
	void U64(uint64_t value) {
		U32(static_cast<uint32_t>(value));
		U32(static_cast<uint32_t>(value >> 32u));
	}
	void Bool(bool value) { U32(value ? 1u : 0u); }
	template <typename E>
	void Enum(E value) {
		U32(static_cast<uint32_t>(value));
	}
	void Words(std::span<const uint32_t> values) {
		U32(static_cast<uint32_t>(values.size()));
		m_words.insert(m_words.end(), values.begin(), values.end());
	}
	void Pairs(const std::vector<std::pair<uint32_t, uint32_t>>& values) {
		U32(static_cast<uint32_t>(values.size()));
		for (const auto& [a, b]: values) {
			U32(a);
			U32(b);
		}
	}
	void String(const std::string& value) {
		U32(static_cast<uint32_t>(value.size()));
		for (size_t i = 0; i < value.size(); i += 4) {
			uint32_t word = 0;
			std::memcpy(&word, value.data() + i, std::min<size_t>(4, value.size() - i));
			U32(word);
		}
	}

private:
	std::vector<uint32_t>& m_words;
};

class In {
public:
	explicit In(std::span<const uint32_t> words): m_words(words) {}
	uint32_t U32() {
		if (m_pos >= m_words.size()) {
			ok = false;
			return 0;
		}
		return m_words[m_pos++];
	}
	uint64_t U64() {
		const uint64_t low = U32();
		return low | (static_cast<uint64_t>(U32()) << 32u);
	}
	bool Bool() { return U32() != 0u; }
	template <typename E>
	E Enum() {
		return static_cast<E>(U32());
	}
	uint32_t Count() {
		const auto count = U32();
		if (count > MaxContainer) ok = false;
		return ok ? count : 0u;
	}
	std::vector<uint32_t> Words() {
		const auto count = U32();
		if (!ok || m_pos + count > m_words.size()) {
			ok = false;
			return {};
		}
		std::vector<uint32_t> values(m_words.begin() + static_cast<std::ptrdiff_t>(m_pos),
		                             m_words.begin() + static_cast<std::ptrdiff_t>(m_pos + count));
		m_pos += count;
		return values;
	}
	std::vector<std::pair<uint32_t, uint32_t>> Pairs() {
		std::vector<std::pair<uint32_t, uint32_t>> values(Count());
		for (auto& [a, b]: values) {
			a = U32();
			b = U32();
		}
		return values;
	}
	std::string String() {
		const auto size = U32();
		if (!ok || size > MaxContainer * 4u) {
			ok = false;
			return {};
		}
		std::string value(size, '\0');
		for (size_t i = 0; i < size; i += 4) {
			const uint32_t word = U32();
			std::memcpy(value.data() + i, &word, std::min<size_t>(4, size - i));
		}
		return value;
	}
	[[nodiscard]] bool AtEnd() const { return m_pos == m_words.size(); }
	bool ok = true;

private:
	std::span<const uint32_t> m_words;
	size_t                    m_pos = 0;
};

void Put(Out& o, const IR::BufferResource& b) {
	o.U32(b.source);
	o.U32(b.descriptor_index);
	o.U32(b.first_use_pc);
	o.U32(b.max_byte_extent);
	o.U32(b.write_extent);
	o.U32(b.packed_stride);
	o.Enum(b.descriptor_format);
	o.U32(b.descriptor_swizzle);
	o.U32(b.image_alias);
	o.Bool(b.read);
	o.Bool(b.written);
	o.Bool(b.atomic);
	o.Bool(b.formatted);
	o.Bool(b.scalar);
	o.Bool(b.fixed_writes);
	o.Pairs(b.fixed_write_ranges);
	o.U32(b.append_counter);
	o.U32(b.append_min_offset);
	o.U32(b.append_max_end);
	o.Bool(b.live);
	o.U32(b.indirect_root);
	o.U32(b.indirect_mapping_offset);
	o.U32(b.indirect_search_iterations);
	o.Words(b.indirect_resources);
}

void Get(In& i, IR::BufferResource& b) {
	b.source             = i.U32();
	b.descriptor_index   = i.U32();
	b.first_use_pc       = i.U32();
	b.max_byte_extent    = i.U32();
	b.write_extent       = i.U32();
	b.packed_stride      = i.U32();
	b.descriptor_format  = i.Enum<Prospero::BufferFormat>();
	b.descriptor_swizzle = i.U32();
	b.image_alias        = i.U32();
	b.read               = i.Bool();
	b.written            = i.Bool();
	b.atomic             = i.Bool();
	b.formatted          = i.Bool();
	b.scalar             = i.Bool();
	b.fixed_writes       = i.Bool();
	b.fixed_write_ranges = i.Pairs();
	b.append_counter     = i.U32();
	b.append_min_offset  = i.U32();
	b.append_max_end     = i.U32();
	b.live               = i.Bool();
	b.indirect_root              = i.U32();
	b.indirect_mapping_offset    = i.U32();
	b.indirect_search_iterations = i.U32();
	b.indirect_resources         = i.Words();
}

void Put(Out& o, const IR::ImageResource& m) {
	o.U32(m.source);
	o.U32(m.descriptor_index);
	o.U32(m.first_use_pc);
	o.Enum(m.resource_class);
	o.Enum(m.numeric_class);
	o.Enum(m.dimension);
	o.Enum(m.mip_mode);
	o.U32(m.mip_count);
	o.Enum(m.conversion_format);
	o.U32(m.shader_swizzle);
	o.Bool(m.read);
	o.Bool(m.written);
	o.Bool(m.atomic);
	o.Bool(m.atomic64);
	o.Bool(m.unfiltered_access_only);
	o.Bool(m.depth_compare);
	o.Bool(m.cube);
	o.Bool(m.r128);
	o.Bool(m.heap);
	o.U32(m.indirect_root);
	o.U32(m.indirect_mapping_offset);
	o.U32(m.indirect_search_iterations);
	o.Bool(m.bindless);
	o.U32(m.bindless_stride_dwords);
	o.Words(m.indirect_resources);
	o.Pairs(m.bindless_fields);
	o.Bool(m.bindless_fields_complete);
	o.Bool(m.bindless_field_split);
}

void Get(In& i, IR::ImageResource& m) {
	m.source                     = i.U32();
	m.descriptor_index           = i.U32();
	m.first_use_pc               = i.U32();
	m.resource_class             = i.Enum<IR::ImageResourceClass>();
	m.numeric_class              = i.Enum<Prospero::TextureNumericClass>();
	m.dimension                  = i.Enum<ShaderRecompiler::Decoder::ImageDimension>();
	m.mip_mode                   = i.Enum<IR::ImageMipMode>();
	m.mip_count                  = i.U32();
	m.conversion_format          = i.Enum<Prospero::BufferFormat>();
	m.shader_swizzle             = i.U32();
	m.read                       = i.Bool();
	m.written                    = i.Bool();
	m.atomic                     = i.Bool();
	m.atomic64                   = i.Bool();
	m.unfiltered_access_only     = i.Bool();
	m.depth_compare              = i.Bool();
	m.cube                       = i.Bool();
	m.r128                       = i.Bool();
	m.heap                       = i.Bool();
	m.indirect_root              = i.U32();
	m.indirect_mapping_offset    = i.U32();
	m.indirect_search_iterations = i.U32();
	m.bindless                   = i.Bool();
	m.bindless_stride_dwords     = i.U32();
	m.indirect_resources         = i.Words();
	m.bindless_fields            = i.Pairs();
	m.bindless_fields_complete   = i.Bool();
	m.bindless_field_split       = i.Bool();
}

void Put(Out& o, const IR::SamplerResource& s) {
	o.U32(s.source);
	o.U32(s.first_use_pc);
	o.U32(s.snapshot_index);
	o.Bool(s.force_point_filtering);
	o.Bool(s.depth_compare);
	o.Bool(s.integer_border);
	o.Bool(s.gather_lod);
	o.U32(s.indirect_root);
	o.U32(s.indirect_mapping_offset);
	o.U32(s.indirect_search_iterations);
	o.Words(s.indirect_resources);
}

void Get(In& i, IR::SamplerResource& s) {
	s.source                     = i.U32();
	s.first_use_pc               = i.U32();
	s.snapshot_index             = i.U32();
	s.force_point_filtering      = i.Bool();
	s.depth_compare              = i.Bool();
	s.integer_border             = i.Bool();
	s.gather_lod                 = i.Bool();
	s.indirect_root              = i.U32();
	s.indirect_mapping_offset    = i.U32();
	s.indirect_search_iterations = i.U32();
	s.indirect_resources         = i.Words();
}

void Put(Out& o, const IR::SampledResourcePair& p) {
	o.U32(p.image);
	o.U32(p.sampler);
	o.U32(p.first_use_pc);
}

void Get(In& i, IR::SampledResourcePair& p) {
	p.image        = i.U32();
	p.sampler      = i.U32();
	p.first_use_pc = i.U32();
}

void Put(Out& o, const IR::IndirectRangeResource& r) {
	o.U32(r.source);
	o.U32(r.first_use_pc);
	o.U32(r.access_begin);
	o.U64(r.access_end);
	o.Bool(r.indexed);
	o.Bool(r.read);
	o.Bool(r.written);
	o.Bool(r.atomic);
}

void Get(In& i, IR::IndirectRangeResource& r) {
	r.source       = i.U32();
	r.first_use_pc = i.U32();
	r.access_begin = i.U32();
	r.access_end   = i.U64();
	r.indexed      = i.Bool();
	r.read         = i.Bool();
	r.written      = i.Bool();
	r.atomic       = i.Bool();
}

void Put(Out& o, const IR::StageInput& s) {
	o.Enum(s.kind);
	o.U32(s.location);
	o.U32(s.component_count);
	o.String(s.debug_name);
	o.Bool(s.per_vertex);
}

void Get(In& i, IR::StageInput& s) {
	s.kind            = i.Enum<IR::StageInputKind>();
	s.location        = i.U32();
	s.component_count = i.U32();
	s.debug_name      = i.String();
	s.per_vertex      = i.Bool();
}

void Put(Out& o, const IR::StageOutput& s) {
	o.Enum(s.kind);
	o.U32(s.index);
	o.U32(s.location);
	o.String(s.debug_name);
}

void Get(In& i, IR::StageOutput& s) {
	s.kind       = i.Enum<IR::StageOutputKind>();
	s.index      = i.U32();
	s.location   = i.U32();
	s.debug_name = i.String();
}

template <typename T>
void PutVector(Out& o, const std::vector<T>& values) {
	o.U32(static_cast<uint32_t>(values.size()));
	for (const auto& value: values) Put(o, value);
}

template <typename T>
void GetVector(In& i, std::vector<T>& values) {
	values.resize(i.Count());
	for (auto& value: values) Get(i, value);
}

// The ShaderInfo flags, in one place for both directions.
template <typename Fn>
void ForEachFlag(IR::ShaderInfo& s, Fn&& fn) {
	for (bool* flag: {&s.has_bitwise_xor, &s.uses_dma, &s.uses_swizzle, &s.uses_lds, &s.uses_gds, &s.uses_flattened_srt,
	                  &s.uses_read_const, &s.uses_image_heap, &s.uses_dispatch_threads, &s.bvh, &s.bvh8, &s.bvh_stack,
	                  &s.helper_invocation, &s.ordered_lds_atomics, &s.subgroup_ballot, &s.subgroup_vote,
	                  &s.subgroup_barrier, &s.subgroup_shuffle, &s.subgroup_arithmetic, &s.subgroup_local_invocation_id,
	                  &s.compute_derivatives, &s.image_gather_extended, &s.function_lds, &s.function_scratch,
	                  &s.pixel_valid_mask, &s.buffer_int64_atomics, &s.buffer_u8, &s.buffer_u16, &s.shared_int64_atomics,
	                  &s.coherent_buffers, &s.buffer_writes, &s.coherent_narrow, &s.float64}) {
		fn(*flag);
	}
}

} // namespace

void SerializeCompiledInfo(const IR::CompiledShaderInfo& info, std::vector<uint32_t>& out) {
	Out o(out);
	o.Enum(info.stage);
	o.U64(info.shader_hash);
	o.U32(info.wave_size);
	o.U32(info.user_data_base);
	o.U32(info.user_data_count);
	o.U32(info.scratch_dwords);
	o.U32(info.param_export_mask);
	o.Bool(info.has_address_writes);
	const auto& s = info.info;
	PutVector(o, s.buffers);
	PutVector(o, s.images);
	PutVector(o, s.samplers);
	PutVector(o, s.sampled_pairs);
	PutVector(o, s.indirect_ranges);
	PutVector(o, s.inputs);
	PutVector(o, s.outputs);
	o.Words(s.user_data_registers);
	for (const auto component: s.vertex_fetch_components) o.U32(component);
	o.U32(static_cast<uint32_t>(s.vertex_offset_sgpr));
	o.U32(static_cast<uint32_t>(s.instance_offset_sgpr));
	auto flags = s; // ForEachFlag takes a mutable ShaderInfo
	ForEachFlag(flags, [&](bool& flag) { o.Bool(flag); });
	const auto& b = info.bindings;
	for (const auto count: b.descriptor_counts) o.U32(count);
	o.U64(b.descriptor_mask);
	o.U32(b.push_data_start_dword);
	o.U32(b.dispatch_thread_dword);
	o.U32(b.memory_offset_dword);
	o.U32(b.memory_stride_dword);
}

bool DeserializeCompiledInfo(std::span<const uint32_t> words, IR::CompiledShaderInfo& info) {
	In i(words);
	info.stage              = i.Enum<ShaderType>();
	info.shader_hash        = i.U64();
	info.wave_size          = i.U32();
	info.user_data_base     = i.U32();
	info.user_data_count    = i.U32();
	info.scratch_dwords     = i.U32();
	info.param_export_mask  = i.U32();
	info.has_address_writes = i.Bool();
	auto& s                 = info.info;
	GetVector(i, s.buffers);
	GetVector(i, s.images);
	GetVector(i, s.samplers);
	GetVector(i, s.sampled_pairs);
	GetVector(i, s.indirect_ranges);
	GetVector(i, s.inputs);
	GetVector(i, s.outputs);
	s.user_data_registers = i.Words();
	for (auto& component: s.vertex_fetch_components) component = static_cast<uint8_t>(i.U32());
	s.vertex_offset_sgpr   = static_cast<int32_t>(i.U32());
	s.instance_offset_sgpr = static_cast<int32_t>(i.U32());
	ForEachFlag(s, [&](bool& flag) { flag = i.Bool(); });
	auto& b = info.bindings;
	for (auto& count: b.descriptor_counts) count = i.U32();
	b.descriptor_mask       = i.U64();
	b.push_data_start_dword = i.U32();
	b.dispatch_thread_dword = i.U32();
	b.memory_offset_dword   = i.U32();
	b.memory_stride_dword   = i.U32();
	return i.ok && i.AtEnd();
}

uint64_t ReuseDisk::ExecutableHash() {
	static const uint64_t hash = [] {
		uint64_t result = 0;
#ifdef _WIN32
		wchar_t path[MAX_PATH] {};
		const auto length = GetModuleFileNameW(nullptr, path, MAX_PATH);
		if (length != 0 && length < MAX_PATH) {
			if (std::FILE* f = _wfopen(path, L"rb")) {
				std::vector<uint8_t> bytes;
				uint8_t              chunk[1u << 16u];
				size_t               read = 0;
				while ((read = std::fread(chunk, 1, sizeof(chunk), f)) != 0) bytes.insert(bytes.end(), chunk, chunk + read);
				std::fclose(f);
				result = XXH3_64bits(bytes.data(), bytes.size());
			}
		}
#endif
		return result;
	}();
	return hash;
}

ReuseDisk::ReuseDisk(std::filesystem::path path, uint64_t identity): m_path(std::move(path)) {
	// The layout of the serialized structures enters the identity (a field added changes a size; the compute tests'
	// round trip catches the rest).
	const uint64_t sizes[] = {sizeof(IR::CompiledShaderInfo), sizeof(IR::ShaderInfo), sizeof(IR::Bindings),
	                          sizeof(IR::BufferResource),     sizeof(IR::ImageResource), sizeof(IR::SamplerResource),
	                          sizeof(IR::IndirectRangeResource), sizeof(IR::StageInput), sizeof(IR::StageOutput),
	                          FileVersion};
	const uint64_t words[] = {identity, XXH3_64bits(sizes, sizeof(sizes))};
	const uint64_t full    = XXH3_64bits(words, sizeof(words));
	std::error_code ec;
	std::filesystem::create_directories(m_path.parent_path(), ec);
	m_file       = std::fopen(m_path.string().c_str(), "r+b");
	bool  fresh  = m_file == nullptr;
	uint32_t magic = 0, version = 0;
	uint64_t stored = 0;
	if (!fresh) {
		fresh = std::fread(&magic, 4, 1, m_file) != 1 || std::fread(&version, 4, 1, m_file) != 1 ||
		        std::fread(&stored, 8, 1, m_file) != 1 || magic != FileMagic || version != FileVersion || stored != full;
	}
	size_t damaged = 0;
	if (!fresh) {
		// Index the frames: {magic, key (2), words, hash (2)} then the payload.
		uint64_t pos = 16;
		for (;;) {
			uint32_t head[6] {};
			if (_fseeki64(m_file, static_cast<int64_t>(pos), SEEK_SET) != 0 || std::fread(head, 4, 6, m_file) != 6) break;
			const uint64_t key   = head[1] | (static_cast<uint64_t>(head[2]) << 32u);
			const uint32_t count = head[3];
			if (head[0] != FrameMagic || count == 0 || count > MaxWords) {
				damaged++;
				break;
			}
			const uint64_t payload = pos + 24;
			const uint64_t end     = payload + uint64_t {count} * 4u;
			if (_fseeki64(m_file, static_cast<int64_t>(end - 1), SEEK_SET) != 0 || std::fgetc(m_file) == EOF) {
				damaged++; // a torn tail
				break;
			}
			m_index[key] = {payload, count};
			pos          = end;
		}
		m_end = pos;
	}
	if (fresh) {
		if (m_file != nullptr) std::fclose(m_file);
		m_file = std::fopen(m_path.string().c_str(), "w+b");
		if (m_file != nullptr) {
			std::fwrite(&FileMagic, 4, 1, m_file);
			std::fwrite(&FileVersion, 4, 1, m_file);
			std::fwrite(&full, 8, 1, m_file);
			std::fflush(m_file);
		}
		m_end = 16;
		m_index.clear();
	}
	std::fprintf(stderr, "SHADER_REUSE_DISK file=%s %s entries=%zu bytes=%llu%s\n", m_path.string().c_str(),
	             fresh ? "started" : "loaded", m_index.size(), static_cast<unsigned long long>(m_end),
	             damaged != 0 ? " (torn tail dropped)" : "");
	std::fflush(stderr);
}

ReuseDisk::~ReuseDisk() {
	if (m_file != nullptr) std::fclose(m_file);
}

bool ReuseDisk::Contains(uint64_t key) const {
	std::lock_guard lock(m_mutex);
	return m_index.contains(key);
}

size_t ReuseDisk::Entries() const {
	std::lock_guard lock(m_mutex);
	return m_index.size();
}

bool ReuseDisk::Load(uint64_t key, std::vector<uint32_t>& spirv, IR::CompiledShaderInfo& info, double& translate_ms,
                     double& compile_ms) {
	std::vector<uint32_t> payload;
	uint64_t              hash = 0;
	{
		std::lock_guard lock(m_mutex);
		const auto      found = m_index.find(key);
		if (found == m_index.end() || m_file == nullptr) return false;
		payload.resize(found->second.words);
		if (_fseeki64(m_file, static_cast<int64_t>(found->second.offset - 8), SEEK_SET) != 0 ||
		    std::fread(&hash, 8, 1, m_file) != 1 ||
		    std::fread(payload.data(), 4, payload.size(), m_file) != payload.size()) {
			return false;
		}
	}
	if (XXH3_64bits(payload.data(), payload.size() * 4u) != hash) return false;
	In i(payload);
	spirv = i.Words();
	const auto info_words = i.Words();
	translate_ms          = static_cast<double>(i.U32()) / 1000.0;
	compile_ms            = static_cast<double>(i.U32()) / 1000.0;
	if (!i.ok || !i.AtEnd() || spirv.empty()) return false;
	return DeserializeCompiledInfo(info_words, info);
}

void ReuseDisk::Store(uint64_t key, std::span<const uint32_t> spirv, const IR::CompiledShaderInfo& info,
                      double translate_ms, double compile_ms) {
	std::vector<uint32_t> info_words;
	SerializeCompiledInfo(info, info_words);
	std::vector<uint32_t> payload;
	payload.reserve(spirv.size() + info_words.size() + 4);
	Out o(payload);
	o.Words(spirv);
	o.Words(info_words);
	o.U32(static_cast<uint32_t>(translate_ms * 1000.0));
	o.U32(static_cast<uint32_t>(compile_ms * 1000.0));
	const uint64_t hash = XXH3_64bits(payload.data(), payload.size() * 4u);
	const uint32_t head[4] {FrameMagic, static_cast<uint32_t>(key), static_cast<uint32_t>(key >> 32u),
	                        static_cast<uint32_t>(payload.size())};
	std::lock_guard lock(m_mutex);
	if (m_file == nullptr || m_index.contains(key)) return;
	if (_fseeki64(m_file, static_cast<int64_t>(m_end), SEEK_SET) != 0) return;
	const bool ok = std::fwrite(head, 4, 4, m_file) == 4 && std::fwrite(&hash, 8, 1, m_file) == 1 &&
	                std::fwrite(payload.data(), 4, payload.size(), m_file) == payload.size();
	if (!ok) return;
	m_index[key] = {m_end + 24, static_cast<uint32_t>(payload.size())};
	m_end += 24 + uint64_t {payload.size()} * 4u;
	std::fflush(m_file); // the normal quit (std::quick_exit) closes no file
}

} // namespace Libs::Graphics
