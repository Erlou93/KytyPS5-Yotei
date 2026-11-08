#include "graphics/host_gpu/renderer/pipeline/reuseDisk.h"

#include "graphics/shader/recompiler/ir/passes/SrtTape.h"

#include <xxhash.h>
#include <zstd.h>

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
constexpr uint32_t FileVersion  = 3u;
constexpr uint32_t FrameMagic   = 0x45525546u; // "FURE"
constexpr uint32_t MaxWords     = 64u << 20u;  // 256 MiB per frame: a damaged size field
constexpr uint32_t HeadBytes    = 40u;         // {magic, key (2), bytes, program (2), flags, words, hash (2)}
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

namespace {

void PutShaderInfo(Out& o, const IR::ShaderInfo& s) {
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
}

void GetShaderInfo(In& i, IR::ShaderInfo& s) {
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
	PutShaderInfo(o, info.info);
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
	GetShaderInfo(i, info.info);
	auto& b = info.bindings;
	for (auto& count: b.descriptor_counts) count = i.U32();
	b.descriptor_mask       = i.U64();
	b.push_data_start_dword = i.U32();
	b.dispatch_thread_dword = i.U32();
	b.memory_offset_dword   = i.U32();
	b.memory_stride_dword   = i.U32();
	return i.ok && i.AtEnd();
}


// ---- The resource plan (Z2) ----
namespace {

// A Value: the type tag, then an instruction index (Opaque) or the immediate's bits (anything else but Void).
struct ValueOut {
	Out&                                          o;
	const std::unordered_map<const IR::Inst*, uint32_t>& index;
	bool                                          ok = true;
	void operator()(const IR::Value& value) {
		const auto type = value.RawType();
		o.Enum(type);
		if (type == IR::Type::Void) return;
		if (type == IR::Type::Opaque) {
			const auto found = index.find(value.Instruction());
			if (found == index.end()) {
				ok = false;
				o.U32(UINT32_MAX);
				return;
			}
			o.U32(found->second);
			return;
		}
		o.U64(value.ImmediateBits());
	}
};

struct ValueIn {
	In&                          i;
	const std::vector<IR::Inst*>& insts;
	IR::Value operator()() {
		const auto type = i.Enum<IR::Type>();
		if (type == IR::Type::Void) return {};
		if (type == IR::Type::Opaque) {
			const auto at = i.U32();
			if (at >= insts.size()) {
				i.ok = false;
				return {};
			}
			return IR::Value(insts[at]);
		}
		return IR::Value::FromImmediateBits(type, i.U64());
	}
};

void PutMemory(Out& o, const IR::MemoryInfo& m) {
	o.Enum(m.kind);
	o.U32(m.resource);
	o.U32(m.sampler);
	o.U32(m.offset);
	o.U32(m.secondary_offset);
	o.U32(m.dmask);
	o.U32(m.data_dwords);
	o.U32(m.data_bits);
	o.U32(m.component_index);
	o.U32(m.component_count);
	o.U32(m.data_format);
	o.U32(m.number_format);
	o.U32(m.image_sample_flags);
	o.Enum(m.image_dimension);
	o.U32(m.image_address_components);
	for (const bool flag: {m.address_is_full, m.data_signed, m.typed, m.formatted, m.image_has_mip, m.image_r128, m.idxen,
	                       m.offen, m.coherent, m.planning_only, m.record_keyed})
		o.Bool(flag);
}

void GetMemory(In& i, IR::MemoryInfo& m) {
	m.kind                     = i.Enum<IR::ResourceKind>();
	m.resource                 = i.U32();
	m.sampler                  = i.U32();
	m.offset                   = i.U32();
	m.secondary_offset         = i.U32();
	m.dmask                    = i.U32();
	m.data_dwords              = i.U32();
	m.data_bits                = i.U32();
	m.component_index          = i.U32();
	m.component_count          = i.U32();
	m.data_format              = i.U32();
	m.number_format            = i.U32();
	m.image_sample_flags       = i.U32();
	m.image_dimension          = i.Enum<ShaderRecompiler::Decoder::ImageDimension>();
	m.image_address_components = i.U32();
	for (bool* flag: {&m.address_is_full, &m.data_signed, &m.typed, &m.formatted, &m.image_has_mip, &m.image_r128,
	                  &m.idxen, &m.offen, &m.coherent, &m.planning_only, &m.record_keyed})
		*flag = i.Bool();
}

using Indirect = IR::DescriptorSource::IndirectDescriptor;

void PutIndirect(Out& o, ValueOut& v, const Indirect& d) {
	o.Bool(d.selector.has_value());
	if (d.selector.has_value()) {
		const auto& s = *d.selector;
		o.U32(s.source);
		o.U32(s.stride);
		o.U32(s.offset);
		o.U32(s.maximum);
		o.U32(s.bit_offset);
		o.U32(s.bit_mask);
		o.Bool(s.global);
		o.Bool(s.bitset);
		o.Bool(s.rows.has_value());
		if (s.rows.has_value()) {
			o.U32(s.rows->source);
			o.U32(s.rows->stride);
		}
		o.Pairs(s.byte_ranges);
	}
	o.Bool(d.table_maximum.has_value());
	if (d.table_maximum.has_value()) o.U32(*d.table_maximum);
	o.Bool(d.pointer_table.has_value());
	if (d.pointer_table.has_value()) {
		o.U32(d.pointer_table->offset);
		o.U32(d.pointer_table->stride);
		o.U32(d.pointer_table->count);
		o.Bool(d.pointer_table->indexed);
	}
	o.U32(d.table_source);
	o.U32(d.table_offset);
	o.U32(d.table_immediate);
	o.U32(d.table_stride);
	o.U32(d.table_record_bytes);
	o.Bool(d.table_scalar);
	o.Bool(d.indexed);
	o.Bool(d.signed_count);
	o.U32(d.workgroup_axis);
	o.Bool(d.all_records);
	o.U32(d.pointer_offset);
	o.Bool(d.record_keyed);
	o.Bool(d.unsigned_bound);
	o.Bool(d.widened);
	o.U32(d.count_pointer);
	o.U32(d.count_field);
	o.U32(d.count_check);
	o.U32(d.count_limit);
	o.Bool(d.composed);
	v(d.key_count);
	v(d.selector_first);
	v(d.selector_mask);
	o.Words(d.sources);
}

void GetIndirect(In& i, ValueIn& v, Indirect& d) {
	if (i.Bool()) {
		auto& s      = d.selector.emplace();
		s.source     = i.U32();
		s.stride     = i.U32();
		s.offset     = i.U32();
		s.maximum    = i.U32();
		s.bit_offset = i.U32();
		s.bit_mask   = i.U32();
		s.global     = i.Bool();
		s.bitset     = i.Bool();
		if (i.Bool()) {
			auto& rows  = s.rows.emplace();
			rows.source = i.U32();
			rows.stride = i.U32();
		}
		s.byte_ranges = i.Pairs();
	}
	if (i.Bool()) d.table_maximum = i.U32();
	if (i.Bool()) {
		auto& t   = d.pointer_table.emplace();
		t.offset  = i.U32();
		t.stride  = i.U32();
		t.count   = i.U32();
		t.indexed = i.Bool();
	}
	d.table_source       = i.U32();
	d.table_offset       = i.U32();
	d.table_immediate    = i.U32();
	d.table_stride       = i.U32();
	d.table_record_bytes = i.U32();
	d.table_scalar       = i.Bool();
	d.indexed            = i.Bool();
	d.signed_count       = i.Bool();
	d.workgroup_axis     = i.U32();
	d.all_records        = i.Bool();
	d.pointer_offset     = i.U32();
	d.record_keyed       = i.Bool();
	d.unsigned_bound     = i.Bool();
	d.widened            = i.Bool();
	d.count_pointer      = i.U32();
	d.count_field        = i.U32();
	d.count_check        = i.U32();
	d.count_limit        = i.U32();
	d.composed           = i.Bool();
	d.key_count          = v();
	d.selector_first     = v();
	d.selector_mask      = v();
	d.sources            = i.Words();
}

} // namespace

bool SerializePlan(const IR::ResourcePlan& plan, std::vector<uint32_t>& out) {
	if (!plan.control_phi_predecessors.empty()) return false;
	std::unordered_map<const IR::Inst*, uint32_t> index;
	for (const auto& inst: plan.value_storage) index.emplace(&inst, static_cast<uint32_t>(index.size()));
	Out      o(out);
	ValueOut v {o, index};
	o.Enum(plan.stage);
	o.U64(plan.shader_hash);
	o.U32(plan.user_data_base);
	o.U32(plan.user_data_count);
	// The graph: every instruction's opcode and flags, then every instruction's arguments.
	o.U32(static_cast<uint32_t>(plan.value_storage.size()));
	for (const auto& inst: plan.value_storage) {
		o.Enum(inst.GetOpcode());
		o.U64(inst.Flags<uint64_t>());
	}
	for (const auto& inst: plan.value_storage) {
		o.U32(static_cast<uint32_t>(inst.NumArgs()));
		for (size_t arg = 0; arg < inst.NumArgs(); arg++) v(inst.Arg(arg));
	}
	o.U32(static_cast<uint32_t>(plan.memory_info.size()));
	for (const auto& memory: plan.memory_info) PutMemory(o, memory);
	o.U32(static_cast<uint32_t>(plan.descriptor_sources.size()));
	for (const auto& source: plan.descriptor_sources) {
		o.U32(source.dword_count);
		for (const auto& dword: source.dwords) v(dword);
		o.Bool(source.indirect_descriptor.has_value());
		if (source.indirect_descriptor.has_value()) PutIndirect(o, v, *source.indirect_descriptor);
	}
	o.Words(plan.selector_groups);
	o.U32(static_cast<uint32_t>(plan.control_flow.size()));
	for (const auto& block: plan.control_flow) {
		v(block.condition);
		o.Words(block.successors);
		o.Words(block.sources);
		o.Words(block.srt_reads);
	}
	o.U32(static_cast<uint32_t>(plan.srt_reads.size()));
	for (const auto& read: plan.srt_reads) {
		v(read.value);
		o.U32(read.flat_offset);
	}
	o.Bool(plan.requires_specialization_memory);
	o.U32(static_cast<uint32_t>(plan.source_reads.size()));
	for (const auto& [first, second]: plan.source_reads) {
		o.U64(first);
		o.U64(second);
	}
	o.Bool(plan.capture_specialization_reads);
	o.Bool(plan.srt_plan_complete);
	o.Bool(plan.resource_tracking_complete);
	PutShaderInfo(o, plan.info);
	const auto& fill = plan.uniform_fill;
	o.Enum(fill.fill.kind);
	o.U32(fill.fill.resource);
	for (const auto stride: fill.fill.group_stride) o.U32(stride);
	o.U32(fill.fill.words);
	o.U32(fill.fill.value);
	o.U32(fill.fill.bounded_elements);
	for (const auto& value: fill.values) v(value);
	v(fill.pattern_size);
	v(fill.element_count);
	o.Bool(plan.srt_tape != nullptr);
	return v.ok;
}

bool DeserializePlan(std::span<const uint32_t> words, IR::ResourcePlan& plan) {
	In i(words);
	plan.stage           = i.Enum<ShaderType>();
	plan.shader_hash     = i.U64();
	plan.user_data_base  = i.U32();
	plan.user_data_count = i.U32();
	const auto count     = i.U32();
	if (!i.ok || count > MaxWords) return false;
	std::vector<IR::Inst*> insts;
	insts.reserve(count);
	for (uint32_t n = 0; n < count && i.ok; n++) {
		const auto opcode = i.Enum<IR::ValueOpcode>();
		const auto flags  = i.U64();
		if (static_cast<uint32_t>(opcode) >= static_cast<uint32_t>(IR::ValueOpcode::Count)) return false;
		insts.push_back(&plan.value_storage.emplace_back(opcode, flags));
	}
	ValueIn v {i, insts};
	for (auto* inst: insts) {
		const auto args = i.U32();
		if (!i.ok) return false;
		const bool phi = inst->GetOpcode() == IR::ValueOpcode::Phi;
		if (!phi && args != inst->NumArgs()) return false;
		for (uint32_t arg = 0; arg < args && i.ok; arg++) {
			const auto value = v();
			if (phi) {
				inst->AddPhiOperand(nullptr, value);
			} else {
				inst->SetArg(arg, value);
			}
		}
	}
	plan.memory_info.resize(i.Count());
	for (auto& memory: plan.memory_info) GetMemory(i, memory);
	plan.descriptor_sources.resize(i.Count());
	for (auto& source: plan.descriptor_sources) {
		source.dword_count = i.U32();
		for (auto& dword: source.dwords) dword = v();
		if (i.Bool()) GetIndirect(i, v, source.indirect_descriptor.emplace());
	}
	plan.selector_groups = i.Words();
	plan.control_flow.resize(i.Count());
	for (auto& block: plan.control_flow) {
		block.condition  = v();
		block.successors = i.Words();
		block.sources    = i.Words();
		block.srt_reads  = i.Words();
	}
	plan.srt_reads.resize(i.Count());
	for (auto& read: plan.srt_reads) {
		read.value       = v();
		read.flat_offset = i.U32();
	}
	plan.requires_specialization_memory = i.Bool();
	plan.source_reads.resize(i.Count());
	for (auto& [first, second]: plan.source_reads) {
		first  = i.U64();
		second = i.U64();
	}
	plan.capture_specialization_reads = i.Bool();
	plan.srt_plan_complete            = i.Bool();
	plan.resource_tracking_complete   = i.Bool();
	GetShaderInfo(i, plan.info);
	auto& fill                  = plan.uniform_fill;
	fill.fill.kind              = i.Enum<IR::UniformFillKind>();
	fill.fill.resource          = i.U32();
	for (auto& stride: fill.fill.group_stride) stride = i.U32();
	fill.fill.words             = i.U32();
	fill.fill.value             = i.U32();
	fill.fill.bounded_elements  = i.U32();
	for (auto& value: fill.values) value = v();
	fill.pattern_size  = v();
	fill.element_count = v();
	const bool tape    = i.Bool();
	if (!i.ok || !i.AtEnd()) return false;
	if (tape) plan.srt_tape = IR::BuildSrtTape(plan);
	return true;
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
	// The layout of the serialized structures enters the identity (a field added changes a size; the =2 check and the
	// round trip catch the rest).
	const uint64_t sizes[] = {sizeof(IR::CompiledShaderInfo), sizeof(IR::ShaderInfo), sizeof(IR::Bindings),
	                          sizeof(IR::BufferResource),     sizeof(IR::ImageResource), sizeof(IR::SamplerResource),
	                          sizeof(IR::IndirectRangeResource), sizeof(IR::StageInput), sizeof(IR::StageOutput),
	                          sizeof(IR::ResourcePlan),       sizeof(IR::MemoryInfo), sizeof(IR::DescriptorSource),
	                          sizeof(IR::DescriptorSource::IndirectDescriptor), sizeof(IR::ResourceBlock),
	                          sizeof(IR::UniformFillPlan),    sizeof(IR::Inst), FileVersion};
	const uint64_t words[] = {identity, XXH3_64bits(sizes, sizeof(sizes))};
	const uint64_t full    = XXH3_64bits(words, sizeof(words));
	std::error_code ec;
	std::filesystem::create_directories(m_path.parent_path(), ec);
	m_file         = std::fopen(m_path.string().c_str(), "r+b");
	bool     fresh = m_file == nullptr;
	uint32_t magic = 0, version = 0;
	uint64_t stored = 0;
	if (!fresh) {
		fresh = std::fread(&magic, 4, 1, m_file) != 1 || std::fread(&version, 4, 1, m_file) != 1 ||
		        std::fread(&stored, 8, 1, m_file) != 1 || magic != FileMagic || version != FileVersion || stored != full;
	}
	size_t damaged = 0;
	if (!fresh) {
		// Index the frames: {magic, key (2), stored bytes, program (2), flags, payload words, hash (2)} then the payload
		// (zstd-compressed, padded to 4 bytes).
		uint64_t pos = 16;
		for (;;) {
			uint32_t head[10] {};
			if (_fseeki64(m_file, static_cast<int64_t>(pos), SEEK_SET) != 0 || std::fread(head, 4, 10, m_file) != 10) break;
			const uint64_t key     = head[1] | (static_cast<uint64_t>(head[2]) << 32u);
			const uint32_t bytes   = head[3];
			const uint64_t program = head[4] | (static_cast<uint64_t>(head[5]) << 32u);
			const uint32_t count   = head[7];
			if (head[0] != FrameMagic || bytes == 0 || bytes > MaxWords * 4u || count == 0 || count > MaxWords) {
				damaged++;
				break;
			}
			const uint64_t payload = pos + HeadBytes;
			const uint64_t end     = payload + ((uint64_t {bytes} + 3u) & ~uint64_t {3});
			if (_fseeki64(m_file, static_cast<int64_t>(end - 1), SEEK_SET) != 0 || std::fgetc(m_file) == EOF) {
				damaged++; // a torn tail
				break;
			}
			const Frame frame {payload, bytes, count};
			m_index[key] = frame;
			if ((head[6] & 1u) != 0u) m_plans.try_emplace(program, frame);
			pos = end;
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
		m_plans.clear();
	}
	std::fprintf(stderr, "SHADER_REUSE_DISK file=%s %s entries=%zu plans=%zu bytes=%llu%s\n", m_path.string().c_str(),
	             fresh ? "started" : "loaded", m_index.size(), m_plans.size(), static_cast<unsigned long long>(m_end),
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

bool ReuseDisk::ContainsPlan(uint64_t program) const {
	std::lock_guard lock(m_mutex);
	return m_plans.contains(program);
}

size_t ReuseDisk::Entries() const {
	std::lock_guard lock(m_mutex);
	return m_index.size();
}

size_t ReuseDisk::Plans() const {
	std::lock_guard lock(m_mutex);
	return m_plans.size();
}

bool ReuseDisk::ReadPayload(const Frame& frame, std::vector<uint32_t>& payload) {
	uint64_t             hash = 0;
	std::vector<uint8_t> stored(frame.bytes);
	if (m_file == nullptr || _fseeki64(m_file, static_cast<int64_t>(frame.offset - 8), SEEK_SET) != 0 ||
	    std::fread(&hash, 8, 1, m_file) != 1 || std::fread(stored.data(), 1, stored.size(), m_file) != stored.size()) {
		return false;
	}
	if (XXH3_64bits(stored.data(), stored.size()) != hash) return false;
	payload.resize(frame.words);
	const size_t size = ZSTD_decompress(payload.data(), payload.size() * 4u, stored.data(), stored.size());
	return !ZSTD_isError(size) && size == payload.size() * 4u;
}

bool ReuseDisk::SplitPayload(std::span<const uint32_t> payload, Payload& out) {
	size_t     pos  = 0;
	const auto take = [&](std::span<const uint32_t>& part) {
		if (pos >= payload.size()) return false;
		const auto count = payload[pos++];
		if (pos + count > payload.size()) return false;
		part = payload.subspan(pos, count);
		pos += count;
		return true;
	};
	if (!take(out.spirv) || !take(out.info) || !take(out.plan) || pos + 2 != payload.size()) return false;
	out.translate_us = payload[pos];
	out.compile_us   = payload[pos + 1];
	return true;
}

bool ReuseDisk::Load(uint64_t key, std::vector<uint32_t>& spirv, IR::CompiledShaderInfo& info, double& translate_ms,
                     double& compile_ms) {
	std::vector<uint32_t> payload;
	{
		std::lock_guard lock(m_mutex);
		const auto      found = m_index.find(key);
		if (found == m_index.end() || !ReadPayload(found->second, payload)) return false;
	}
	Payload parts;
	if (!SplitPayload(payload, parts) || parts.spirv.empty()) return false;
	spirv.assign(parts.spirv.begin(), parts.spirv.end());
	translate_ms = static_cast<double>(parts.translate_us) / 1000.0;
	compile_ms   = static_cast<double>(parts.compile_us) / 1000.0;
	return DeserializeCompiledInfo(parts.info, info);
}

bool ReuseDisk::LoadPlan(uint64_t program, IR::ResourcePlan& plan) {
	std::vector<uint32_t> payload;
	{
		std::lock_guard lock(m_mutex);
		const auto      found = m_plans.find(program);
		if (found == m_plans.end() || !ReadPayload(found->second, payload)) return false;
	}
	Payload parts;
	if (!SplitPayload(payload, parts) || parts.plan.empty()) return false;
	return DeserializePlan(parts.plan, plan);
}

void ReuseDisk::Store(uint64_t key, uint64_t program, std::span<const uint32_t> spirv, const IR::CompiledShaderInfo& info,
                      const IR::ResourcePlan* plan, double translate_ms, double compile_ms) {
	std::vector<uint32_t> info_words, plan_words;
	SerializeCompiledInfo(info, info_words);
	if (plan != nullptr && !SerializePlan(*plan, plan_words)) plan_words.clear();
	std::vector<uint32_t> payload;
	payload.reserve(spirv.size() + info_words.size() + plan_words.size() + 5);
	Out o(payload);
	o.Words(spirv);
	o.Words(info_words);
	o.Words(plan_words);
	o.U32(static_cast<uint32_t>(translate_ms * 1000.0));
	o.U32(static_cast<uint32_t>(compile_ms * 1000.0));
	// zstd level 3 on the worker (outside the lock): SPIR-V and the serialized structures shrink several times.
	std::vector<uint8_t> stored(ZSTD_compressBound(payload.size() * 4u) + 4u);
	const size_t         bytes = ZSTD_compress(stored.data(), stored.size(), payload.data(), payload.size() * 4u, 3);
	if (ZSTD_isError(bytes)) return;
	stored.resize((bytes + 3u) & ~size_t {3});
	const uint64_t hash     = XXH3_64bits(stored.data(), bytes);
	const bool     has_plan = !plan_words.empty();
	uint32_t       head[8] {FrameMagic, static_cast<uint32_t>(key), static_cast<uint32_t>(key >> 32u),
	                        static_cast<uint32_t>(bytes), static_cast<uint32_t>(program),
	                        static_cast<uint32_t>(program >> 32u), has_plan ? 1u : 0u,
	                        static_cast<uint32_t>(payload.size())};
	std::lock_guard lock(m_mutex);
	if (m_file == nullptr) return;
	if (m_index.contains(key)) {
		// Only a plan the file lacks is added: under a key of its own (no permutation looks it up).
		if (!has_plan || m_plans.contains(program)) return;
		key ^= 0x9e3779b97f4a7c15ull;
		if (m_index.contains(key)) return;
		head[1] = static_cast<uint32_t>(key);
		head[2] = static_cast<uint32_t>(key >> 32u);
	}
	if (_fseeki64(m_file, static_cast<int64_t>(m_end), SEEK_SET) != 0) return;
	const bool ok = std::fwrite(head, 4, 8, m_file) == 8 && std::fwrite(&hash, 8, 1, m_file) == 1 &&
	                std::fwrite(stored.data(), 1, stored.size(), m_file) == stored.size();
	if (!ok) return;
	const Frame frame {m_end + HeadBytes, static_cast<uint32_t>(bytes), static_cast<uint32_t>(payload.size())};
	m_index[key] = frame;
	if (has_plan) m_plans.try_emplace(program, frame);
	m_end += HeadBytes + stored.size();
	std::fflush(m_file); // the normal quit (std::quick_exit) closes no file
}

} // namespace Libs::Graphics
