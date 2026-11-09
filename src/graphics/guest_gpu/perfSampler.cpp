#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "common/hostException.h"

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#include <psapi.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cwchar>
#include <map>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <tlhelp32.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#endif

namespace Libs::Graphics {

#if defined(_WIN32)

namespace {

// The exported function of a loaded module at or below `address` ("?" when none): names the system call or
// library routine a sampled thread is in (NtWaitForSingleObject, NtFreeVirtualMemory, memcpy, ...).
std::string NearestExport(uint64_t module_base, uint64_t address) {
	const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(module_base);
	if (module_base == 0 || dos->e_magic != IMAGE_DOS_SIGNATURE) return "?";
	const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(module_base + dos->e_lfanew);
	const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
	if (dir.VirtualAddress == 0) return "?";
	const auto* exports  = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(module_base + dir.VirtualAddress);
	const auto* names    = reinterpret_cast<const DWORD*>(module_base + exports->AddressOfNames);
	const auto* ordinals = reinterpret_cast<const WORD*>(module_base + exports->AddressOfNameOrdinals);
	const auto* rvas     = reinterpret_cast<const DWORD*>(module_base + exports->AddressOfFunctions);
	const auto  target   = address - module_base;
	DWORD       best_rva = 0;
	const char* best     = "?";
	for (DWORD i = 0; i < exports->NumberOfNames; i++) {
		const DWORD rva = rvas[ordinals[i]];
		if (rva <= target && rva > best_rva) {
			best_rva = rva;
			best     = reinterpret_cast<const char*>(module_base + names[i]);
		}
	}
	return best;
}

// KYTY_TRACE_PERF=1 and KYTY_TRACE_PERF_SAMPLE=all (diagnostic): one sampler thread samples every thread of the
// process that ran since its previous sample (QueryThreadCycleTime moved), about every 2 ms, for phases where the
// GPU thread waits and other threads do the work (loading). Every 5 s: the busiest threads by name, the leaf RVAs
// inside kyty_emulator.exe, the leaf module and exported routine with the nearest emulator return address on the
// stack (who called the allocator, the system call, the driver), and leaf addresses outside every module (guest
// code: resolve with the run's module map). A thread in a page fault is sampled at the faulting instruction.
void StartAllThreadsSampler() {
	static std::atomic_bool started {false};
	if (started.exchange(true)) return;
	std::thread([] {
		SetThreadDescription(GetCurrentThread(), L"perf_sampler_all");
		auto*      exe = GetModuleHandleW(nullptr);
		MODULEINFO info {};
		GetModuleInformation(GetCurrentProcess(), exe, &info, sizeof(info));
		const auto  base     = reinterpret_cast<uint64_t>(info.lpBaseOfDll);
		const auto  end      = base + info.SizeOfImage;
		const DWORD self_pid = GetCurrentProcessId();
		const DWORD self_tid = GetCurrentThreadId();
		const auto is_return = [&](uint64_t address) {
			if (address < base + 0x1000u || address >= end) return false;
			const auto* code = reinterpret_cast<const uint8_t*>(address);
			return code[-5] == 0xe8 || (code[-2] == 0xff && (code[-1] & 0x38) == 0x10) ||
			       (code[-3] == 0xff && (code[-2] & 0x38) == 0x10) ||
			       (code[-6] == 0xff && (code[-5] & 0x38) == 0x10) ||
			       (code[-7] == 0xff && (code[-6] & 0x38) == 0x10);
		};
		struct Thread {
			HANDLE       handle = nullptr;
			ULONG64      cycles = 0;
			std::wstring name;
			uint32_t     samples = 0;
		};
		std::unordered_map<DWORD, Thread>                 threads;
		std::unordered_map<uint64_t, uint32_t>            exe_leaf;
		std::unordered_map<uint64_t, uint32_t>            guest_leaf;
		std::map<std::pair<uint64_t, uint64_t>, uint32_t> module_leaf; // (leaf address, nearest exe return rva)
		constexpr size_t                                  StackBytes = 16u * 1024u;
		std::vector<uint64_t>                             stack(StackBytes / 8u);
		uint32_t samples = 0;
		auto     refresh = GetTickCount64() - 1000u;
		auto     window  = GetTickCount64();
		for (;;) {
			auto now = GetTickCount64();
			if (now - refresh >= 1000u) {
				refresh = now;
				for (auto it = threads.begin(); it != threads.end();) {
					if (WaitForSingleObject(it->second.handle, 0) == WAIT_OBJECT_0) {
						CloseHandle(it->second.handle);
						it = threads.erase(it);
					} else {
						++it;
					}
				}
				HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
				if (snapshot != INVALID_HANDLE_VALUE) {
					THREADENTRY32 entry {.dwSize = sizeof(THREADENTRY32)};
					for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
						if (entry.th32OwnerProcessID != self_pid || entry.th32ThreadID == self_tid ||
						    threads.contains(entry.th32ThreadID)) {
							continue;
						}
						HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
						                               THREAD_QUERY_LIMITED_INFORMATION | SYNCHRONIZE,
						                           FALSE, entry.th32ThreadID);
						if (handle == nullptr) continue;
						Thread thread {.handle = handle};
						PWSTR  description = nullptr;
						if (SUCCEEDED(GetThreadDescription(handle, &description)) && description != nullptr) {
							thread.name = description;
							LocalFree(description);
						}
						QueryThreadCycleTime(handle, &thread.cycles);
						threads.emplace(entry.th32ThreadID, std::move(thread));
					}
					CloseHandle(snapshot);
				}
			}
			for (auto& [tid, thread]: threads) {
				ULONG64 cycles = 0;
				if (!QueryThreadCycleTime(thread.handle, &cycles) || cycles - thread.cycles < 20000u) continue;
				thread.cycles = cycles;
				if (SuspendThread(thread.handle) == static_cast<DWORD>(-1)) continue;
				CONTEXT context {};
				context.ContextFlags = CONTEXT_CONTROL;
				const bool ok    = GetThreadContext(thread.handle, &context) != 0;
				size_t     words = 0;
				if (ok) {
					MEMORY_BASIC_INFORMATION region {};
					if (VirtualQuery(reinterpret_cast<void*>(context.Rsp), &region, sizeof(region)) != 0 &&
					    region.State == MEM_COMMIT && (region.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0) {
						const auto region_end = reinterpret_cast<uint64_t>(region.BaseAddress) + region.RegionSize;
						const auto bytes      = std::min<uint64_t>(StackBytes, region_end - context.Rsp) & ~7ull;
						std::memcpy(stack.data(), reinterpret_cast<const void*>(context.Rsp), bytes);
						words = bytes / 8u;
					}
				}
				ResumeThread(thread.handle);
				if (!ok) continue;
				samples++;
				thread.samples++;
				const auto rip = static_cast<uint64_t>(context.Rip);
				if (rip >= base && rip < end) {
					exe_leaf[rip - base]++;
					continue;
				}
				HMODULE owner = nullptr;
				GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				                   reinterpret_cast<LPCWSTR>(rip), &owner);
				if (owner == nullptr) {
					guest_leaf[rip]++;
					continue;
				}
				uint64_t caller = 0;
				for (size_t i = 0; i < words && i < 512u; i++) {
					if (is_return(stack[i])) {
						caller = stack[i] - base;
						break;
					}
				}
				module_leaf[{rip, caller}]++;
			}
			Sleep(2);
			now = GetTickCount64();
			if (now - window < 5000u) continue;
			window = now;
			std::fprintf(stderr, "PERF_SAMPLE_ALL samples=%u threads=%zu\n", samples, threads.size());
			{
				std::vector<std::pair<uint32_t, const Thread*>> busy;
				for (auto& [tid, thread]: threads) {
					if (thread.samples != 0) busy.emplace_back(tid, &thread);
				}
				std::ranges::sort(busy, [](const auto& a, const auto& b) { return a.second->samples > b.second->samples; });
				for (size_t i = 0; i < busy.size() && i < 16u; i++) {
					std::fprintf(stderr, "PERF_SAMPLE_ALL thread tid=%u name=%ls n=%u\n", busy[i].first,
					             busy[i].second->name.c_str(), busy[i].second->samples);
				}
				for (auto& [tid, thread]: threads) thread.samples = 0;
			}
			const auto top = [](const auto& map, size_t count) {
				std::vector<std::pair<uint64_t, uint32_t>> sorted(map.begin(), map.end());
				std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
				if (sorted.size() > count) sorted.resize(count);
				return sorted;
			};
			for (const auto& [rva, count]: top(exe_leaf, 40)) {
				std::fprintf(stderr, "PERF_SAMPLE_ALL excl rva=0x%llx n=%u\n", static_cast<unsigned long long>(rva), count);
			}
			for (const auto& [rip, count]: top(guest_leaf, 30)) {
				std::fprintf(stderr, "PERF_SAMPLE_ALL guest rip=0x%llx n=%u\n", static_cast<unsigned long long>(rip), count);
			}
			{
				// Aggregate by (module, routine, caller): the leaf address itself varies inside a routine.
				std::map<std::tuple<std::wstring, std::string, uint64_t>, uint32_t> routines;
				for (const auto& [key, count]: module_leaf) {
					HMODULE owner = nullptr;
					GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
					                   reinterpret_cast<LPCWSTR>(key.first), &owner);
					wchar_t name[MAX_PATH] {};
					if (owner != nullptr) GetModuleFileNameW(owner, name, MAX_PATH);
					const wchar_t* leaf = std::wcsrchr(name, L'\\');
					routines[{leaf != nullptr ? leaf + 1 : name, NearestExport(reinterpret_cast<uint64_t>(owner), key.first),
					          key.second}] += count;
				}
				std::vector<std::pair<std::tuple<std::wstring, std::string, uint64_t>, uint32_t>> sorted(routines.begin(),
				                                                                                            routines.end());
				std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
				for (size_t i = 0; i < sorted.size() && i < 40u; i++) {
					const auto& [module, routine, caller] = sorted[i].first;
					std::fprintf(stderr, "PERF_SAMPLE_ALL module=%ls fn=%s caller_rva=0x%llx n=%u\n", module.c_str(),
					             routine.c_str(), static_cast<unsigned long long>(caller), sorted[i].second);
				}
			}
			std::fflush(stderr);
			samples = 0;
			exe_leaf.clear();
			guest_leaf.clear();
			module_leaf.clear();
		}
	}).detach();
}

// KYTY_TRACE_PERF_SAMPLE=1 call chains: a register that points into the sampled thread's stack is moved into the copy.
void MoveIntoCopy(DWORD64& reg, uint64_t original, uint64_t copy, uint64_t bytes) {
	if (reg >= original && reg < original + bytes) reg = reg - original + copy;
}

void MoveRegistersIntoCopy(CONTEXT& c, uint64_t original, uint64_t copy, uint64_t bytes) {
	MoveIntoCopy(c.Rsp, original, copy, bytes);
	MoveIntoCopy(c.Rbp, original, copy, bytes);
	MoveIntoCopy(c.Rbx, original, copy, bytes);
	MoveIntoCopy(c.Rsi, original, copy, bytes);
	MoveIntoCopy(c.Rdi, original, copy, bytes);
	MoveIntoCopy(c.R12, original, copy, bytes);
	MoveIntoCopy(c.R13, original, copy, bytes);
	MoveIntoCopy(c.R14, original, copy, bytes);
	MoveIntoCopy(c.R15, original, copy, bytes);
}

// The sampled thread's call chain, unwound (RtlVirtualUnwind) on the copy of its stack taken while it was suspended:
// frames[i] = the start RVA of frame i's function in the executable (leaf first), 0 for a frame in another module.
// Runs after the thread was resumed (the lookups may take loader locks). A bad frame ends the walk (SEH).
size_t UnwindCopiedStack(const CONTEXT& context, const uint64_t* stack, size_t words, uint64_t base, uint64_t end,
                         uint32_t* frames, size_t max_frames) {
	size_t count = 0;
	__try {
		CONTEXT        walk     = context;
		const uint64_t original = context.Rsp;
		const uint64_t copy     = reinterpret_cast<uint64_t>(stack);
		const uint64_t bytes    = static_cast<uint64_t>(words) * 8u;
		MoveRegistersIntoCopy(walk, original, copy, bytes);
		while (count < max_frames) {
			const uint64_t pc = walk.Rip;
			if (pc == 0 || walk.Rsp < copy || walk.Rsp + 8u > copy + bytes) break;
			DWORD64 image_base = 0;
			// A caller's return address can be the first byte of the next function: look up the call instruction.
			auto* entry = RtlLookupFunctionEntry(count == 0 ? pc : pc - 1u, &image_base, nullptr);
			frames[count++] = (pc >= base && pc < end && entry != nullptr && image_base == base) ? entry->BeginAddress : 0u;
			if (entry == nullptr) {
				if (count != 1) break; // only a leaf may lack unwind data
				walk.Rip = *reinterpret_cast<const DWORD64*>(walk.Rsp);
				walk.Rsp += 8u;
			} else {
				// A frame register (UNWIND_INFO byte 3, low 4 bits; CONTEXT holds Rax..R15 in that order) must point
				// into the copy, or the unwind would read through a value that is not this stack's.
				if ((entry->UnwindData & 1u) == 0) {
					const auto* info           = reinterpret_cast<const uint8_t*>(image_base + entry->UnwindData);
					const uint32_t frame_register = info[3] & 0x0fu;
					if (frame_register != 0) {
						const DWORD64 value = (&walk.Rax)[frame_register];
						if (value < copy || value >= copy + bytes) break;
					}
				}
				void*   handler_data = nullptr;
				DWORD64 establisher  = 0;
				RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, pc, entry, &walk, &handler_data, &establisher, nullptr);
			}
			MoveRegistersIntoCopy(walk, original, copy, bytes);
		}
	} __except (EXCEPTION_EXECUTE_HANDLER) {
	}
	return count;
}

} // namespace

// Diagnostics (a stuck wait): the function starts (RVAs of kyty_emulator.exe, innermost first, 0 = another module) of
// another thread's stack, unwound from a copy taken while it was suspended.
void PerfDumpThreadStack(void* thread_handle, const char* label) {
	auto*      thread = static_cast<HANDLE>(thread_handle);
	auto*      module = GetModuleHandleW(nullptr);
	MODULEINFO info {};
	GetModuleInformation(GetCurrentProcess(), module, &info, sizeof(info));
	const auto            base = reinterpret_cast<uint64_t>(info.lpBaseOfDll);
	const auto            end  = base + info.SizeOfImage;
	std::vector<uint64_t> stack((128u * 1024u) / 8u);
	if (SuspendThread(thread) == static_cast<DWORD>(-1)) return;
	CONTEXT context {};
	context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
	const bool ok        = GetThreadContext(thread, &context) != 0;
	size_t     words     = 0;
	if (ok) {
		MEMORY_BASIC_INFORMATION region {};
		if (VirtualQuery(reinterpret_cast<void*>(context.Rsp), &region, sizeof(region)) != 0 && region.State == MEM_COMMIT &&
		    (region.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0) {
			const auto region_end = reinterpret_cast<uint64_t>(region.BaseAddress) + region.RegionSize;
			const auto bytes      = std::min<uint64_t>(stack.size() * 8u, region_end - context.Rsp) & ~7ull;
			std::memcpy(stack.data(), reinterpret_cast<const void*>(context.Rsp), bytes);
			words = bytes / 8u;
		}
	}
	ResumeThread(thread);
	std::array<uint32_t, 64> frames {};
	Common::HostException::PassAccessViolations(true);
	const size_t count = ok && words != 0 ? UnwindCopiedStack(context, stack.data(), words, base, end, frames.data(), 64) : 0;
	Common::HostException::PassAccessViolations(false);
	std::fprintf(stderr, "STACK_DUMP %s rip=0x%llx frames=", label,
	             static_cast<unsigned long long>(context.Rip >= base && context.Rip < end ? context.Rip - base : 0));
	for (size_t i = 0; i < count; i++) std::fprintf(stderr, "%x,", frames[i]);
	std::fprintf(stderr, "\n");
	std::fflush(stderr);
}

void PerfDumpCurrentStack(const char* label) {
	void*        frames[62] {};
	const USHORT count = RtlCaptureStackBackTrace(0, 62, frames, nullptr);
	const auto   base  = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	std::fprintf(stderr, "STACK_DUMP %s return_rvas=", label);
	for (USHORT i = 0; i < count; i++) {
		const auto a = reinterpret_cast<uint64_t>(frames[i]);
		std::fprintf(stderr, "%llx,", static_cast<unsigned long long>(a >= base ? a - base : 0));
	}
	std::fprintf(stderr, "\n");
	std::fflush(stderr);
}

// KYTY_TRACE_PERF=1 and KYTY_TRACE_PERF_SAMPLE=1: samples the calling (GPU) thread about every
// millisecond. The leaf instruction pointer gives exclusive counts; return addresses found on a copy
// of the top of the stack (values inside this executable that follow a call instruction) give
// approximate inclusive counts. Addresses are printed as RVAs of kyty_emulator.exe for offline
// resolution with the linker map. The thread is resumed before any analysis, so the sampler never
// takes a lock while the GPU thread is suspended. =all: every thread instead (StartAllThreadsSampler).
namespace {
// The sampler of one thread (the calling one): its lines start with `prefix`.
void StartThreadSampler(const char* prefix) {
	HANDLE thread = nullptr;
	if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &thread,
	                     THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0)) {
		return;
	}
	std::thread([thread, prefix] {
		Common::HostException::PassAccessViolations(true); // a bad unwound frame ends at UnwindCopiedStack's __except
		auto*       module = GetModuleHandleW(nullptr);
		MODULEINFO  info {};
		GetModuleInformation(GetCurrentProcess(), module, &info, sizeof(info));
		const auto  base = reinterpret_cast<uint64_t>(info.lpBaseOfDll);
		const auto  end  = base + info.SizeOfImage;
		constexpr size_t StackBytes = 128u * 1024u; // the unwound chains need the whole GPU-thread stack
		std::vector<uint64_t> stack(StackBytes / 8u);
		std::unordered_map<uint64_t, uint32_t> exclusive;
		std::unordered_map<uint64_t, uint32_t> inclusive;
		// Unwound chains: every function on the stack once per sample, and the innermost ChainFrames frames.
		constexpr size_t ChainFrames = 6;
		constexpr size_t MaxFrames   = 128;
		std::unordered_map<uint64_t, uint32_t>                       function_inclusive;
		std::unordered_map<uint64_t, uint32_t>                       function_exclusive; // leaf function start RVA
		std::map<std::array<uint32_t, ChainFrames>, uint32_t>        chains;
		std::unordered_map<uint64_t, uint32_t>                       edges; // caller << 32 | callee (function RVAs)
		std::unordered_set<uint64_t>                                 seen_edges;
		std::array<uint32_t, MaxFrames>                              frames {};
		std::unordered_map<uint64_t, uint32_t> external; // module base -> count
		// Leaf outside the executable: (module base, nearest return address into the executable) ->
		// count, i.e. which emulator code is in the library (allocator, memcpy, driver) at the time.
		std::map<std::pair<uint64_t, uint64_t>, uint32_t> callers;
		std::unordered_set<uint64_t> seen;
		uint32_t samples = 0;
		auto     window  = GetTickCount64();
		const auto is_return = [&](uint64_t address) {
			if (address < base + 0x1000u || address >= end) return false;
			const auto* code = reinterpret_cast<const uint8_t*>(address);
			return code[-5] == 0xe8 || (code[-2] == 0xff && (code[-1] & 0x38) == 0x10) ||
			       (code[-3] == 0xff && (code[-2] & 0x38) == 0x10) ||
			       (code[-6] == 0xff && (code[-5] & 0x38) == 0x10) ||
			       (code[-7] == 0xff && (code[-6] & 0x38) == 0x10);
		};
		for (;;) {
			Sleep(1);
			if (SuspendThread(thread) == static_cast<DWORD>(-1)) return;
			CONTEXT context {};
			context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER; // the integer registers for the unwinding
			const bool ok = GetThreadContext(thread, &context) != 0;
			size_t words = 0;
			if (ok) {
				MEMORY_BASIC_INFORMATION region {};
				// The stack pointer can sit on the thread's guard page while the stack grows; touching
				// it from here raised STATUS_GUARD_PAGE_VIOLATION in this thread and ended the process.
				if (VirtualQuery(reinterpret_cast<void*>(context.Rsp), &region, sizeof(region)) != 0 &&
				    region.State == MEM_COMMIT && (region.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0) {
					const auto region_end = reinterpret_cast<uint64_t>(region.BaseAddress) + region.RegionSize;
					const auto bytes = std::min<uint64_t>(StackBytes, region_end - context.Rsp) & ~7ull;
					std::memcpy(stack.data(), reinterpret_cast<const void*>(context.Rsp), bytes);
					words = bytes / 8u;
				}
			}
			ResumeThread(thread);
			if (!ok) continue;
			samples++;
			const auto rip = static_cast<uint64_t>(context.Rip);
			if (rip >= base && rip < end) {
				exclusive[rip - base]++;
			} else {
				HMODULE owner = nullptr;
				GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
				                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				                   reinterpret_cast<LPCWSTR>(rip), &owner);
				external[reinterpret_cast<uint64_t>(owner)]++;
				uint64_t caller = 0;
				for (size_t i = 0; i < words && i < 512u; i++) {
					if (is_return(stack[i])) {
						caller = stack[i] - base;
						break;
					}
				}
				callers[{reinterpret_cast<uint64_t>(owner), caller}]++;
			}
			seen.clear();
			for (size_t i = 0; i < words; i++) {
				if (is_return(stack[i]) && seen.insert(stack[i]).second) inclusive[stack[i] - base]++;
			}
			if (words != 0) {
				const size_t count = UnwindCopiedStack(context, stack.data(), words, base, end, frames.data(), MaxFrames);
				if (count != 0 && frames[0] != 0) function_exclusive[frames[0]]++;
				std::array<uint32_t, ChainFrames> chain {};
				for (size_t i = 0; i < count && i < ChainFrames; i++) chain[i] = frames[i];
				chains[chain]++;
				// Call graph edges (caller function -> callee function, 0 = another module), each once per sample.
				seen_edges.clear();
				for (size_t i = 1; i < count; i++) {
					if (frames[i] == 0 || frames[i] == frames[i - 1]) continue;
					const uint64_t edge = (static_cast<uint64_t>(frames[i]) << 32u) | frames[i - 1];
					if (seen_edges.insert(edge).second) edges[edge]++;
				}
				std::sort(frames.begin(), frames.begin() + static_cast<std::ptrdiff_t>(count));
				const auto last = std::unique(frames.begin(), frames.begin() + static_cast<std::ptrdiff_t>(count));
				for (auto it = frames.begin(); it != last; ++it) {
					if (*it != 0) function_inclusive[*it]++;
				}
			}
			const auto now = GetTickCount64();
			if (now - window < 10000u) continue;
			window = now;
			const auto top = [](const auto& map, size_t count) {
				std::vector<std::pair<uint64_t, uint32_t>> sorted(map.begin(), map.end());
				std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
				if (sorted.size() > count) sorted.resize(count);
				return sorted;
			};
			std::fprintf(stderr, "%s samples=%u\n", prefix, samples);
			for (const auto& [rva, count]: top(exclusive, 400))
				std::fprintf(stderr, "%s excl rva=0x%llx n=%u\n", prefix, static_cast<unsigned long long>(rva), count);
			for (const auto& [rva, count]: top(inclusive, 150))
				std::fprintf(stderr, "%s incl rva=0x%llx n=%u\n", prefix, static_cast<unsigned long long>(rva), count);
			// Unwound: function start RVA (each function once per sample), and the innermost frames (leaf first, 0 =
			// another module).
			for (const auto& [rva, count]: top(function_inclusive, 500))
				std::fprintf(stderr, "%s fincl rva=0x%llx n=%u\n", prefix, static_cast<unsigned long long>(rva), count);
			for (const auto& [rva, count]: top(function_exclusive, 300))
				std::fprintf(stderr, "%s fexcl rva=0x%llx n=%u\n", prefix, static_cast<unsigned long long>(rva), count);
			{
				std::vector<std::pair<std::array<uint32_t, ChainFrames>, uint32_t>> sorted(chains.begin(), chains.end());
				std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
				for (size_t i = 0; i < sorted.size() && i < 600u; i++) {
					const auto& c = sorted[i].first;
					std::fprintf(stderr, "%s chain n=%u f=%x,%x,%x,%x,%x,%x\n", prefix, sorted[i].second, c[0], c[1], c[2], c[3],
					             c[4], c[5]);
				}
				chains.clear();
			}
			function_inclusive.clear();
			function_exclusive.clear();
			for (const auto& [edge, count]: top(edges, 3000)) {
				std::fprintf(stderr, "%s edge n=%u p=%x c=%x\n", prefix, count, static_cast<uint32_t>(edge >> 32u),
				             static_cast<uint32_t>(edge));
			}
			edges.clear();
			{
				std::vector<std::pair<std::pair<uint64_t, uint64_t>, uint32_t>> sorted(callers.begin(), callers.end());
				std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
				for (size_t i = 0; i < sorted.size() && i < 40u; i++) {
					wchar_t name[MAX_PATH] {};
					if (sorted[i].first.first != 0) GetModuleFileNameW(reinterpret_cast<HMODULE>(sorted[i].first.first), name, MAX_PATH);
					const wchar_t* leaf = std::wcsrchr(name, L'\\');
					std::fprintf(stderr, "%s caller module=%ls rva=0x%llx n=%u\n", prefix, leaf != nullptr ? leaf + 1 : name,
					             static_cast<unsigned long long>(sorted[i].first.second), sorted[i].second);
				}
				callers.clear();
			}
			for (const auto& [owner, count]: top(external, 12)) {
				wchar_t name[MAX_PATH] {};
				if (owner != 0) GetModuleFileNameW(reinterpret_cast<HMODULE>(owner), name, MAX_PATH);
				std::fprintf(stderr, "%s module=%ls n=%u\n", prefix, owner != 0 ? name : L"?", count);
			}
			samples = 0;
			exclusive.clear();
			inclusive.clear();
			external.clear();
		}
	}).detach();
}
} // namespace

// KYTY_TRACE_PERF=1 and KYTY_TRACE_PERF_SAMPLE=1: samples the calling (GPU) thread (PERF_SAMPLE lines). =all: every thread.
void PerfSamplerStartForCurrentThread() {
	if (!PerfEnabled()) return;
	const char* flag = std::getenv("KYTY_TRACE_PERF_SAMPLE");
	if (flag != nullptr && std::strcmp(flag, "all") == 0) {
		StartAllThreadsSampler();
		return;
	}
	if (flag == nullptr || flag[0] != '1') return;
	StartThreadSampler("PERF_SAMPLE");
}

// KYTY_TRACE_PERF=1 and KYTY_TRACE_PERF_SAMPLE=resolve (project 15 N2.0b): samples the resolve thread instead.
void PerfSamplerStartForResolveThread() {
	if (!PerfEnabled()) return;
	const char* flag = std::getenv("KYTY_TRACE_PERF_SAMPLE");
	if (flag != nullptr && std::strcmp(flag, "resolve") == 0) StartThreadSampler("PERF_SAMPLE_RESOLVE");
}



// KYTY_TRACE_PERF: every report window, on the GPU thread, a fixed dependent integer loop (its time shows a slower core:
// clock, or another thread on the same physical core), the logical CPU it ran on, and the process's page faults and the
// GPU thread's context switches since the last report (the "slow runs" of 18-19.10).
void CpuProbeReport() {
	const auto start = std::chrono::steady_clock::now();
	uint64_t   x     = 12345;
	for (uint32_t i = 0; i < 400000u; ++i) {
		x = x * 6364136223846793005ull + 1442695040888963407ull;
	}
	const auto ns =
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
	PROCESSOR_NUMBER number {};
	GetCurrentProcessorNumberEx(&number);
	PROCESS_MEMORY_COUNTERS counters {};
	GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
	static DWORD last_faults = counters.PageFaultCount;
	const DWORD  faults      = counters.PageFaultCount - last_faults;
	last_faults              = counters.PageFaultCount;
	{
		auto&          st       = GetRenderLockStats();
		const uint64_t presents = st.presents.exchange(0);
		const uint64_t draws    = st.draws.exchange(0);
		std::fprintf(stderr,
		             "PERF render_lock presents=%llu present_wait_ms=%.1f present_hold_ms=%.1f present_hold_max_ms=%.2f "
		             "present_submit_ms=%.1f draws=%llu draw_wait_ms=%.1f draw_wait_max_ms=%.2f\n",
		             static_cast<unsigned long long>(presents), static_cast<double>(st.present_wait_ns.exchange(0)) / 1e6,
		             static_cast<double>(st.present_hold_ns.exchange(0)) / 1e6,
		             static_cast<double>(st.present_hold_max_ns.exchange(0)) / 1e6,
		             static_cast<double>(st.present_submit_ns.exchange(0)) / 1e6, static_cast<unsigned long long>(draws),
		             static_cast<double>(st.draw_wait_ns.exchange(0)) / 1e6,
		             static_cast<double>(st.draw_wait_max_ns.exchange(0)) / 1e6);
	}
	std::fprintf(stderr, "PERF cpu_probe loop_us=%.1f cpu=L%u page_faults=%lu working_set_mb=%llu chk=%llu\n",
	             static_cast<double>(ns) / 1e3, static_cast<unsigned>(number.Number), static_cast<unsigned long>(faults),
	             static_cast<unsigned long long>(counters.WorkingSetSize >> 20u), static_cast<unsigned long long>(x & 1u));
}

// KYTY_TRACE_PERF: the texture cache lock's contended waits by holder and by waiter code address (rva), the top entries
// by time: gpu/worker/other = count/ms of the waiting thread's kind.
void PerfReportTcLockPcs() {
	const auto base   = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
	const auto report = [&](const char* name, PerfPcSlot* slots) {
		struct Entry {
			uintptr_t pc;
			uint64_t  waits[3];
			uint64_t  nanos[3];
			uint64_t  total;
		};
		std::vector<Entry> entries;
		for (uint32_t i = 0; i < PerfPcSlotCount; i++) {
			auto&     slot = slots[i];
			Entry     e {slot.pc.load(std::memory_order_relaxed), {}, {}, 0};
			if (e.pc == 0) continue;
			for (uint32_t k = 0; k < 3; k++) {
				e.waits[k] = slot.waits[k].exchange(0, std::memory_order_relaxed);
				e.nanos[k] = slot.nanos[k].exchange(0, std::memory_order_relaxed);
				e.total += e.nanos[k];
			}
			if (e.total != 0) entries.push_back(e);
		}
		std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.total > b.total; });
		if (entries.size() > 12) entries.resize(12);
		for (const auto& e: entries) {
			const bool in_exe = e.pc >= base && e.pc - base < 0x40000000u;
			std::fprintf(stderr, "PERF %s rva=0x%llx gpu=%llu/%.2fms worker=%llu/%.2fms other=%llu/%.2fms\n", name,
			             static_cast<unsigned long long>(in_exe ? e.pc - base : 0u), static_cast<unsigned long long>(e.waits[0]),
			             static_cast<double>(e.nanos[0]) / 1e6, static_cast<unsigned long long>(e.waits[1]),
			             static_cast<double>(e.nanos[1]) / 1e6, static_cast<unsigned long long>(e.waits[2]),
			             static_cast<double>(e.nanos[2]) / 1e6);
		}
	};
	report("tc_lock_holders", PerfTcHolders());
	report("tc_lock_waiters", PerfTcWaiters());
}

#else

void PerfSamplerStartForCurrentThread() {}
void PerfSamplerStartForResolveThread() {}
void PerfReportTcLockPcs() {}
void CpuProbeReport() {}
void PerfDumpThreadStack(void* /*thread_handle*/, const char* /*label*/) {}
void PerfDumpCurrentStack(const char* /*label*/) {}

#endif

} // namespace Libs::Graphics
