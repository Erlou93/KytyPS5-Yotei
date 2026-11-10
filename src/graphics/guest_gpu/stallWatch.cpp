#include "graphics/guest_gpu/stallWatch.h"

#include "graphics/host_gpu/renderer/debugSample.h"

#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <thread>

// KYTY_DIAG_STALL_WATCH=1 (diagnostic only, hitch project 6.11, removed after the fix): a thread watches the presented-frame
// counter during gameplay; when no frame arrives for KYTY_DIAG_STALL_WATCH_MS (default 500) it suspends every other thread
// of the process, records each one's RIP and an unwound call stack (no allocation while they are suspended), resumes them
// and prints STALL_STACK lines (exe-relative RVAs, other modules by name+offset; thread names). One capture per stall, at
// most 12 per run, and a second capture 1 s later while the same stall lasts (what moved).
namespace Libs::Graphics::StallWatch {

namespace {

constexpr int MaxThreads = 256;
constexpr int MaxFrames  = 28;

constexpr size_t StackCopy = 16384;

struct ThreadStack {
	DWORD   tid    = 0;
	DWORD64 rip    = 0;
	DWORD64 rsp    = 0;
	size_t  bytes  = 0;
	alignas(8) uint8_t stack[StackCopy] {};
};

ThreadStack g_stacks[MaxThreads];

// While the other threads are suspended only kernel calls and memcpy run here (an unwinder that takes a user-mode lock
// held by a suspended thread hung the process on 6.11); the stacks are read after the threads run again.
int Capture(DWORD self) {
	DWORD  ids[MaxThreads];
	int    count = 0;
	HANDLE snap  = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snap == INVALID_HANDLE_VALUE) return 0;
	THREADENTRY32 te {};
	te.dwSize = sizeof(te);
	const DWORD pid = GetCurrentProcessId();
	if (Thread32First(snap, &te) != 0) {
		do {
			if (te.th32OwnerProcessID == pid && te.th32ThreadID != self && count < MaxThreads) ids[count++] = te.th32ThreadID;
			te.dwSize = sizeof(te);
		} while (Thread32Next(snap, &te) != 0);
	}
	CloseHandle(snap);
	static HANDLE handles[MaxThreads];
	int           open = 0;
	for (int i = 0; i < count; i++) {
		HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_LIMITED_INFORMATION, FALSE, ids[i]);
		if (h == nullptr) continue;
		if (SuspendThread(h) == static_cast<DWORD>(-1)) {
			CloseHandle(h);
			continue;
		}
		handles[open]       = h;
		g_stacks[open].tid  = ids[i];
		g_stacks[open].rip  = 0;
		g_stacks[open].bytes = 0;
		open++;
	}
	for (int i = 0; i < open; i++) {
		CONTEXT ctx {};
		ctx.ContextFlags = CONTEXT_CONTROL;
		if (GetThreadContext(handles[i], &ctx) == 0) continue;
		auto& t = g_stacks[i];
		t.rip   = ctx.Rip;
		t.rsp   = ctx.Rsp;
		MEMORY_BASIC_INFORMATION info {};
		if (ctx.Rsp != 0 && VirtualQuery(reinterpret_cast<void*>(ctx.Rsp), &info, sizeof(info)) != 0 && info.State == MEM_COMMIT) {
			const auto end   = reinterpret_cast<DWORD64>(info.BaseAddress) + info.RegionSize;
			const auto avail = end > ctx.Rsp ? end - ctx.Rsp : 0;
			t.bytes          = static_cast<size_t>(avail < StackCopy ? avail : StackCopy);
			std::memcpy(t.stack, reinterpret_cast<const void*>(ctx.Rsp), t.bytes);
		}
	}
	for (int i = 0; i < open; i++) {
		ResumeThread(handles[i]);
	}
	for (int i = 0; i < open; i++) {
		wchar_t* name = nullptr;
		char     text[128] {};
		if (SUCCEEDED(GetThreadDescription(handles[i], &name)) && name != nullptr) {
			std::snprintf(text, sizeof(text), "%ls", name);
			LocalFree(name);
		}
		CloseHandle(handles[i]);
		std::fprintf(stderr, "STALL_THREAD tid=%lu name=\"%s\"\n", static_cast<unsigned long>(g_stacks[i].tid), text);
	}
	return open;
}

// "exe+0x..." or "module+0x..." for an address inside a loaded module; empty when none.
bool Describe(DWORD64 address, DWORD64 exe, char* out, size_t size) {
	HMODULE mod = nullptr;
	if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCWSTR>(address), &mod) == 0 ||
	    mod == nullptr) {
		return false;
	}
	const auto base = reinterpret_cast<DWORD64>(mod);
	if (base == exe) {
		std::snprintf(out, size, "exe+0x%llx", static_cast<unsigned long long>(address - base));
	} else {
		wchar_t path[MAX_PATH] {};
		GetModuleFileNameW(mod, path, MAX_PATH);
		const wchar_t* slash = std::wcsrchr(path, L'\\');
		std::snprintf(out, size, "%ls+0x%llx", slash != nullptr ? slash + 1 : path, static_cast<unsigned long long>(address - base));
	}
	return true;
}

void Print(int threads, uint32_t frame, long long ms, int capture) {
	const auto exe = reinterpret_cast<DWORD64>(GetModuleHandleW(nullptr));
	MODULEINFO info {};
	K32GetModuleInformation(GetCurrentProcess(), GetModuleHandleW(nullptr), &info, sizeof(info));
	const DWORD64 exe_end = exe + info.SizeOfImage;
	for (int i = 0; i < threads; i++) {
		const auto& t = g_stacks[i];
		char        where[160] {};
		if (!Describe(t.rip, exe, where, sizeof(where))) std::snprintf(where, sizeof(where), "?0x%llx", static_cast<unsigned long long>(t.rip));
		std::fprintf(stderr, "STALL_STACK capture=%d frame=%u stalled_ms=%lld tid=%lu rip=%s scan:", capture, frame, ms,
		             static_cast<unsigned long>(t.tid), where);
		// Heuristic call stack: stack words that point into the exe image (return addresses and some stale values).
		int shown = 0;
		for (size_t off = 0; off + 8 <= t.bytes && shown < 32; off += 8) {
			DWORD64 v = 0;
			std::memcpy(&v, t.stack + off, 8);
			if (v > exe && v < exe_end) {
				std::fprintf(stderr, " exe+0x%llx", static_cast<unsigned long long>(v - exe));
				shown++;
			}
		}
		std::fprintf(stderr, "\n");
	}
	std::fflush(stderr);
}

void Run() {
	const char*     value     = std::getenv("KYTY_DIAG_STALL_WATCH_MS");
	const long long threshold = value != nullptr && std::atoll(value) > 0 ? std::atoll(value) : 500;
	const DWORD     self      = GetCurrentThreadId();
	SetThreadDescription(GetCurrentThread(), L"Kyty.StallWatch");
	const char*     idle_value = std::getenv("KYTY_DIAG_STALL_WATCH_IDLE_MS");
	const long long idle_ms    = idle_value != nullptr ? std::atoll(idle_value) : 0;
	int64_t         idle_taken = 0;
	uint32_t last_frame = DebugPresentCount().load(std::memory_order_relaxed);
	auto     last_move  = std::chrono::steady_clock::now();
	int      captures   = 0;
	int      in_stall   = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms > 0 ? 5 : 20));
		const uint32_t frame = DebugPresentCount().load(std::memory_order_relaxed);
		const auto     now   = std::chrono::steady_clock::now();
		if (frame != last_frame) {
			if (in_stall != 0) {
				std::fprintf(stderr, "STALL_END frame=%u stalled_ms=%lld\n", frame,
				             static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(now - last_move).count()));
			}
			last_frame = frame;
			last_move  = now;
			in_stall   = 0;
			continue;
		}
		if (GameplayStartNs().load(std::memory_order_relaxed) == 0) continue;
		if (idle_ms > 0) {
			// the execution thread idle (KYTY_DIAG_STALL_WATCH_IDLE_MS): one capture per idle wait, at most 20
			const int64_t since = g_exec_idle_since.load(std::memory_order_acquire);
			const int64_t now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
			if (since != 0 && since != idle_taken && captures < 20 && (now_ns - since) / 1000000 >= idle_ms) {
				idle_taken = since;
				captures++;
				const int threads = Capture(self);
				Print(threads, frame, (now_ns - since) / 1000000, captures);
			}
			continue;
		}
		const auto ms = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(now - last_move).count());
		if (captures >= 12) continue;
		if ((in_stall == 0 && ms >= threshold) || (in_stall == 1 && ms >= threshold + 1000)) {
			in_stall++;
			captures++;
			const int threads = Capture(self);
			Print(threads, frame, ms, captures);
		}
	}
}

} // namespace

void StartIfEnabled() {
	const char* value = std::getenv("KYTY_DIAG_STALL_WATCH");
	if (value == nullptr || value[0] != '1') return;
	std::fprintf(stderr, "DIAG_STALL_WATCH enabled\n");
	std::thread(Run).detach();
}

} // namespace Libs::Graphics::StallWatch
