#include "graphics/presentation/brushStroke.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>

namespace Libs::Graphics::Brush {

namespace {

constexpr float PI = std::numbers::pi_v<float>;

ImVec2 Place(ImVec2 origin, ImVec2 point, float scale) {
	return {origin.x + point.x * scale, origin.y + point.y * scale};
}

// ---- F2 panel frame: style 2 "Narożniki" (linePts() + brushLine(), PR.stroke, DR.end) ----

constexpr float LINE_WOBBLE   = 1.2f;  // linePts() wob, mockup px
constexpr float LINE_STEP     = 2.0f;  // linePts() step, mockup px between samples
constexpr float LINE_WIDTH    = 8.0f;  // frame() s = 8: brush width, mockup px
constexpr int   LINE_BRISTLES = 14;    // brushLine() n
constexpr float LINE_OVERHANG = 4.0f;  // each stroke starts 4 px past its corner
constexpr float LINE_ACROSS   = 0.42f; // horizontal strokes: a = W * .42
constexpr float LINE_DOWN     = 0.38f; // vertical strokes: b = H * .38
constexpr int   LINE_STRIDE   = 2;     // every 2nd sample drawn (4 px apart; the noise is ~20 px)
// Noise tables sized for this many samples whatever the stroke's length, so a panel that grows or shrinks by a few
// pixels (auto-resize when a value gains a digit) keeps its strokes instead of getting new ones. The mockup sizes them
// by the stroke's length; its panels never resize.
constexpr int LINE_NOISE_SAMPLES = 1024;

// PR.stroke: a pressed start, an even body, a lifted end.
float StrokePress(float f) {
	return f < 0.05f  ? 0.5f + 10.0f * f
	       : f > 0.8f ? std::max(0.18f, 1.0f - (f - 0.8f) * 4.0f)
	                  : 1.0f - 0.12f * f;
}

// DR.end: the brush runs dry over the last 28%.
float StrokeDryEnd(float f) {
	return f > 0.72f ? (f - 0.72f) * 1.7f : 0.03f;
}

// A straight brush stroke from `from` to `to` in pixels; `bow` bends it across (mockup px), `unit` is px per mockup px.
void BuildLineStroke(Stroke& out, ImVec2 from, ImVec2 to, uint32_t path_seed, float bow, uint32_t brush_seed,
                     float unit) {
	out.Clear();
	const float dx     = to.x - from.x;
	const float dy     = to.y - from.y;
	const float length = std::sqrt(dx * dx + dy * dy);
	if (length < 1.0f) {
		return;
	}
	const int n = std::clamp(static_cast<int>(std::lround(length / (LINE_STEP * unit))), 3, LINE_NOISE_SAMPLES);
	// linePts(): the centre line, bowed and wobbling across its direction.
	std::array<ImVec2, LINE_NOISE_SAMPLES> centre;
	std::array<ImVec2, LINE_NOISE_SAMPLES> normal;
	Random      path_random(path_seed);
	const Noise wobble(path_random, LINE_NOISE_SAMPLES, 14.0f);
	const float nx = -dy / length;
	const float ny = dx / length;
	for (int i = 0; i < n; i++) {
		const float f = static_cast<float>(i) / static_cast<float>(n - 1);
		const float d =
		    (bow * std::sin(PI * f) + (wobble(static_cast<float>(i)) - 0.5f) * LINE_WOBBLE * 2.0f) * unit;
		centre[i] = {from.x + dx * f + nx * d, from.y + dy * f + ny * d};
	}
	// brushLine(): normals from the neighbours, then the bristles across them.
	for (int i = 0; i < n; i++) {
		const ImVec2 a  = centre[std::max(0, i - 1)];
		const ImVec2 b  = centre[std::min(n - 1, i + 1)];
		const float  ex = b.x - a.x;
		const float  ey = b.y - a.y;
		float        l  = std::sqrt(ex * ex + ey * ey);
		l               = l > 0.0f ? l : 1.0f;
		normal[i]       = {-ey / l, ex / l};
	}
	const float width = LINE_WIDTH * unit;
	Random      random(brush_seed);
	out.samples = n;
	out.bristles.reserve(LINE_BRISTLES);
	out.points.reserve(static_cast<size_t>(LINE_BRISTLES) * n);
	out.pen.reserve(static_cast<size_t>(LINE_BRISTLES) * n);
	for (int j = 0; j < LINE_BRISTLES; j++) {
		// The mockup's order of random draws: off, a, lw, sh, dn, wn.
		const float off =
		    (static_cast<float>(j) / (LINE_BRISTLES - 1) * 2.0f - 1.0f) * (1.0f - 0.1f * random.Next());
		const float alpha  = (0.5f + 0.5f * random.Next()) * (1.0f - 0.3f * std::abs(off));
		const float lw     = 0.6f + 0.8f * random.Next();
		const float shade  = random.Next();
		const Noise dry(random, LINE_NOISE_SAMPLES, 9.0f);
		const Noise wiggle(random, LINE_NOISE_SAMPLES, 11.0f);
		// GOLD: hsl(41, 66..84%, 40..62%).
		out.bristles.push_back({Hsl(41.0f, (66.0f + 18.0f * shade) / 100.0f, (40.0f + 22.0f * shade) / 100.0f),
		                        alpha, std::max(0.6f * unit, width / LINE_BRISTLES * 1.9f * lw)});
		for (int i = 0; i < n; i++) {
			const float f      = static_cast<float>(i) / static_cast<float>(n - 1);
			const float k      = static_cast<float>(i);
			const bool  down   = !(dry(k) < StrokeDryEnd(f) + 0.12f * std::abs(off));
			const float across = (off + (wiggle(k) - 0.5f) * 0.3f) * width * StrokePress(f) * 0.5f;
			out.points.push_back({centre[i].x + normal[i].x * across, centre[i].y + normal[i].y * across});
			out.pen.push_back(down ? 1 : 0);
		}
	}
}

// ---- Shader badge: G9 "Żar ze stygnącymi rozpryskami" ----
// o = {size:56, r:.28, w:.132, sweep:.9, bristles:18, press:P.classic, dry:D.late, wob:.025, ghost:.28,
//      head:{r:.14, hot:true}, embers:14, glow:{a:.35, blur:.04}, splat:30, splatSpread:1.25, splatGlow:true,
//      splatCool:true}, seed 11. Lengths are in units of the badge's side (the mockup's S).

constexpr uint32_t ENSO_SEED         = 11;
constexpr int      ENSO_K            = 240;    // brushState() K: samples along the stroke
constexpr float    ENSO_RADIUS       = 0.28f;  // o.r
constexpr float    ENSO_WIDTH        = 0.132f; // o.w
constexpr float    ENSO_SWEEP        = 0.9f;   // o.sweep: the ensō stays open
constexpr float    ENSO_START        = -100.0f * PI / 180.0f; // o.start ?? -100 degrees
constexpr int      ENSO_BRISTLES     = 18;     // o.bristles
constexpr float    ENSO_WOBBLE       = 0.025f; // o.wob
constexpr float    ENSO_GHOST        = 0.28f;  // o.ghost: the whole ensō in faint ink
constexpr float    ENSO_HEAD_RADIUS  = 0.14f;  // o.head.r
constexpr int      ENSO_EMBERS       = 14;     // o.embers
constexpr float    ENSO_GLOW_ALPHA   = 0.35f;  // o.glow.a
constexpr float    ENSO_GLOW_BLUR    = 0.04f;  // o.glow.blur
constexpr int      ENSO_SPLATS       = 30;     // o.splat
constexpr float    ENSO_SPLAT_SPREAD = 1.25f;  // o.splatSpread
constexpr float    ENSO_COOL_SPAN    = 0.18f;  // splatCool: a drop is gold once the brush is 18% further
// The mockup strokes the bristles twice (the glow pass at .8, then the main pass); one pass at this multiplier gives
// about the same opacity (1 - (1 - a)(1 - .8a) for a in .36..1).
constexpr float ENSO_MAIN_ALPHA = 1.45f;
constexpr ImU32 ENSO_INK        = IM_COL32(0x17, 0x15, 0x13, 255); // ghost: --ink #171513
constexpr ImU32 ENSO_ON_DARK    = IM_COL32(0xec, 0xe8, 0xe0, 255); // the number: --on-dark #ece8e0

// P.classic: a heavy first touch, then thinning out.
float ClassicPress(float f) {
	return f < 0.05f ? 0.55f + 9.0f * f : std::max(0.25f, 1.0f - 0.7f * (f - 0.05f));
}

// D.late: dry gaps after 55% of the stroke.
float LateDry(float f) {
	return f > 0.55f ? (f - 0.55f) * 1.3f : 0.04f;
}

float EnsoAngle(float f) {
	return ENSO_START + 2.0f * PI * ENSO_SWEEP * f;
}

struct Enso {
	struct Ember {
		float lag   = 0.0f;
		float side  = 0.0f;
		float size  = 0.0f;
		float phase = 0.0f;
	};
	struct Splat {
		float  at     = 0.0f; // progress at which the brush throws it
		float  side   = 0.0f;
		float  size   = 0.0f;
		ImVec2 pos    = {};
		float  radius = 0.0f;
	};
	Stroke                         stroke; // unit square, centre (0.5, 0.5)
	Noise                          wobble;
	std::array<ImVec2, ENSO_K + 1> centre {};
	std::array<ImVec2, ENSO_K + 1> normal {};
	std::array<float, ENSO_K + 1>  half_width {};
	std::array<float, ENSO_K + 1>  coverage {}; // share of the bristles' ink on the paper at each sample
	std::array<Ember, ENSO_EMBERS> embers {};
	std::array<Splat, ENSO_SPLATS> splats {};
};

// ringPoint(): a point of the ensō in the unit square, `off` across the stroke in half-widths.
ImVec2 RingPoint(const Noise& wobble, int k, float off) {
	const float f      = static_cast<float>(k) / ENSO_K;
	const float angle  = EnsoAngle(f);
	const float radius = ENSO_RADIUS + (wobble(static_cast<float>(k)) - 0.5f) * ENSO_WOBBLE +
	                     off * ENSO_WIDTH * ClassicPress(f) * 0.5f;
	return {0.5f + radius * std::cos(angle), 0.5f + radius * std::sin(angle)};
}

Enso BuildEnso() {
	Enso   enso;
	Random random(ENSO_SEED);
	// brushState(): the mockup's order of random draws (per bristle off, a, sh, lw, dn, wn; then wob; then the splats).
	struct Hair {
		float off   = 0.0f;
		float alpha = 0.0f;
		float shade = 0.0f;
		float lw    = 0.0f;
		Noise dry;
		Noise wiggle;
	};
	std::array<Hair, ENSO_BRISTLES> hairs {};
	for (int j = 0; j < ENSO_BRISTLES; j++) {
		auto& hair = hairs[j];
		hair.off   = (static_cast<float>(j) / (ENSO_BRISTLES - 1) * 2.0f - 1.0f) * (1.0f - 0.1f * random.Next());
		hair.alpha = (0.55f + 0.45f * random.Next()) * (1.0f - 0.35f * std::abs(hair.off));
		hair.shade = random.Next();
		hair.lw    = 0.6f + 0.8f * random.Next();
		hair.dry   = Noise(random, ENSO_K, 7.0f);
		hair.wiggle = Noise(random, ENSO_K, 9.0f);
	}
	enso.wobble = Noise(random, ENSO_K, 45.0f);
	for (auto& splat: enso.splats) {
		splat.at   = random.Next();
		splat.side = random.Next() * 2.0f - 1.0f;
		splat.size = random.Next();
	}

	// strokeRing(): every bristle along the whole ensō; drawing cuts it at the progress.
	auto& stroke   = enso.stroke;
	stroke.samples = ENSO_K + 1;
	stroke.bristles.reserve(ENSO_BRISTLES);
	stroke.points.reserve(static_cast<size_t>(ENSO_BRISTLES) * (ENSO_K + 1));
	stroke.pen.reserve(static_cast<size_t>(ENSO_BRISTLES) * (ENSO_K + 1));
	std::array<float, ENSO_K + 1> ink {};
	float                         ink_total = 0.0f;
	for (const auto& hair: hairs) {
		// gold: hsl(41, 68..84%, 40..62%); line width max(S * .004, S * w / n * 1.8) * lw.
		stroke.bristles.push_back({Hsl(41.0f, (68.0f + 16.0f * hair.shade) / 100.0f,
		                               (40.0f + 22.0f * hair.shade) / 100.0f),
		                           hair.alpha,
		                           std::max(0.004f, ENSO_WIDTH / ENSO_BRISTLES * 1.8f) * hair.lw});
		const float weight = std::min(1.0f, hair.alpha);
		ink_total += weight;
		for (int k = 0; k <= ENSO_K; k++) {
			const float f    = static_cast<float>(k) / ENSO_K;
			const float kf   = static_cast<float>(k);
			const bool  down = !(hair.dry(kf) < LateDry(f) + 0.14f * std::abs(hair.off));
			stroke.points.push_back(RingPoint(enso.wobble, k, hair.off + (hair.wiggle(kf) - 0.5f) * 0.35f));
			stroke.pen.push_back(down ? 1 : 0);
			if (down) {
				ink[k] += weight;
			}
		}
	}
	for (int k = 0; k <= ENSO_K; k++) {
		const float f      = static_cast<float>(k) / ENSO_K;
		const float angle  = EnsoAngle(f);
		enso.centre[k]     = RingPoint(enso.wobble, k, 0.0f);
		enso.normal[k]     = {std::cos(angle), std::sin(angle)};
		enso.half_width[k] = ENSO_WIDTH * ClassicPress(f) * 0.5f;
		enso.coverage[k]   = ink_total > 0.0f ? ink[k] / ink_total : 0.0f;
	}

	// Embers have their own generator (seed + 99): lag, d, s, ph.
	Random ember_random(ENSO_SEED + 99);
	for (auto& ember: enso.embers) {
		ember.lag   = ember_random.Next();
		ember.side  = ember_random.Next() * 2.0f - 1.0f;
		ember.size  = ember_random.Next();
		ember.phase = ember_random.Next() * 6.3f;
	}
	// Drops sit on the stroke's centre line where the brush passed, thrown outwards or inwards.
	for (auto& splat: enso.splats) {
		const int    k     = static_cast<int>(std::floor(splat.at * ENSO_K));
		const ImVec2 q     = RingPoint(enso.wobble, k, 0.0f);
		const float  angle = EnsoAngle(splat.at);
		const float  d     = splat.side * ENSO_WIDTH * ENSO_SPLAT_SPREAD;
		splat.pos          = {q.x + d * std::cos(angle), q.y + d * std::sin(angle)};
		splat.radius       = 0.006f + 0.016f * splat.size * splat.size;
	}
	return enso;
}

// A band along the ensō's centre line from sample 0 to `last`, `color` at `alpha` (times the bristle coverage) over
// the stroke's width and fading to nothing over `fringe` pixels on both sides: the canvas shadows (glow, drop shadow).
void AddEnsoBand(ImDrawList* draw, const Enso& enso, int last, ImVec2 origin, float size, int stride, ImU32 color,
                 float alpha, float fringe) {
	if (last < 1) {
		return;
	}
	const int    count = last / stride + 1 + (last % stride != 0 ? 1 : 0);
	const ImVec2 uv    = ImGui::GetFontTexUvWhitePixel();
	const ImU32  clear = color & ~IM_COL32_A_MASK;
	draw->PrimReserve((count - 1) * 18, count * 4);
	const unsigned int base = draw->_VtxCurrentIdx;
	for (int i = 0; i < count; i++) {
		const int    k     = std::min(i * stride, last);
		const ImVec2 c     = Place(origin, enso.centre[k], size);
		const ImVec2 n     = enso.normal[k];
		const float  inner = enso.half_width[k] * size;
		const float  outer = inner + fringe;
		const ImU32  ink   = WithAlpha(color, alpha * enso.coverage[k]);
		draw->PrimWriteVtx({c.x - n.x * outer, c.y - n.y * outer}, uv, clear);
		draw->PrimWriteVtx({c.x - n.x * inner, c.y - n.y * inner}, uv, ink);
		draw->PrimWriteVtx({c.x + n.x * inner, c.y + n.y * inner}, uv, ink);
		draw->PrimWriteVtx({c.x + n.x * outer, c.y + n.y * outer}, uv, clear);
	}
	for (int i = 0; i + 1 < count; i++) {
		const unsigned int v = base + static_cast<unsigned int>(i) * 4u;
		for (unsigned int band = 0; band < 3; band++) {
			const auto a = static_cast<ImDrawIdx>(v + band);
			const auto b = static_cast<ImDrawIdx>(v + band + 1u);
			const auto c = static_cast<ImDrawIdx>(v + 4u + band + 1u);
			const auto d = static_cast<ImDrawIdx>(v + 4u + band);
			draw->PrimWriteIdx(a);
			draw->PrimWriteIdx(b);
			draw->PrimWriteIdx(c);
			draw->PrimWriteIdx(a);
			draw->PrimWriteIdx(c);
			draw->PrimWriteIdx(d);
		}
	}
}

constexpr int GLOW_SEGMENTS = 12;

const std::array<ImVec2, GLOW_SEGMENTS>& GlowDirections() {
	static const auto directions = [] {
		std::array<ImVec2, GLOW_SEGMENTS> result {};
		for (int i = 0; i < GLOW_SEGMENTS; i++) {
			const float angle = 2.0f * PI * static_cast<float>(i) / GLOW_SEGMENTS;
			result[i]         = {std::cos(angle), std::sin(angle)};
		}
		return result;
	}();
	return directions;
}

} // namespace

float Random::Next() noexcept {
	m_state += 0x6D2B79F5u;
	uint32_t t = (m_state ^ (m_state >> 15u)) * (1u | m_state);
	t          = (t + ((t ^ (t >> 7u)) * (61u | t))) ^ t;
	return static_cast<float>(static_cast<double>(t ^ (t >> 14u)) / 4294967296.0);
}

float Hash01(uint32_t value) noexcept {
	uint32_t x = (value ^ 0x9e3779b9u) * 0x85ebca6bu;
	x ^= x >> 13u;
	x *= 0xc2b2ae35u;
	return static_cast<float>(static_cast<double>(x ^ (x >> 16u)) / 4294967296.0);
}

Noise::Noise(Random& random, int length, float scale) noexcept: m_scale(scale) {
	const int wanted =
	    static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(scale))) + 3;
	m_count = std::clamp(wanted, 2, static_cast<int>(m_points.size()));
	for (int i = 0; i < m_count; i++) {
		m_points[i] = random.Next();
	}
}

float Noise::operator()(float k) const noexcept {
	if (m_count < 2) {
		return 0.5f;
	}
	const float x = k / m_scale;
	const int   i = std::clamp(static_cast<int>(std::floor(x)), 0, m_count - 2);
	const float f = std::clamp(x - static_cast<float>(i), 0.0f, 1.0f);
	const float u = f * f * (3.0f - 2.0f * f);
	return m_points[i] * (1.0f - u) + m_points[i + 1] * u;
}

ImU32 Hsl(float hue, float saturation, float lightness, float alpha) noexcept {
	const float c = (1.0f - std::abs(2.0f * lightness - 1.0f)) * saturation;
	const float h = std::fmod(hue, 360.0f) / 60.0f;
	const float x = c * (1.0f - std::abs(std::fmod(h, 2.0f) - 1.0f));
	float       r = 0.0f;
	float       g = 0.0f;
	float       b = 0.0f;
	if (h < 1.0f) {
		r = c;
		g = x;
	} else if (h < 2.0f) {
		r = x;
		g = c;
	} else if (h < 3.0f) {
		g = c;
		b = x;
	} else if (h < 4.0f) {
		g = x;
		b = c;
	} else if (h < 5.0f) {
		r = x;
		b = c;
	} else {
		r = c;
		b = x;
	}
	const float m    = lightness - c * 0.5f;
	const auto  to_byte = [](float v) { return static_cast<int>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
	return IM_COL32(to_byte(r + m), to_byte(g + m), to_byte(b + m), to_byte(alpha));
}

ImU32 WithAlpha(ImU32 color, float alpha) noexcept {
	const auto a = static_cast<ImU32>(std::lround(std::clamp(alpha, 0.0f, 1.0f) * 255.0f));
	return (color & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT);
}

void Stroke::Clear() noexcept {
	samples = 0;
	bristles.clear();
	points.clear();
	pen.clear();
}

void Stroke::Draw(ImDrawList* draw, int last, ImVec2 origin, float scale, int stride, float alpha,
                  ImU32 color) const {
	last = std::min(last, samples - 1);
	if (draw == nullptr || last < 1) {
		return;
	}
	stride = std::max(stride, 1);
	std::array<ImVec2, 128> run;
	for (size_t b = 0; b < bristles.size(); b++) {
		const auto& bristle   = bristles[b];
		float       thickness = bristle.thickness * scale;
		float       opacity   = std::min(1.0f, bristle.alpha * alpha);
		// ImGui widens anti-aliased lines thinner than a pixel to one pixel: fade them instead.
		if (thickness < 1.0f) {
			opacity *= thickness;
			thickness = 1.0f;
		}
		const ImU32 col = WithAlpha(color != 0 ? color : bristle.color, opacity);
		if ((col & IM_COL32_A_MASK) == 0) {
			continue;
		}
		const size_t   first = b * static_cast<size_t>(samples);
		const ImVec2*  pts   = points.data() + first;
		const uint8_t* down  = pen.data() + first;
		int            count = 0;
		int            start = 0;
		for (int k = 0; k <= last; k++) {
			if (down[k] == 0) {
				continue;
			}
			if (count == 0) {
				start = k;
			}
			const bool end = k == last || down[k + 1] == 0;
			if ((k - start) % stride == 0 || end) {
				run[count++] = Place(origin, pts[k], scale);
				if (count == static_cast<int>(run.size()) && !end) {
					draw->AddPolyline(run.data(), count, col, thickness);
					run[0] = run[count - 1];
					count  = 1;
				}
			}
			if (end) {
				if (count >= 2) {
					draw->AddPolyline(run.data(), count, col, thickness);
				}
				count = 0;
			}
		}
	}
}

void AddRadialGlow(ImDrawList* draw, ImVec2 centre, float radius, float middle_radius, ImU32 inner, ImU32 middle,
                   ImU32 outer) {
	if (draw == nullptr || radius <= 0.0f) {
		return;
	}
	const auto&  directions = GlowDirections();
	const ImVec2 uv         = ImGui::GetFontTexUvWhitePixel();
	draw->PrimReserve(GLOW_SEGMENTS * 9, 1 + GLOW_SEGMENTS * 2);
	const unsigned int base = draw->_VtxCurrentIdx;
	draw->PrimWriteVtx(centre, uv, inner);
	for (const auto& d: directions) {
		draw->PrimWriteVtx({centre.x + d.x * middle_radius, centre.y + d.y * middle_radius}, uv, middle);
		draw->PrimWriteVtx({centre.x + d.x * radius, centre.y + d.y * radius}, uv, outer);
	}
	for (unsigned int i = 0; i < GLOW_SEGMENTS; i++) {
		const unsigned int j  = (i + 1u) % GLOW_SEGMENTS;
		const auto         c  = static_cast<ImDrawIdx>(base);
		const auto         m0 = static_cast<ImDrawIdx>(base + 1u + i * 2u);
		const auto         o0 = static_cast<ImDrawIdx>(base + 2u + i * 2u);
		const auto         m1 = static_cast<ImDrawIdx>(base + 1u + j * 2u);
		const auto         o1 = static_cast<ImDrawIdx>(base + 2u + j * 2u);
		draw->PrimWriteIdx(c);
		draw->PrimWriteIdx(m0);
		draw->PrimWriteIdx(m1);
		draw->PrimWriteIdx(m0);
		draw->PrimWriteIdx(o0);
		draw->PrimWriteIdx(o1);
		draw->PrimWriteIdx(m0);
		draw->PrimWriteIdx(o1);
		draw->PrimWriteIdx(m1);
	}
}

void DrawCornerFrame(ImDrawList* draw, ImVec2 rect_min, ImVec2 rect_max, float unit) {
	// Built for one panel size; the compact and the detailed panel swap it when toggled.
	struct Frame {
		ImVec2                size {-1.0f, -1.0f};
		float                 unit = 0.0f;
		std::array<Stroke, 4> strokes;
	};
	static Frame frame;
	const float  width  = rect_max.x - rect_min.x;
	const float  height = rect_max.y - rect_min.y;
	if (draw == nullptr || width < 16.0f || height < 16.0f) {
		return;
	}
	if (width != frame.size.x || height != frame.size.y || unit != frame.unit) {
		frame.size     = {width, height};
		frame.unit     = unit;
		const float a  = width * LINE_ACROSS;
		const float b  = height * LINE_DOWN;
		const float o  = LINE_OVERHANG * unit;
		// frame(): top-left along the top and down the left edge, bottom-right along the bottom and up the right edge;
		// the bow (0.8 px) leans each stroke into the panel. Seeds: linePts 3 / 5 / 7 / 9, brushLine 11 / 13 / 17 / 19.
		BuildLineStroke(frame.strokes[0], {-o, 0.0f}, {a, 0.0f}, 3, 0.8f, 11, unit);
		BuildLineStroke(frame.strokes[1], {0.0f, -o}, {0.0f, b}, 5, -0.8f, 13, unit);
		BuildLineStroke(frame.strokes[2], {width + o, height}, {width - a, height}, 7, 0.8f, 17, unit);
		BuildLineStroke(frame.strokes[3], {width, height + o}, {width, height - b}, 9, -0.8f, 19, unit);
	}
	const float pad = 16.0f * unit;
	draw->PushClipRect({rect_min.x - pad, rect_min.y - pad}, {rect_max.x + pad, rect_max.y + pad}, false);
	for (const auto& stroke: frame.strokes) {
		stroke.Draw(draw, stroke.samples - 1, rect_min, 1.0f, LINE_STRIDE, 1.0f);
	}
	draw->PopClipRect();
}

void DrawShaderEnso(ImDrawList* draw, ImVec2 top_left, float size, float progress, double seconds, ImFont* font,
                    int percent) {
	static const Enso enso = BuildEnso();
	if (draw == nullptr || size < 4.0f) {
		return;
	}
	const float  p    = std::clamp(progress, 0.0f, 1.0f);
	const int    kmax = static_cast<int>(std::floor(static_cast<float>(ENSO_K) * p));
	const float  u    = size / 56.0f; // pixels per mockup pixel
	const double t    = seconds;
	const ImVec2 middle {top_left.x + size * 0.5f, top_left.y + size * 0.5f};
	// The mockup's 240 samples serve a 224 px canvas; keep about 1.5 px between drawn points.
	const float arc_per_sample = 2.0f * PI * ENSO_RADIUS * ENSO_SWEEP * size / ENSO_K;
	const int   stride         = std::clamp(static_cast<int>(std::lround(1.5f / arc_per_sample)), 1, 4);

	char text[8];
	std::snprintf(text, sizeof(text), "%d", std::clamp(percent, 0, 100));
	const float  text_size = 18.0f * u;
	const ImVec2 text_dim =
	    font != nullptr ? font->CalcTextSizeA(text_size, std::numeric_limits<float>::max(), 0.0f, text) : ImVec2 {};
	// The number's text-shadow (0 1px 3px #000, 0 0 6px rgba(0,0,0,.6)) as a soft dark patch under everything, so it
	// reads on snow; the 1 px drop copy goes with the text at the end. It stays inside the ring (inner edge ~12 px).
	if (font != nullptr) {
		const float halo = std::max(text_dim.x * 0.5f, text_size * 0.35f) + 2.0f * u;
		AddRadialGlow(draw, middle, halo, halo * 0.5f, IM_COL32(0, 0, 0, 92), IM_COL32(0, 0, 0, 64),
		              IM_COL32(0, 0, 0, 0));
	}
	// The canvas drop-shadow (0 1px 2px rgba(0,0,0,.6)) under the painted part, so the gold reads on snow.
	if (kmax >= 2) {
		AddEnsoBand(draw, enso, kmax, {top_left.x, top_left.y + u}, size, stride, IM_COL32(0, 0, 0, 255), 0.45f,
		            2.0f * u);
	}
	// Ghost: the whole ensō in faint ink, so the share still to come shows.
	enso.stroke.Draw(draw, ENSO_K, top_left, size, stride, ENSO_GHOST, ENSO_INK);
	if (kmax >= 2) {
		// Glow: the canvas shadowBlur rgba(255,190,70,.35) around the painted part, as a soft band.
		AddEnsoBand(draw, enso, kmax, top_left, size, stride, IM_COL32(255, 190, 70, 255), ENSO_GLOW_ALPHA,
		            ENSO_GLOW_BLUR * size * 1.5f);
		enso.stroke.Draw(draw, kmax, top_left, size, stride, ENSO_MAIN_ALPHA);

		// Embers: sparks flickering behind the brush tip and drifting up.
		for (const auto& ember: enso.embers) {
			const int    k       = std::max(0, kmax - static_cast<int>(std::floor(ember.lag * ENSO_K * 0.12f)));
			const ImVec2 q       = RingPoint(enso.wobble, k, ember.side * 1.4f);
			const float  flicker =
			    0.5f + 0.5f * static_cast<float>(std::sin(t * (4.0 + 5.0 * ember.size) + ember.phase));
			const float  rise    = static_cast<float>(std::fmod(t * 0.6 + ember.phase, 1.0)) * 0.03f * ember.size;
			const ImVec2 at      = Place(top_left, {q.x, q.y - rise}, size);
			const float  radius  = std::max(1.0f, (0.005f + 0.008f * ember.size) * size);
			const auto   green   = static_cast<int>(170.0f + 60.0f * ember.size);
			draw->AddCircleFilled(at, radius, IM_COL32(255, green, 80, static_cast<int>(255.0f * 0.75f * flicker)), 8);
		}
		// Head: the brush tip glowing like an ember, white at the core.
		{
			const ImVec2 q = RingPoint(enso.wobble, kmax, 0.0f);
			const float  flicker =
			    0.8f + 0.2f * static_cast<float>(std::sin(t * 9.0) * std::sin(t * 5.3));
			const float radius = ENSO_HEAD_RADIUS * flicker * size;
			AddRadialGlow(draw, Place(top_left, q, size), radius, radius * 0.25f, IM_COL32(255, 250, 225, 255),
			              IM_COL32(255, 170, 60, 230), IM_COL32(255, 160, 40, 0));
		}
	}
	// Drops: thrown as the brush passes; white-hot right behind it, gold once it is 18% further.
	for (const auto& splat: enso.splats) {
		if (splat.at > p) {
			continue;
		}
		const float heat    = std::max(0.0f, 1.0f - (p - splat.at) / ENSO_COOL_SPAN);
		const float flicker = 0.85f + 0.15f * static_cast<float>(std::sin(t * (3.0 + 4.0 * splat.size) +
		                                                                     splat.side * 9.0f));
		const float core    = splat.radius * (1.0f + 0.4f * heat) * size;
		const float blur    = (0.025f + 0.04f * heat) * size;
		const ImU32 hot     = IM_COL32(255, static_cast<int>(165.0f + 80.0f * heat), static_cast<int>(40.0f + 170.0f * heat),
		                               static_cast<int>(255.0f * 0.9f * flicker));
		const ImU32 glow    = IM_COL32(255, static_cast<int>(170.0f + 50.0f * heat), 70, 0);
		AddRadialGlow(draw, Place(top_left, splat.pos, size), core + blur, core, hot,
		              WithAlpha(glow, (0.55f + 0.45f * heat) * flicker * 0.7f), glow);
	}
	// The number (the mockup's brush font is not shipped: the launcher's display font, Cormorant Garamond).
	if (font != nullptr) {
		const ImVec2 pos {std::round(middle.x - text_dim.x * 0.5f), std::round(middle.y - text_dim.y * 0.5f)};
		draw->AddText(font, text_size, {pos.x, pos.y + u}, IM_COL32(0, 0, 0, 200), text);
		draw->AddText(font, text_size, pos, ENSO_ON_DARK, text);
	}
}

} // namespace Libs::Graphics::Brush
