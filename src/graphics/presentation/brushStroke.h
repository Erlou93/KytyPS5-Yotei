#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_BRUSHSTROKE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_BRUSHSTROKE_H_

#include "imgui.h"

#include <array>
#include <cstdint>
#include <vector>

// Brush-painted overlay parts in the launcher's look (ink, paper, ginkgo gold), ported from the HTML mockups in
// design mockups: the shader precompile badge (G9) and the F2 panel frame (style
// style 2 "Narożniki"). Geometry is seeded and built once; a frame only animates and emits ImDrawList primitives.
namespace Libs::Graphics::Brush {

// The mockups' seeded generator (mulberry32), bit for bit: the same seed gives the same shape as the preview.
class Random {
public:
	explicit Random(uint32_t seed): m_state(seed) {}
	float Next() noexcept;

private:
	uint32_t m_state;
};

// The mockups' hash(): a 32-bit finaliser mapped to [0, 1).
[[nodiscard]] float Hash01(uint32_t value) noexcept;

// The mockups' noise1(): smooth value noise over sample indices, one random value every `scale` samples. Building it
// takes ceil(length / scale) + 3 values from `random`.
class Noise {
public:
	Noise() = default;
	Noise(Random& random, int length, float scale) noexcept;
	[[nodiscard]] float operator()(float k) const noexcept;

private:
	std::array<float, 128> m_points {};
	int                    m_count = 0;
	float                  m_scale = 1.0f;
};

// CSS hsl(): hue in degrees, saturation and lightness 0..1.
[[nodiscard]] ImU32 Hsl(float hue, float saturation, float lightness, float alpha = 1.0f) noexcept;
// The colour with its alpha replaced (0..1).
[[nodiscard]] ImU32 WithAlpha(ImU32 color, float alpha) noexcept;

// One brush stroke as precomputed bristle polylines; drawing emits only the runs where a bristle touches the paper.
struct Stroke {
	struct Bristle {
		ImU32 color     = 0;    // opaque
		float alpha     = 1.0f; // before the draw call's multiplier
		float thickness = 1.0f; // in the stroke's units
	};
	int                  samples = 0; // points per bristle
	std::vector<Bristle> bristles;
	std::vector<ImVec2>  points; // bristle-major, `samples` per bristle
	std::vector<uint8_t> pen;    // 1 where the bristle touches the paper

	void Clear() noexcept;
	// Draws samples [0, last] of every bristle at origin + point * scale, keeping every `stride`-th point and the ends
	// of each run. A non-zero `color` replaces the bristles' colours; `alpha` multiplies their alpha.
	void Draw(ImDrawList* draw, int last, ImVec2 origin, float scale, int stride, float alpha,
	          ImU32 color = 0) const;
};

// Soft radial gradient: `inner` at the centre, `middle` at `middle_radius`, `outer` at `radius` (give it zero alpha).
void AddRadialGlow(ImDrawList* draw, ImVec2 centre, float radius, float middle_radius, ImU32 inner, ImU32 middle,
                   ImU32 outer);

// The F2 panel frame "Narożniki": gold brush strokes along the top-left and bottom-right corners of the rect only.
// `unit` is pixels per mockup pixel (the panel's scale). The strokes reach ~6 units outside the rect.
void DrawCornerFrame(ImDrawList* draw, ImVec2 rect_min, ImVec2 rect_max, float unit);

// The shader precompile badge G9 "Żar ze stygnącymi rozpryskami": a gold ensō that the brush paints as `progress`
// (0..1) grows, an ember at the brush tip with sparks behind it, glowing drops around the stroke that start white-hot
// and cool to gold, and `percent` in the middle. `size` is the badge's square side in pixels (56 at 1080p); `seconds`
// is a steady clock that drives the motion, which goes on while the progress stalls.
void DrawShaderEnso(ImDrawList* draw, ImVec2 top_left, float size, float progress, double seconds, ImFont* font,
                    int percent);

} // namespace Libs::Graphics::Brush

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_BRUSHSTROKE_H_
