#ifndef LAUNCHER_BRUSH_H
#define LAUNCHER_BRUSH_H

#include <QPointF>
#include <QVector>

#include <cstdint>

class QPainter;

// Hand-painted brush strokes of the launcher style v08 (design mockup, "brush engine"): a stroke is
// a bundle of bristles along a wobbly line, each bristle with its own offset, shade, opacity and dry gaps. Everything
// is seeded, so a stroke looks the same on every repaint. Sizes are logical pixels.
namespace LauncherBrush {

// Width along the stroke: Stroke (thin start, tapering end), Sweep (fast start, long fade), Fat (full almost to the
// ends).
enum class Pressure { Stroke, Sweep, Fat };

// Where the bristles run dry: End (only towards the end), Mid (gaps along the whole stroke).
enum class Dryness { End, Mid };

// Bristle colour: ginkgo gold, or the warm ink of the Play swipe (lighter than the ink bar it lies on).
enum class Colour { Gold, WarmInk };

struct Style {
	qreal         width     = 6.0;
	int           bristles  = 12;
	std::uint32_t seed      = 1;
	Pressure      pressure  = Pressure::Stroke;
	Dryness       dryness   = Dryness::End;
	Colour        colour    = Colour::Gold;
	qreal         alpha     = 1.0; // bristle opacity factor (clamped to 1 per bristle)
	qreal         dry_scale = 9.0; // length of the dry gaps, in points of the line
};

// Points of a hand-drawn line from a to b, one every step pixels: bowed by bow pixels in the middle (positive: to the
// right of the direction a -> b) and wobbling by up to wobble pixels.
QVector<QPointF> Line(QPointF a, QPointF b, std::uint32_t seed, qreal bow = 0.0, qreal wobble = 1.2, qreal step = 2.0);

// Paints the stroke along the points with the painter's current opacity.
void Paint(QPainter& painter, const QVector<QPointF>& points, const Style& style);

} // namespace LauncherBrush

#endif // LAUNCHER_BRUSH_H
