#include "launcherBrush.h"

#include <QColor>
#include <QPainter>
#include <QPainterPath>
#include <QPen>

#include <algorithm>
#include <cmath>
#include <vector>

namespace LauncherBrush {

namespace {

constexpr qreal PI = 3.14159265358979323846;

// Seeded generator (mulberry32), values in [0, 1).
class Random {
public:
	explicit Random(std::uint32_t seed): m_state(seed) {}

	qreal Next() {
		m_state += 0x6D2B79F5u;
		std::uint32_t t = m_state;
		t = (t ^ (t >> 15u)) * (t | 1u);
		t = (t + (t ^ (t >> 7u)) * (t | 61u)) ^ t;
		return static_cast<qreal>(t ^ (t >> 14u)) / 4294967296.0;
	}

private:
	std::uint32_t m_state;
};

// Smooth value noise over the points 0..count-1 of a line, one random value every scale points (values 0..1).
class Noise {
public:
	Noise(Random& random, int count, qreal scale): m_scale(scale) {
		const int values = static_cast<int>(std::ceil(count / scale)) + 3;
		m_values.reserve(values);
		for (int i = 0; i < values; i++) {
			m_values.push_back(random.Next());
		}
	}

	[[nodiscard]] qreal operator()(int k) const {
		const qreal x = k / m_scale;
		const auto  i = static_cast<size_t>(x);
		const qreal f = x - static_cast<qreal>(i);
		const qreal u = f * f * (3.0 - 2.0 * f);
		return m_values[i] * (1.0 - u) + m_values[i + 1] * u;
	}

private:
	qreal              m_scale;
	std::vector<qreal> m_values;
};

qreal PressureAt(Pressure pressure, qreal f) {
	switch (pressure) {
		case Pressure::Stroke: return f < 0.05 ? 0.5 + 10.0 * f : (f > 0.8 ? std::max(0.18, 1.0 - (f - 0.8) * 4.0) : 1.0 - 0.12 * f);
		case Pressure::Sweep: return f < 0.05 ? 0.55 + 9.0 * f : std::max(0.25, 1.0 - 0.7 * (f - 0.05));
		case Pressure::Fat: return f < 0.04 ? 0.7 + 7.0 * f : (f > 0.85 ? std::max(0.3, 1.0 - (f - 0.85) * 4.0) : 1.0);
	}
	return 1.0;
}

// A bristle leaves a gap where its dry noise is below this threshold (raised for the outer bristles).
qreal DrynessAt(Dryness dryness, qreal f) {
	switch (dryness) {
		case Dryness::End: return f > 0.72 ? (f - 0.72) * 1.7 : 0.03;
		case Dryness::Mid: return 0.1 + 0.2 * f;
	}
	return 0.0;
}

// The mockup's CSS colours: gold hsl(41, 66..84 %, 40..62 %), warm ink hsl(32, 10 %, 17..29 %), by bristle shade.
QColor ColourAt(Colour colour, qreal shade) {
	if (colour == Colour::WarmInk) {
		return QColor::fromHslF(32.0f / 360.0f, 0.10f, static_cast<float>(0.17 + 0.12 * shade));
	}
	return QColor::fromHslF(41.0f / 360.0f, static_cast<float>(0.66 + 0.18 * shade), static_cast<float>(0.40 + 0.22 * shade));
}

struct Bristle {
	qreal offset; // -1..1 across the stroke width
	qreal alpha;
	qreal width; // line width factor
	qreal shade; // 0..1 colour shade
	Noise dry;
	Noise wobble;
};

} // namespace

QVector<QPointF> Line(QPointF a, QPointF b, std::uint32_t seed, qreal bow, qreal wobble, qreal step) {
	const qreal dx     = b.x() - a.x();
	const qreal dy     = b.y() - a.y();
	const qreal length = std::hypot(dx, dy);
	if (length <= 0.0 || step <= 0.0) {
		return {};
	}
	const int   count = std::max(3, static_cast<int>(std::lround(length / step)));
	Random      random(seed);
	const Noise noise(random, count, 14.0);
	const qreal nx = -dy / length;
	const qreal ny = dx / length;

	QVector<QPointF> points(count);
	for (int i = 0; i < count; i++) {
		const qreal f = i / static_cast<qreal>(count - 1);
		const qreal d = bow * std::sin(PI * f) + (noise(i) - 0.5) * wobble * 2.0;
		points[i]     = {a.x() + dx * f + nx * d, a.y() + dy * f + ny * d};
	}
	return points;
}

void Paint(QPainter& painter, const QVector<QPointF>& points, const Style& style) {
	const int count = static_cast<int>(points.size());
	if (count < 2 || style.bristles < 1) {
		return;
	}

	Random               random(style.seed);
	std::vector<Bristle> bristles;
	bristles.reserve(style.bristles);
	for (int j = 0; j < style.bristles; j++) {
		const qreal offset =
		    style.bristles == 1 ? 0.0 : (j / static_cast<qreal>(style.bristles - 1) * 2.0 - 1.0) * (1.0 - 0.1 * random.Next());
		const qreal alpha = (0.5 + 0.5 * random.Next()) * (1.0 - 0.3 * std::abs(offset));
		const qreal width = 0.6 + 0.8 * random.Next();
		const qreal shade = random.Next();
		Noise       dry(random, count, style.dry_scale);
		Noise       wobble(random, count, 11.0);
		bristles.push_back({offset, alpha, width, shade, std::move(dry), std::move(wobble)});
	}

	std::vector<QPointF> normals(count);
	for (int i = 0; i < count; i++) {
		const auto& a      = points[std::max(0, i - 1)];
		const auto& b      = points[std::min(count - 1, i + 1)];
		const qreal dx     = b.x() - a.x();
		const qreal dy     = b.y() - a.y();
		const qreal length = std::hypot(dx, dy);
		normals[i]         = length > 0.0 ? QPointF(-dy / length, dx / length) : QPointF(0.0, 0.0);
	}

	painter.save();
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setBrush(Qt::NoBrush);
	for (const auto& bristle: bristles) {
		QPainterPath path;
		bool         pen_down = false;
		for (int i = 0; i < count; i++) {
			const qreal f = i / static_cast<qreal>(count - 1);
			if (bristle.dry(i) < DrynessAt(style.dryness, f) + 0.12 * std::abs(bristle.offset)) {
				pen_down = false;
				continue;
			}
			const qreal   width  = style.width * PressureAt(style.pressure, f);
			const qreal   offset = (bristle.offset + (bristle.wobble(i) - 0.5) * 0.3) * width / 2.0;
			const QPointF point  = points[i] + normals[i] * offset;
			if (pen_down) {
				path.lineTo(point);
			} else {
				path.moveTo(point);
				pen_down = true;
			}
		}
		auto colour = ColourAt(style.colour, bristle.shade);
		colour.setAlphaF(static_cast<float>(std::min(1.0, bristle.alpha * style.alpha)));
		painter.setPen(QPen(colour, std::max(0.7, style.width / style.bristles * 1.9 * bristle.width), Qt::SolidLine,
		                    Qt::RoundCap, Qt::RoundJoin));
		painter.drawPath(path);
	}
	painter.restore();
}

} // namespace LauncherBrush
