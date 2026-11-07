#include "launcherWidgets.h"

#include "launcherBrush.h"
#include "launcherTheme.h"

#include <QEnterEvent>
#include <QFontMetrics>
#include <QHash>
#include <QIcon>
#include <QLinearGradient>
#include <QPainter>
#include <QPainterPath>
#include <QRegion>
#include <QStyle>

#include <algorithm>
#include <cmath>

namespace {

// The part of the image to draw: the design crop, trimmed around its centre to the target's aspect ratio.
QRectF CoverSource(const QPixmap& art, const QRectF& crop, const QSizeF& target) {
	QRectF source(crop.x() * art.width(), crop.y() * art.height(), crop.width() * art.width(),
	              crop.height() * art.height());
	if (target.isEmpty() || source.isEmpty()) {
		return source;
	}
	const qreal target_aspect = target.width() / target.height();
	const qreal source_aspect = source.width() / source.height();
	const auto  centre        = source.center();
	if (target_aspect > source_aspect) {
		source.setHeight(source.width() / target_aspect);
	} else {
		source.setWidth(source.height() * target_aspect);
	}
	source.moveCenter(centre);
	return source;
}

QColor WithAlpha(QColor color, qreal alpha) {
	color.setAlphaF(static_cast<float>(alpha));
	return color;
}

// A stable seed for a widget's brush strokes (the same shape on every repaint, a different one per widget).
std::uint32_t StrokeSeed(const QWidget& widget) {
	return static_cast<std::uint32_t>(qHash(widget.objectName(), 0));
}

// The heading underlines of the mockup (version 08), relative to the text: x from the text's left edge (end: from its
// right edge when end_from_right), drops below the baseline, bow and wobble of the line, stroke width and bristles.
struct HeadingStroke {
	qreal start;
	qreal end;
	bool  end_from_right;
	qreal drop_start;
	qreal drop_end;
	qreal bow;
	qreal wobble;
	qreal width;
	int   bristles;

	// How far the stroke can reach below the baseline (outer bristles, wobble, line width).
	[[nodiscard]] qreal Reach() const {
		return std::max(drop_start, drop_end) + bow + wobble + width * 0.575 + width / bristles * 1.33 + 0.5;
	}
};

// Heading/Title: past the end of the text, 10 px broad. Heading/Section: 92 px long, 4.5 px.
constexpr HeadingStroke TITLE_STROKE {-6.0, 30.0, true, 10.0, 6.0, 3.0, 1.0, 10.0, 18};
constexpr HeadingStroke SECTION_STROKE {-2.0, 90.0, false, 8.7, 7.7, 1.0, 0.6, 4.5, 9};

const HeadingStroke* HeadingStrokeOf(const QWidget& label) {
	const auto text_style = label.property("textStyle").toString();
	if (text_style == QLatin1String("Heading/Title")) {
		return &TITLE_STROKE;
	}
	if (text_style == QLatin1String("Heading/Section")) {
		return &SECTION_STROKE;
	}
	return nullptr;
}

} // namespace

ArtBackground::ArtBackground(QWidget* parent): QWidget(parent) {
	setAttribute(Qt::WA_StyledBackground, false);
}

void ArtBackground::SetArt(const QString& resource, const QRectF& crop) {
	m_art  = QPixmap(resource);
	m_crop = crop;
	update();
}

void ArtBackground::SetScrim(qreal alpha, qreal stop) {
	m_scrim_alpha = alpha;
	m_scrim_stop  = stop;
	update();
}

void ArtBackground::SetBottomFade(int height) {
	m_bottom_fade = height;
	update();
}

void ArtBackground::SetTopWash(qreal alpha, int height) {
	m_top_wash_alpha  = alpha;
	m_top_wash_height = height;
	update();
}

void ArtBackground::SetPicture(const QPixmap& picture, qreal ink_alpha) {
	m_picture     = picture;
	m_picture_ink = ink_alpha;
	update();
}

void ArtBackground::SetCardShadows() {
	for (auto* child: findChildren<QWidget*>(Qt::FindDirectChildrenOnly)) {
		if (child->property("card").toBool() && !m_cards.contains(child)) {
			m_cards.append(child);
			child->installEventFilter(this);
		}
	}
	update();
}

bool ArtBackground::eventFilter(QObject* watched, QEvent* event) {
	switch (event->type()) {
		case QEvent::Move:
		case QEvent::Resize:
		case QEvent::Show:
		case QEvent::Hide: update(); break; // a card's shadow reaches past the card
		default: break;
	}
	return QWidget::eventFilter(watched, event);
}

void ArtBackground::PaintCardShadows(QPainter& painter) const {
	// The CSS blur of 30 px is a Gaussian of sigma 15 px: rounded rectangles grown or shrunk in 5 px steps over
	// -30..+30 px, each with the share of the 12 % alpha that the Gaussian puts in its step, add up to that blur.
	constexpr qreal OFFSET = 10.0;
	constexpr qreal SIGMA  = 15.0;
	constexpr qreal ALPHA  = 0.12;
	constexpr qreal RADIUS = 2.0;
	constexpr qreal STEP   = 5.0;
	constexpr int   STEPS  = 12;
	const auto      ink    = LauncherTheme::Color(QStringLiteral("ink"));
	const auto      cdf    = [](qreal x) { return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))); };
	for (const auto& card: m_cards) {
		if (card == nullptr || !card->isVisible()) {
			continue;
		}
		const QRectF box = QRectF(card->geometry()).translated(0.0, OFFSET);
		painter.save();
		painter.setRenderHint(QPainter::Antialiasing);
		painter.setPen(Qt::NoPen);
		// As in CSS, nothing under the card itself (it is translucent).
		painter.setClipRegion(QRegion(rect()).subtracted(QRegion(card->geometry())));
		for (int k = 0; k < STEPS; k++) {
			const qreal grow  = 2.0 * SIGMA - STEP * (k + 0.5); // the widest first
			const qreal share = cdf((grow + STEP / 2.0) / SIGMA) - cdf((grow - STEP / 2.0) / SIGMA);
			const auto  layer = box.adjusted(-grow, -grow, grow, grow);
			if (layer.isEmpty()) {
				continue;
			}
			const qreal radius = std::max(0.0, RADIUS + grow);
			painter.setBrush(WithAlpha(ink, ALPHA * share));
			painter.drawRoundedRect(layer, radius, radius);
		}
		painter.restore();
	}
}

void ArtBackground::paintEvent(QPaintEvent* /*event*/) {
	QPainter p(this);
	if (!m_picture.isNull()) {
		const auto ink = LauncherTheme::Color(QStringLiteral("ink"));
		p.fillRect(rect(), ink);
		p.setRenderHint(QPainter::SmoothPixmapTransform);
		p.drawPixmap(QRectF(rect()), m_picture, CoverSource(m_picture, {0.0, 0.0, 1.0, 1.0}, size()));
		p.fillRect(rect(), WithAlpha(ink, m_picture_ink));
		PaintCardShadows(p);
		return;
	}
	const auto page = LauncherTheme::Color(QStringLiteral("bg/page"));
	p.fillRect(rect(), page);
	if (!m_art.isNull()) {
		p.setRenderHint(QPainter::SmoothPixmapTransform);
		p.drawPixmap(QRectF(rect()), m_art, CoverSource(m_art, m_crop, size()));
	}
	if (m_scrim_alpha > 0.0 && m_scrim_stop > 0.0) {
		QLinearGradient scrim(0, 0, width(), 0);
		scrim.setColorAt(0.0, WithAlpha(page, m_scrim_alpha));
		scrim.setColorAt(m_scrim_stop, WithAlpha(page, 0.0));
		p.fillRect(rect(), scrim);
	}
	if (m_top_wash_alpha > 0.0 && m_top_wash_height > 0) {
		const QRect     band(0, 0, width(), m_top_wash_height);
		QLinearGradient wash(band.topLeft(), band.bottomLeft());
		wash.setColorAt(0.0, WithAlpha(page, m_top_wash_alpha));
		wash.setColorAt(1.0, WithAlpha(page, 0.0));
		p.fillRect(band, wash);
	}
	if (m_bottom_fade > 0) {
		const QRect     band(0, height() - m_bottom_fade, width(), m_bottom_fade);
		QLinearGradient fade(band.topLeft(), band.bottomLeft());
		fade.setColorAt(0.0, WithAlpha(page, 0.0));
		fade.setColorAt(1.0, page);
		p.fillRect(band, fade);
	}
	PaintCardShadows(p);
}

ArtButton::ArtButton(QWidget* parent): QPushButton(parent) {
	setAttribute(Qt::WA_Hover);
	setCursor(Qt::PointingHandCursor);
}

void ArtButton::SetArt(const QString& resource, const QRectF& crop) {
	m_art  = QPixmap(resource);
	m_crop = crop;
	update();
}

void ArtButton::SetBrush(const QString& brush) {
	m_brush = brush;
	update();
}

void ArtButton::changeEvent(QEvent* event) {
	// The swipe is painted by the parent bar and fades with the button when it is disabled.
	if (event->type() == QEvent::EnabledChange && m_brush == QLatin1String("swipe") && parentWidget() != nullptr) {
		parentWidget()->update();
	}
	QPushButton::changeEvent(event);
}

QSize ArtButton::sizeHint() const {
	constexpr int PADDING = 16;
	const int     width   = QFontMetrics(font()).horizontalAdvance(text()) + 2 * PADDING;
	return {std::max(width, minimumWidth()), std::max(40, minimumHeight())};
}

void ArtButton::enterEvent(QEnterEvent* event) {
	update();
	QPushButton::enterEvent(event);
}

void ArtButton::leaveEvent(QEvent* event) {
	update();
	QPushButton::leaveEvent(event);
}

void ArtButton::paintEvent(QPaintEvent* /*event*/) {
	constexpr qreal RADIUS = 6.0; // radius/sm
	constexpr qreal BORDER = 1.5;

	const bool enabled = isEnabled();
	const bool hover   = enabled && underMouse();
	const bool pressed = enabled && isDown();

	QPainter p(this);
	p.setRenderHint(QPainter::Antialiasing);
	p.setRenderHint(QPainter::SmoothPixmapTransform);
	if (!enabled) {
		p.setOpacity(0.4);
	}
	if (!m_brush.isEmpty()) {
		PaintBrushLook(p, hover, pressed);
		return;
	}

	QPainterPath shape;
	shape.addRoundedRect(QRectF(rect()), RADIUS, RADIUS);
	p.save();
	p.setClipPath(shape);
	const auto ink = LauncherTheme::Color(QStringLiteral("bg/bar"));
	if (!m_art.isNull()) {
		// Figma stretches the same image region into every button size.
		const QRectF source(m_crop.x() * m_art.width(), m_crop.y() * m_art.height(), m_crop.width() * m_art.width(),
		                    m_crop.height() * m_art.height());
		p.drawPixmap(QRectF(rect()), m_art, source);
	}
	p.fillRect(rect(), WithAlpha(ink, pressed ? 0.70 : (hover ? 0.48 : 0.60)));
	p.restore();

	const auto frame = LauncherTheme::Color(hover || pressed ? QStringLiteral("accent/default") : QStringLiteral("accent/hover"));
	p.setPen(QPen(frame, BORDER));
	p.setBrush(Qt::NoBrush);
	p.drawRoundedRect(QRectF(rect()).adjusted(BORDER / 2, BORDER / 2, -BORDER / 2, -BORDER / 2), RADIUS - BORDER / 2,
	                  RADIUS - BORDER / 2);

	p.setPen(LauncherTheme::Color(hover || pressed ? QStringLiteral("text/on-dark-strong") : QStringLiteral("text/on-dark")));
	p.setFont(font());
	p.drawText(rect(), Qt::AlignCenter, text());
}

void ArtButton::PaintBrushLook(QPainter& painter, bool hover, bool pressed) {
	using LauncherTheme::Color;
	const auto gold = Color(QStringLiteral("accent/default"));
	QColor     label;
	if (m_brush == QLatin1String("swipe")) {
		// PLAY: a gold brush underline under the gold label (8.11: instead of the mockup's warm-ink swipe).
		LauncherBrush::Style style;
		style.width    = 4.5;
		style.bristles = 9;
		style.seed     = StrokeSeed(*this) ^ 0x9E3779B9u;
		style.alpha    = hover || pressed ? 1.3 : 1.0;
		const qreal y  = height() - 6.0;
		const auto  line = LauncherBrush::Line({width() * 0.18, y}, {width() * 0.82, y - 1.0}, StrokeSeed(*this), 0.8, 0.5);
		LauncherBrush::Paint(painter, line, style);
		label = pressed ? Color(QStringLiteral("accent/hover")) : (hover ? gold.lighter(118) : gold);
	} else {
		// Mockup: a thin gold stroke 4 px above the bottom edge, along the whole button, under a paper label.
		LauncherBrush::Style style;
		style.width    = 3.0;
		style.bristles = 7;
		style.seed     = StrokeSeed(*this) ^ 0x9E3779B9u;
		style.alpha    = hover || pressed ? 1.3 : 1.0;
		const qreal y  = height() - 4.0;
		const auto  line =
		    LauncherBrush::Line({0.0, y}, {static_cast<qreal>(width()), y - 1.0}, StrokeSeed(*this), 0.6, 0.5);
		LauncherBrush::Paint(painter, line, style);
		label = pressed ? Color(QStringLiteral("accent/hover"))
		                : (hover ? gold : Color(QStringLiteral("text/on-dark-strong")));
	}
	// v0.1.6: the shaders' state (the dynamic property shaderState): a red frame until they are prepared, green when done.
	if (const auto state = property("shaderState").toString(); !state.isEmpty()) {
		painter.save();
		painter.setRenderHint(QPainter::Antialiasing, true);
		painter.setPen(QPen(state == QLatin1String("done") ? QColor(0x4f, 0x8a, 0x3c) : QColor(0xb8, 0x43, 0x2f), 2.0));
		painter.setBrush(Qt::NoBrush);
		painter.drawRoundedRect(QRectF(rect()).adjusted(1.0, 1.0, -1.0, -1.0), 4.0, 4.0);
		painter.restore();
	}
	painter.setPen(label);
	painter.setFont(font());
	painter.drawText(rect(), Qt::AlignCenter, text());
}

BrushBar::BrushBar(QWidget* parent): QFrame(parent) {}

void BrushBar::paintEvent(QPaintEvent* event) {
	QFrame::paintEvent(event);
	QPainter p(this);
	for (const auto* button: findChildren<ArtButton*>(Qt::FindDirectChildrenOnly)) {
		if (!button->isVisible() || button->Brush() != QLatin1String("ink-swipe")) { // the mockup's swipe (unused now)
			continue;
		}
		// Mockup: from 30 px before the button to 20 px after it, 1.05 times its height, fat pressure, dry gaps along
		// the way; at 40 % like the disabled button's label.
		const QRectF         box = button->geometry();
		const qreal          y   = box.center().y();
		LauncherBrush::Style style;
		style.width     = box.height() * 1.05;
		style.bristles  = 30;
		style.seed      = StrokeSeed(*button) ^ 0x9E3779B9u;
		style.pressure  = LauncherBrush::Pressure::Fat;
		style.dryness   = LauncherBrush::Dryness::Mid;
		style.colour    = LauncherBrush::Colour::WarmInk;
		style.alpha     = 1.4;
		style.dry_scale = 20.0;
		p.setOpacity(button->isEnabled() ? 1.0 : 0.4);
		const auto line =
		    LauncherBrush::Line({box.left() - 30.0, y + 2.0}, {box.right() + 20.0, y - 2.0}, StrokeSeed(*button), 3.0, 1.2);
		LauncherBrush::Paint(p, line, style);
	}
}

BrushHeading::BrushHeading(QWidget* parent): QLabel(parent) {}

bool BrushHeading::event(QEvent* event) {
	const bool result = QLabel::event(event);
	// QFrame resets the contents margins to its frame widths (QSS border and padding) when it is polished, restyled
	// or reparented: the room goes back on after it; a font change moves the baseline.
	switch (event->type()) {
		case QEvent::Polish:
		case QEvent::StyleChange:
		case QEvent::ParentChange:
		case QEvent::FontChange: UpdateRoom(); break;
		default: break;
	}
	return result;
}

// Room below the text for the part of the stroke that reaches past the font's descent, on top of the frame's own
// bottom width.
void BrushHeading::UpdateRoom() {
	const auto* stroke       = HeadingStrokeOf(*this);
	const qreal reach        = stroke == nullptr ? 0.0 : stroke->Reach() - QFontMetricsF(font()).descent();
	const int   room         = std::max(0, static_cast<int>(std::ceil(reach)));
	const int   frame_bottom = frameRect().bottom() - contentsRect().bottom();
	const auto  margins      = contentsMargins();
	if (margins.bottom() != frame_bottom + room) {
		setContentsMargins(margins.left(), margins.top(), margins.right(), frame_bottom + room);
	}
}

void BrushHeading::paintEvent(QPaintEvent* event) {
	if (const auto* stroke = HeadingStrokeOf(*this); stroke != nullptr && !text().isEmpty()) {
		// The text's place as QLabel draws it (plain text, one line): aligned in the contents rectangle.
		const QFontMetricsF fm(font());
		const auto          contents   = QRectF(contentsRect()).adjusted(margin(), margin(), -margin(), -margin());
		const qreal         text_width = std::min(fm.horizontalAdvance(text()), contents.width());
		const auto          align      = QStyle::visualAlignment(layoutDirection(), alignment());
		qreal               left       = contents.left();
		if (align.testFlag(Qt::AlignRight)) {
			left = contents.right() - text_width;
		} else if (align.testFlag(Qt::AlignHCenter)) {
			left = contents.center().x() - text_width / 2.0;
		}
		qreal top = contents.top() + (contents.height() - fm.height()) / 2.0;
		if (align.testFlag(Qt::AlignTop)) {
			top = contents.top();
		} else if (align.testFlag(Qt::AlignBottom)) {
			top = contents.bottom() - fm.height();
		}
		const qreal baseline = top + fm.ascent();
		const qreal x0       = std::max(1.0, left + stroke->start);
		const qreal x1       = std::min(width() - 1.0, (stroke->end_from_right ? left + text_width : left) + stroke->end);
		if (x1 > x0) {
			LauncherBrush::Style style;
			style.width    = stroke->width;
			style.bristles = stroke->bristles;
			style.seed     = StrokeSeed(*this) ^ 0x9E3779B9u;
			const auto line = LauncherBrush::Line({x0, baseline + stroke->drop_start}, {x1, baseline + stroke->drop_end},
			                                      StrokeSeed(*this), stroke->bow, stroke->wobble);
			QPainter   p(this);
			LauncherBrush::Paint(p, line, style);
		}
	}
	QLabel::paintEvent(event);
}

NavButton::NavButton(QWidget* parent): QPushButton(parent) {
	setCheckable(true);
	setAttribute(Qt::WA_Hover);
	setCursor(Qt::PointingHandCursor);
}

void NavButton::SetAccent(bool accent) {
	m_accent = accent;
	update();
}

void NavButton::enterEvent(QEnterEvent* event) {
	update();
	QPushButton::enterEvent(event);
}

void NavButton::leaveEvent(QEvent* event) {
	update();
	QPushButton::leaveEvent(event);
}

void NavButton::paintEvent(QPaintEvent* /*event*/) {
	constexpr int PADDING_LEFT = 24; // space/24
	constexpr int DOT          = 6;
	constexpr int GAP          = 12; // space/12

	const bool active = isChecked();

	QPainter p(this);
	p.setRenderHint(QPainter::Antialiasing);
	if (active) {
		// Mockup (version 08): a gold stroke 10 px from the left edge, from 6 px below the top to 4 px above the bottom.
		LauncherBrush::Style style;
		style.width    = 6.0;
		style.bristles = 10;
		style.seed     = StrokeSeed(*this) ^ 0x9E3779B9u;
		style.pressure = LauncherBrush::Pressure::Sweep;
		LauncherBrush::Paint(p, LauncherBrush::Line({10.0, 6.0}, {9.0, height() - 4.0}, StrokeSeed(*this), 1.0, 0.6), style);
	} else if (underMouse()) {
		p.fillRect(rect(), WithAlpha(LauncherTheme::Color(QStringLiteral("bg/sidebar-active")), 0.5));
	}

	const auto dot = active ? (m_accent ? QStringLiteral("accent/default") : QStringLiteral("nav/dot-active"))
	                        : QStringLiteral("nav/dot");
	p.setPen(Qt::NoPen);
	p.setBrush(LauncherTheme::Color(dot));
	p.drawEllipse(QRectF(PADDING_LEFT, (height() - DOT) / 2.0, DOT, DOT));

	p.setPen(LauncherTheme::Color(active ? QStringLiteral("text/on-dark") : QStringLiteral("text/on-dark-muted")));
	p.setFont(font());
	p.drawText(rect().adjusted(PADDING_LEFT + DOT + GAP, 0, 0, 0), Qt::AlignVCenter | Qt::AlignLeft, text());
}

StatusIcon::StatusIcon(QWidget* parent): QWidget(parent) {
	setFixedSize(20, 20);
}

void StatusIcon::SetStatus(Status status) {
	m_status = status;
	update();
}

void StatusIcon::paintEvent(QPaintEvent* /*event*/) {
	constexpr int GLYPH = 12;

	auto color = QStringLiteral("status/success");
	auto glyph = QStringLiteral(":/icons/status-ok-glyph.svg");
	if (m_status == Status::Warn) {
		color = QStringLiteral("status/warning");
		glyph = QStringLiteral(":/icons/status-warn-glyph.svg");
	} else if (m_status == Status::Error) {
		color = QStringLiteral("status/error");
		glyph = QStringLiteral(":/icons/status-error-glyph.svg");
	}

	QPainter p(this);
	p.setRenderHint(QPainter::Antialiasing);
	p.setPen(Qt::NoPen);
	p.setBrush(LauncherTheme::Color(color));
	p.drawEllipse(QRectF(rect()));
	QIcon(glyph).paint(&p, QRect((width() - GLYPH) / 2, (height() - GLYPH) / 2, GLYPH, GLYPH));
}
