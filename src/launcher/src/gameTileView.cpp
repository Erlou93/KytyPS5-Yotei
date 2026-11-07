#include "gameTileView.h"

#include "launcherTheme.h"

#include <QFontMetrics>
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>
#include <QResizeEvent>
#include <QStyledItemDelegate>

#include <algorithm>

namespace {

// Tile metrics (docs/launcher/DESIGN-SPEC.md, Play): 156 px square cover, 8 px around it, name and "serial · version"
// below; tiles 8 px apart (three in a row at the design size).
constexpr int   COVER        = 156;
constexpr int   PAD          = 8;
constexpr int   TEXT_GAP     = 6;
constexpr int   NAME_LINE    = 20; // Body/Strong
constexpr int   META_GAP     = 2;
constexpr int   META_LINE    = 16; // Mono/Path
constexpr int   TILE_WIDTH   = COVER + 2 * PAD;
constexpr int   TILE_HEIGHT  = PAD + COVER + TEXT_GAP + NAME_LINE + META_GAP + META_LINE + PAD;
constexpr int   SPACING      = 8;
constexpr qreal TILE_RADIUS  = 8.0; // radius/md
constexpr qreal COVER_RADIUS = 6.0; // radius/sm
constexpr qreal META_TRACKING = -0.2; // "PPSA26344 · 01.512.000" fits the cover width
constexpr int   EMPTY_HEIGHT = 160;  // the empty-list message

// The serial and version columns of the game list (configurationListWidget.cpp).
constexpr int SERIAL_COLUMN  = 1;
constexpr int VERSION_COLUMN = 2;

QColor WithAlpha(QColor color, qreal alpha) {
	color.setAlphaF(static_cast<float>(alpha));
	return color;
}

// "GY" for "Ghost of Yotei": the cover of a game without icon0.png.
QString Initials(const QString& name) {
	QString initials;
	for (const auto& word: name.split(QLatin1Char(' '), Qt::SkipEmptyParts)) {
		if (word.at(0).isLetterOrNumber() && (word.at(0).isUpper() || word.at(0).isDigit())) {
			initials += word.at(0);
		}
		if (initials.size() == 2) {
			break;
		}
	}
	return initials.isEmpty() ? name.left(1).toUpper() : initials;
}

class GameTileDelegate: public QStyledItemDelegate {
public:
	explicit GameTileDelegate(GameTileView* view): QStyledItemDelegate(view), m_view(view) {}

	[[nodiscard]] QSize sizeHint(const QStyleOptionViewItem& /*option*/, const QModelIndex& /*index*/) const override {
		return {TILE_WIDTH, TILE_HEIGHT};
	}

	void paint(QPainter* p, const QStyleOptionViewItem& option, const QModelIndex& index) const override {
		using LauncherTheme::Color;
		const bool dark     = m_view->IsDark();
		const bool selected = (option.state & QStyle::State_Selected) != 0;
		const bool hover    = (option.state & QStyle::State_MouseOver) != 0;
		const auto tile     = QRectF(option.rect.topLeft(), QSizeF(TILE_WIDTH, TILE_HEIGHT));

		p->save();
		p->setRenderHint(QPainter::Antialiasing);
		p->setRenderHint(QPainter::SmoothPixmapTransform);

		const auto paper = Color(QStringLiteral("paper"));
		if (selected) {
			p->setPen(Qt::NoPen);
			p->setBrush(dark ? WithAlpha(paper, 0.12) : Color(QStringLiteral("bg/page")));
			p->drawRoundedRect(tile, TILE_RADIUS, TILE_RADIUS);
		} else if (hover) {
			p->setPen(Qt::NoPen);
			p->setBrush(dark ? WithAlpha(paper, 0.07) : WithAlpha(Color(QStringLiteral("mist")), 0.35));
			p->drawRoundedRect(tile, TILE_RADIUS, TILE_RADIUS);
		}

		// Cover: the game's icon0.png (cover crop), or its initials on a plain square.
		const QRectF cover(tile.left() + PAD, tile.top() + PAD, COVER, COVER);
		QPainterPath cover_shape;
		cover_shape.addRoundedRect(cover, COVER_RADIUS, COVER_RADIUS);
		const auto pixmap = index.data(GameTileView::CoverRole).value<QPixmap>();
		p->save();
		p->setClipPath(cover_shape);
		if (!pixmap.isNull()) {
			const qreal side = std::min(pixmap.width(), pixmap.height());
			p->drawPixmap(cover, pixmap,
			              QRectF((pixmap.width() - side) / 2.0, (pixmap.height() - side) / 2.0, side, side));
		} else {
			p->fillRect(cover, dark ? Color(QStringLiteral("ink-2")) : Color(QStringLiteral("mist")));
			auto font = LauncherTheme::TextStyle(QStringLiteral("Heading/Title"));
			font.setPixelSize(44);
			p->setFont(font);
			p->setPen(dark ? Color(QStringLiteral("text/on-dark-muted")) : Color(QStringLiteral("text/secondary")));
			p->drawText(cover, Qt::AlignCenter, Initials(index.data(Qt::DisplayRole).toString()));
		}
		if (index.data(GameTileView::RunningRole).toBool()) {
			// "Playing" badge on the cover's bottom edge.
			const auto  font = LauncherTheme::TextStyle(QStringLiteral("Label/Caption"));
			const auto  text = GameTileView::tr("Playing");
			const QRectF badge(cover.left() + 8, cover.bottom() - 8 - 20,
			                   QFontMetrics(font).horizontalAdvance(text) + 16, 20);
			p->setPen(Qt::NoPen);
			p->setBrush(WithAlpha(Color(QStringLiteral("ink")), 0.85));
			p->drawRoundedRect(badge, 10, 10);
			p->setFont(font);
			p->setPen(Color(QStringLiteral("text/on-dark")));
			p->drawText(badge, Qt::AlignCenter, text);
		}
		p->restore();
		p->setPen(QPen(dark ? WithAlpha(paper, 0.14) : Color(QStringLiteral("border/default")), 1.0));
		p->setBrush(Qt::NoBrush);
		p->drawRoundedRect(cover.adjusted(0.5, 0.5, -0.5, -0.5), COVER_RADIUS, COVER_RADIUS);

		// Name, then "serial · version".
		const QRectF name_rect(tile.left() + PAD, cover.bottom() + TEXT_GAP, COVER, NAME_LINE);
		const auto   name_font = LauncherTheme::TextStyle(QStringLiteral("Body/Strong"));
		p->setFont(name_font);
		p->setPen(dark ? Color(QStringLiteral("text/on-dark-strong")) : Color(QStringLiteral("text/primary")));
		p->drawText(name_rect, Qt::AlignLeft | Qt::AlignVCenter,
		            QFontMetrics(name_font).elidedText(index.data(Qt::DisplayRole).toString(), Qt::ElideRight, COVER));

		QStringList meta;
		const auto  serial  = index.siblingAtColumn(SERIAL_COLUMN).data(Qt::DisplayRole).toString();
		const auto  version = index.siblingAtColumn(VERSION_COLUMN).data(Qt::DisplayRole).toString();
		if (!serial.isEmpty()) {
			meta << serial;
		}
		if (!version.isEmpty() && version != QStringLiteral("—")) {
			meta << version;
		}
		const QRectF meta_rect(tile.left() + PAD, name_rect.bottom() + META_GAP, COVER, META_LINE);
		auto         meta_font = LauncherTheme::TextStyle(QStringLiteral("Mono/Path"));
		meta_font.setLetterSpacing(QFont::AbsoluteSpacing, META_TRACKING);
		p->setFont(meta_font);
		p->setPen(dark ? Color(QStringLiteral("text/on-dark-muted")) : Color(QStringLiteral("text/secondary")));
		p->drawText(meta_rect, Qt::AlignLeft | Qt::AlignVCenter,
		            QFontMetrics(meta_font).elidedText(meta.join(QStringLiteral(" · ")), Qt::ElideRight, COVER));
		p->restore();
	}

private:
	GameTileView* m_view;
};

} // namespace

GameTileView::GameTileView(QWidget* parent): QListView(parent) {
	setItemDelegate(new GameTileDelegate(this));
	setViewMode(QListView::IconMode);
	setFlow(QListView::LeftToRight);
	setWrapping(true);
	setResizeMode(QListView::Adjust);
	setMovement(QListView::Static);
	setUniformItemSizes(true);
	setSpacing(SPACING);
	setSelectionMode(QAbstractItemView::SingleSelection);
	setSelectionBehavior(QAbstractItemView::SelectRows);
	setSelectionRectVisible(false);
	setEditTriggers(QAbstractItemView::NoEditTriggers);
	setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setFrameShape(QFrame::NoFrame);
	setMouseTracking(true);
	viewport()->setAttribute(Qt::WA_Hover);
	viewport()->setAutoFillBackground(false);
	viewport()->setCursor(Qt::PointingHandCursor);
	setContextMenuPolicy(Qt::CustomContextMenu);
	setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
}

int GameTileView::ShownCount() const {
	int shown = 0;
	if (model() != nullptr) {
		for (int row = 0; row < model()->rowCount(rootIndex()); row++) {
			shown += isRowHidden(row) ? 0 : 1;
		}
	}
	return shown;
}

QSize GameTileView::sizeHint() const {
	const int columns = std::max(1, (viewport()->width() - SPACING) / (TILE_WIDTH + SPACING));
	const int rows    = (ShownCount() + columns - 1) / columns;
	return {TILE_WIDTH + 2 * SPACING, rows == 0 ? EMPTY_HEIGHT : SPACING + rows * (TILE_HEIGHT + SPACING)};
}

QSize GameTileView::minimumSizeHint() const {
	return {TILE_WIDTH + 2 * SPACING, EMPTY_HEIGHT};
}

void GameTileView::resizeEvent(QResizeEvent* event) {
	QListView::resizeEvent(event);
	if (event->size().width() != event->oldSize().width()) {
		updateGeometry(); // more or fewer tiles in a row
	}
}

void GameTileView::SetDark(bool dark) {
	m_dark = dark;
	viewport()->update();
}

void GameTileView::SetEmptyText(const QString& title, const QString& description) {
	m_empty_title       = title;
	m_empty_description = description;
	viewport()->update();
}

void GameTileView::paintEvent(QPaintEvent* event) {
	QListView::paintEvent(event);
	if (ShownCount() > 0) {
		return;
	}
	// No tile shown: the reason and what to do, in the middle.
	using LauncherTheme::Color;
	QPainter   p(viewport());
	const auto title_font       = LauncherTheme::TextStyle(QStringLiteral("Heading/Section"));
	const auto description_font = LauncherTheme::TextStyle(QStringLiteral("Body/Description"));
	const auto area             = viewport()->rect().adjusted(24, 0, -24, 0);
	const int  width            = std::min(area.width(), 420);
	const auto description      = QFontMetrics(description_font)
	                             .boundingRect(QRect(0, 0, width, 1000), Qt::AlignHCenter | Qt::TextWordWrap,
	                                           m_empty_description);
	const int  height = 32 + 8 + description.height();
	const QRect block(area.left() + (area.width() - width) / 2, area.top() + (area.height() - height) / 2, width, height);
	p.setFont(title_font);
	p.setPen(m_dark ? Color(QStringLiteral("text/on-dark")) : Color(QStringLiteral("text/primary")));
	p.drawText(QRect(block.left(), block.top(), width, 32), Qt::AlignCenter, m_empty_title);
	p.setFont(description_font);
	p.setPen(m_dark ? Color(QStringLiteral("text/on-dark-muted")) : Color(QStringLiteral("text/secondary")));
	p.drawText(QRect(block.left(), block.top() + 40, width, description.height()), Qt::AlignHCenter | Qt::TextWordWrap,
	           m_empty_description);
}
