#ifndef LAUNCHER_WIDGETS_H
#define LAUNCHER_WIDGETS_H

#include <QFrame>
#include <QLabel>
#include <QList>
#include <QPixmap>
#include <QPointer>
#include <QPushButton>
#include <QRectF>
#include <QWidget>

class QPaintEvent;
class QPainter;
class QEnterEvent;
class QEvent;

// A container that paints a crop of an ukiyo-e image behind its children (Figma: page backgrounds and banners).
// The crop is a normalised rectangle of the image chosen for the design size; other sizes keep its centre and cover.
class ArtBackground: public QWidget {
	Q_OBJECT

public:
	explicit ArtBackground(QWidget* parent = nullptr);

	void SetArt(const QString& resource, const QRectF& crop);
	// Launcher style v08: a soft paper shadow under each child card (QFrame[card="true"]), as the CSS box-shadow
	// 0 10px 30px rgba(ink, 0.12) of the mockup. Takes the cards that are children at the time of the call.
	void SetCardShadows();
	// Paper fade from the left edge: alpha at the edge, gone at stop (0..1 of the width).
	void SetScrim(qreal alpha, qreal stop);
	// Fade into the page colour over the bottom height pixels.
	void SetBottomFade(int height);
	// Paper wash from the top edge: alpha at the top, gone after height pixels (Figma First start "Header wash").
	void SetTopWash(qreal alpha, int height);
	// A picture shown instead of the art while it is set (Play: the selected game's picture from its sce_sys
	// folder), whole picture as cover, under an ink layer of the given alpha so paper text stays readable. The art's
	// scrim and washes are for paper text and are not drawn over it. A null pixmap brings the art back.
	void SetPicture(const QPixmap& picture, qreal ink_alpha);
	[[nodiscard]] bool HasPicture() const { return !m_picture.isNull(); }

protected:
	void paintEvent(QPaintEvent* event) override;
	bool eventFilter(QObject* watched, QEvent* event) override;

private:
	void PaintCardShadows(QPainter& painter) const;

	QList<QPointer<QWidget>> m_cards;
	QPixmap m_picture;
	qreal   m_picture_ink = 0.0;
	QPixmap m_art;
	QRectF  m_crop {0.0, 0.0, 1.0, 1.0};
	qreal   m_scrim_alpha = 0.0;
	qreal   m_scrim_stop  = 0.0;
	int     m_bottom_fade = 0;
	qreal   m_top_wash_alpha  = 0.0;
	int     m_top_wash_height = 0;
};

// Figma Button kind=image / Button/Play style=image: a crop of the artwork under an ink tint, a thin gold frame and a
// paper label. Hover lightens the tint and turns the frame gold; pressed darkens it; disabled is 40 % opacity.
// The "brush" property switches to the launcher style v08 (no art, no frame) for buttons on the ink bar:
// "swipe" is a gold label whose broad warm-ink swipe the parent BrushBar paints under it (it is longer than the
// button); "underline" is a paper label over a thin gold brush stroke. Hover and pressed change the label colour.
class ArtButton: public QPushButton {
	Q_OBJECT
	Q_PROPERTY(QString brush READ Brush WRITE SetBrush)

public:
	explicit ArtButton(QWidget* parent = nullptr);

	void SetArt(const QString& resource, const QRectF& crop);

	[[nodiscard]] QString Brush() const { return m_brush; }
	void                  SetBrush(const QString& brush);

	[[nodiscard]] QSize sizeHint() const override;

protected:
	void paintEvent(QPaintEvent* event) override;
	void changeEvent(QEvent* event) override;
	void enterEvent(QEnterEvent* event) override;
	void leaveEvent(QEvent* event) override;

private:
	void PaintBrushLook(QPainter& painter, bool hover, bool pressed);

	QPixmap m_art;
	QRectF  m_crop {0.0, 0.0, 1.0, 1.0};
	QString m_brush;
};

// Launcher style v08: the Play bottom bar (QSS ink background) that paints the long warm-ink swipe under its
// ArtButton children with brush "swipe", reaching past both ends of the button as in the mockup.
class BrushBar: public QFrame {
	Q_OBJECT

public:
	explicit BrushBar(QWidget* parent = nullptr);

protected:
	void paintEvent(QPaintEvent* event) override;
};

// Launcher style v08: a heading underlined by a gold brush stroke painted under its text. Heading/Title gets the broad
// stroke that runs past the end of the text, Heading/Section a short thin one; the label keeps room below its text
// for the stroke. Each label's stroke shape comes from its object name, so it stays the same on every repaint.
class BrushHeading: public QLabel {
	Q_OBJECT

public:
	explicit BrushHeading(QWidget* parent = nullptr);

protected:
	bool event(QEvent* event) override;
	void paintEvent(QPaintEvent* event) override;

private:
	void UpdateRoom();
};

// Figma NavItem: dot, label and, on the active item, a vertical gold brush stroke (launcher style v08) at the left
// edge; "accent" makes the active dot gold.
class NavButton: public QPushButton {
	Q_OBJECT
	Q_PROPERTY(bool accent READ IsAccent WRITE SetAccent)

public:
	explicit NavButton(QWidget* parent = nullptr);

	[[nodiscard]] bool IsAccent() const { return m_accent; }
	void               SetAccent(bool accent);

protected:
	void paintEvent(QPaintEvent* event) override;
	void enterEvent(QEnterEvent* event) override;
	void leaveEvent(QEvent* event) override;

private:
	bool m_accent = false;
};

// Figma StatusIcon: a 20 px circle in the status colour (ok: success, warn: ochre, error: error) with a 12 px
// paper-light glyph.
class StatusIcon: public QWidget {
	Q_OBJECT

public:
	enum class Status { Ok, Warn, Error };

	explicit StatusIcon(QWidget* parent = nullptr);

	void SetStatus(Status status);

protected:
	void paintEvent(QPaintEvent* event) override;

private:
	Status m_status = Status::Ok;
};

#endif // LAUNCHER_WIDGETS_H
