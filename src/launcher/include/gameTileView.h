#ifndef GAME_TILE_VIEW_H
#define GAME_TILE_VIEW_H

#include <QListView>
#include <QString>

class QPaintEvent;
class QResizeEvent;

// Play: the games as a grid of tiles (square cover from the game's sce_sys/icon0.png, name, "serial · version").
// It shows the model of the game list (ConfigurationListWidget's tree, kept hidden) and shares its selection, so
// selecting, sorting and the search filter stay in one place. The selected game has a paper backing (no frame in the
// launcher style v08), the tile under the mouse a soft highlight. Dark: drawn for the selected game's darkened background (docs/launcher/DESIGN-SPEC.md).
class GameTileView: public QListView {
	Q_OBJECT

public:
	// Data roles of the name column, set by ConfigurationItem.
	static constexpr int CoverRole   = Qt::UserRole + 1; // QPixmap, null when the game has no icon0.png
	static constexpr int RunningRole = Qt::UserRole + 2; // bool

	explicit GameTileView(QWidget* parent = nullptr);

	void               SetDark(bool dark);
	[[nodiscard]] bool IsDark() const { return m_dark; }
	// Shown in the middle while no tile is shown.
	void SetEmptyText(const QString& title, const QString& description);

	// Tall enough for every shown tile at the current width: the Games card ends under the last row and the page's
	// picture shows below it; with more games than fit, the card fills the page and the tiles scroll. Call
	// updateGeometry() when tiles are shown or hidden.
	[[nodiscard]] QSize sizeHint() const override;
	[[nodiscard]] QSize minimumSizeHint() const override;

protected:
	void paintEvent(QPaintEvent* event) override;
	void resizeEvent(QResizeEvent* event) override;

private:
	[[nodiscard]] int ShownCount() const;

	bool    m_dark = false;
	QString m_empty_title;
	QString m_empty_description;
};

#endif // GAME_TILE_VIEW_H
