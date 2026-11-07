#include "mainDialog.h"

#include "bugReport.h"
#include "cheatFile.h"
#include "configuration.h"
#include "configurationItem.h"
#include "configurationListWidget.h"
#include "controllerLightbar.h"
#include "forkInfo.h"
#include "gameContent.h"
#include "gameTileView.h"
#include "hostInfo.h"
#include "launcherTheme.h"
#include "launcherWidgets.h"
#include "updateChecker.h"

#include <QApplication>
#include <QButtonGroup>
#include <QBoxLayout>
#include <QFrame>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <QSet>
#include <utility>
#include <QByteArray>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialog>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QHoverEvent>
#include <QIODevice>
#include <QImage>
#include <QLabel>
#include <QLocale>
#include <QMap>
#include <QMessageBox>
#include <QObject>
#include <QPixmap>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QRegularExpression>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStringList>
#include <QStyle>
#include <QTreeWidget>
#include <QTextStream>
#include <QUrl>
#include <QVariant>
#include <QtConcurrentRun>
#include <QtCore>

#include <algorithm>
#include <cstdint>
#include <optional>

#include "common/systemInfo.h"
#include "ui_main_dialog.h"

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#endif

// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <processthreadsapi.h>
// IWYU pragma: no_include <winbase.h>

class QWidget;

#if defined(_WIN32)
constexpr char EMULATOR_EXE[] = "kyty_emulator.exe";
#else
constexpr char EMULATOR_EXE[] = "kyty_emulator";
#endif

#if defined(_WIN32)
constexpr char CMD_EXE[] = "cmd.exe";
#elif defined(__linux__)
constexpr char KYTY_BASH_FILE[] = "kyty_run.sh";
#endif
#if defined(_WIN32)
constexpr DWORD CMD_X_CHARS = 175;
constexpr DWORD CMD_Y_CHARS = 1000;
#endif
constexpr char SETTINGS_MAIN_DIALOG[]        = "MainDialog";
constexpr char SETTINGS_MAIN_LAST_GEOMETRY[] = "geometry";
constexpr char SETTINGS_CHECK_UPDATES[]       = "check_updates_on_startup";
constexpr char SETTINGS_PRECOMPILE_THREADS[]  = "precompile_threads";
constexpr char SETTINGS_PRECOMPILE_ON_START[] = "precompile_on_start";
// "<compiled> <all>" per title id, from the last precompile's PRECOMPILE_* lines (Play checks: % precompiled).
constexpr char SETTINGS_PRECOMPILE_PROGRESS[] = "PrecompileProgress";
// One log per game run (the emulator's stderr, after a header with the command line), next to the emulator.
constexpr char LOG_DIR[]       = "_Logs";
constexpr char LOG_PREFIX[]    = "kyty-";
constexpr int  LOG_KEEP        = 10;
constexpr int  BUG_REPORT_LOGS = 3; // newest run logs in a bug report
// The emulator's stderr of a shader precompile (kyty_emulator --precompile-only), next to the emulator.
constexpr char PRECOMPILE_LOG_FILE[] = "_precompile_log.txt";
// The emulator's pipeline cache folder (relative to its working directory, the emulator's folder).
constexpr char PIPELINE_CACHE_DIR[] = "_PipelineCache";
// The shader hints folder next to the emulator: the package's hints (<title id>_<version>.kyhints) and the default of the
// hints file dialogs.
constexpr char HINTS_DIR[] = "hints";
// Saves a picture of every page into this folder and quits, without looking for the emulator (design check against
// the Figma frames, docs/launcher/DESIGN-SPEC.md).
constexpr char SCREENSHOT_DIR_ENV[] = "KYTY_LAUNCHER_SCREENSHOT_DIR";

// Normalised crops of the ukiyo-e images, as in the Figma frames (docs/launcher/DESIGN-SPEC.md).
constexpr QRectF ART_PLAY_BACKGROUND {0.0, 0.0, 1.0, 1.0};              // image B whole, as cover (style v08)
constexpr QRectF ART_GINKGO_BRANCH {0.6171, 0.0826, 0.3639, 0.1238};   // image B in the buttons
constexpr QRectF ART_DISPLAY_BANNER {0.0, 0.1786, 1.0, 0.4568};         // 1032x200
constexpr QRectF ART_GRAPHICS_BANNER {0.0, 0.1935, 1.0, 0.4568};        // 1032x200
constexpr QRectF ART_SHADERS_BANNER {0.0, 0.2976, 1.0, 0.4568};         // 1032x200
constexpr QRectF ART_ADVANCED_BANNER {0.0, 0.1190, 1.0, 0.4568};        // 1032x200
constexpr QRectF ART_FIRST_START {0.0671, 0.0, 0.8658, 1.0};            // 1032x800
constexpr int    BANNER_FADE = 56;
constexpr int    PAGE_PLAY        = 0;
constexpr int    PAGE_DISPLAY     = 1;
constexpr int    PAGE_GRAPHICS    = 2;
constexpr int    PAGE_SHADERS     = 3;
constexpr int    PAGE_ADVANCED    = 4;
constexpr int    PAGE_MODS        = 5;
// v0.1.6: player guide pages (KytyPS5-Yotei-Player-Guide.pdf) of the first-start steps.
constexpr int    GUIDE_PAGE_IMPORT_HINTS = 4;
constexpr int    GUIDE_PAGE_PRECOMPILE   = 4;
constexpr int    PAGE_FIRST_START = 6; // not in the sidebar: shown while no game is found
// Play background with a game selected: its picture from sce_sys (pic0.png, else pic1.png), scaled to this width
// (enough for the page on a 1.5x screen), under an ink layer of this alpha.
constexpr int   GAME_ART_WIDTH = 2048;
constexpr qreal GAME_ART_INK   = 0.62;

// A game's picture, decoded and scaled off the UI thread; key is the game's folder (basedir).
struct GameArt {
	QString key;
	QImage  image;
};

class MainDialogPrivate: public QObject {
	Q_OBJECT

public:
	explicit MainDialogPrivate(QObject* parent = nullptr): QObject(parent) {}
	~MainDialogPrivate() override;

	void Setup(MainDialog* main_dialog);

	/*slots:*/

	void Update();
	void FindInterpreter();
	void Run();
	// "Precompile shaders" / "Continue precompiling": starts kyty_emulator --precompile-only for the selected game.
	// `extra`: more emulator arguments (--import-hints <file> or --export-hints <file>, which ends the run after it).
	void Precompile(const QStringList& extra = {});
	// "Import hints..." / "Export hints...": a file dialog, then Precompile with the hints argument.
	void ImportHints();
	void ExportHints();
	// "Pause": asks the running precompile to stop (it finishes the running compiles and saves; the next precompile
	// goes on from there); a second press ends it at once.
	void PausePrecompile();

	[[nodiscard]] const QString& GetInterpreter() const { return m_interpreter; }
	// The game's environment: KYTY_SHADER_PRECOMPILE and its threads (Shaders).
	[[nodiscard]] QProcessEnvironment GameEnvironment(const Configuration& info) const;
	// A new run log in _Logs with a header (launcher, emulator, command line); keeps the newest LOG_KEEP logs.
	[[nodiscard]] QString StartRunLog(const QString& command) const;

	static void WriteSettings(QSettings& s);
	static void ReadSettings(QSettings& s);

private:
	struct Check {
		StatusIcon::Status status = StatusIcon::Status::Ok;
		QString            title;
		QString            description;
	};

	// kyty_emulator next to the launcher (its folder or the one above).
	enum class Emulator {
		NotLooked, // until FindInterpreter
		Found,
		Missing,
		NoAnswer, // found, but it printed no version
	};

	void PrecompileOutput();
	void PrecompileLine(const QString& line);
	void PrecompileFinished(int exit_code, QProcess::ExitStatus exit_status);
	void ShowPage(int page);
	// The Mods page: the selected game's patches from its catalog (:/mods/<title id>.json), each with its measured gain;
	// a change is saved into the game's patch plan (Cheats::PlanPath), which Play passes as --game-patch.
	void BuildModsPage();
	void SaveModsPlan();
	void UpdatePlaySummary(const ConfigurationItem* item);
	void UpdateChecks(const ConfigurationItem* item);
	// Play background: starts loading the selected game's picture (the ukiyo-e art while none is shown).
	void UpdateGameArt(const ConfigurationItem* item);
	void ShowGameArt(const QImage& image);
	// Play over a game's picture: the [dark="true"] rules of launcher.qss (paper text, cards as dark glass).
	void SetPlayDark(bool dark);
	void LoadSettingsPages();
	void ConnectSettingsPages();
	void UpdateCacheInfo();
	void ChooseGameFolder();
	void ClearCache();
	void ExportBugReport();
	[[nodiscard]] QList<BugReport::File> BugReportFiles() const;
	[[nodiscard]] QString PipelineCacheDir() const;
	[[nodiscard]] QString HintsDir() const;
	// The hints file shipped for the game (hints/<title>_*.kyhints, the newest name) when the pipeline cache has no
	// shader journal of it yet: its first start. Empty otherwise.
	[[nodiscard]] QString FirstStartHints(const QString& title, const QString& version) const;
	[[nodiscard]] QString LogDir() const;
	[[nodiscard]] Check   CacheCheck(const ConfigurationItem* item) const;
	// What is wrong with the emulator; nothing while it is found (or not looked for yet).
	[[nodiscard]] std::optional<Check> EmulatorCheck() const;
	// The selected game's version against the one the fork is tested on (forkInfo.h); nothing without a game.
	[[nodiscard]] static std::optional<Check> VersionCheck(const ConfigurationItem* item);
	void SaveScreenshots(const QString& dir);

	static void SetCheck(StatusIcon* icon, QLabel* title, QLabel* description, const Check& check);

	static QByteArray                           g_last_geometry;
	static bool                                 g_check_updates_on_startup;
	static int                                  g_precompile_threads; // 0: the emulator's setting
	static bool                                 g_precompile_on_start;
	static QMap<QString, QPair<qint64, qint64>> g_precompile_progress; // title id -> compiled, all
	// v0.1.6 first-start steps for a game version whose hints the package ships (hints\<title>_<version>.kyhints):
	// 1 = the hints were imported, 2 = a precompile then finished. Play waits for 2, Precompile for 1.
	static QMap<QString, int> g_shader_setup; // "<title id>_<version>" -> step
	[[nodiscard]] QString ShippedHintsKey(const ConfigurationItem* item) const; // empty: no hints shipped for it
	[[nodiscard]] int     ShaderSetupStep(const ConfigurationItem* item) const;   // 2 when no hints are shipped
	// v0.1.6: the player guide (KytyPS5-Yotei-Player-Guide.pdf next to the launcher) opened at a step's page.
	void         OpenGuide(int page);
	QPushButton* m_guide_play_button = nullptr; // "How to do this?" on Play while a first-start step is missing

	Ui::MainDialog* m_ui             = {nullptr};
	MainDialog*     m_main_dialog    = nullptr;
	UpdateChecker*  m_update_checker = nullptr;
	QButtonGroup*   m_nav            = nullptr;
	ControllerLightbar m_lightbar;
	QString         m_interpreter; // empty without a working emulator
	QString         m_emulator_path; // where it was looked for last (Missing: next to the launcher)
	Emulator        m_emulator = Emulator::NotLooked;
	QString         m_emulator_version; // the emulator's build line (its output without arguments)

	QFutureWatcher<GameArt> m_game_art;
	QString                 m_game_art_key; // basedir of the game whose picture is shown or loading

	QProcess m_process;
	// kyty_emulator --precompile-only: its stdout PRECOMPILE_* lines drive the progress bar.
	QProcess               m_precompile;
	bool                   m_precompile_stop_sent = false;
	QString                m_precompile_title; // title id of the game being precompiled
	int                    m_precompile_cached    = 0;
	int                    m_precompile_old       = 0; // journal files an older emulator wrote (not read)
	QMap<QString, QString> m_precompile_done; // the PRECOMPILE_DONE line's fields
	QMap<QString, QString> m_precompile_hints; // the PRECOMPILE_HINTS line's fields (kind, status, counts)
	// The Mods page: its rows (rebuilt when the page is shown), the catalog, the toggles by mod name, the ladder's choice,
	// and the plan file's bytes as last read or written (SaveDocument refuses a file changed meanwhile).
	QWidget*                  m_mods_rows = nullptr;
	QJsonObject               m_mods_catalog;
	QMap<QString, QCheckBox*> m_mod_toggles;
	QComboBox*                m_mods_ladder = nullptr;
	QLabel*                   m_mods_status = nullptr;
	QString                   m_mods_title;
	QByteArray                m_mods_plan_data;
	QString                m_hints_file;       // the hints file of the running import or export
	bool                   m_play_after_precompile = false; // Play on a first start: the game starts once the shaders are ready
	QSet<QString>          m_first_start_hints; // titles whose shipped hints this session imported (or tried to)

	QPointer<ConfigurationItem> m_running_item;
};

QByteArray MainDialogPrivate::g_last_geometry;
bool       MainDialogPrivate::g_check_updates_on_startup = true;
int        MainDialogPrivate::g_precompile_threads       = 0;
bool       MainDialogPrivate::g_precompile_on_start      = true;
QMap<QString, QPair<qint64, qint64>> MainDialogPrivate::g_precompile_progress;
QMap<QString, int>                   MainDialogPrivate::g_shader_setup;

MainDialog::MainDialog(QWidget* parent): QDialog(parent), m_p(new MainDialogPrivate(this)) {
	m_p->Setup(this);
}

MainDialogPrivate::~MainDialogPrivate() {
	m_process.disconnect(this);
	m_precompile.disconnect(this);
	if (m_precompile.state() != QProcess::NotRunning) {
		// A killed precompile marks the shaders it was compiling as suspects (skipped for this emulator build): ask
		// it to stop and save first (its running compiles take up to about a minute).
		m_precompile.write("quit\n");
		if (!m_precompile.waitForFinished(180000)) {
			m_precompile.kill();
			m_precompile.waitForFinished(5000);
		}
	}
	delete m_ui;
}

void MainDialogPrivate::Setup(MainDialog* main_dialog) {
	m_ui = new Ui::MainDialog;
	m_ui->setupUi(main_dialog);

	m_main_dialog = main_dialog;
	m_update_checker = new UpdateChecker(main_dialog);
	m_ui->check_updates_on_startup->setChecked(g_check_updates_on_startup);
	m_ui->updates_section->setVisible(UpdateChecker::IsSupported());

	LauncherTheme::ApplyTextStyles(main_dialog);
	const auto play_art = QStringLiteral(":/images/ukiyoe-play.jpg");
	// Launcher style v08 (design mockup): Play shows its picture without the paper scrim, the cards
	// are paper sheets with soft shadows, the Play bar's buttons are brush-painted (forms/main_dialog.ui, "brush").
	m_ui->play_art->SetArt(play_art, ART_PLAY_BACKGROUND);
	m_ui->play_art->SetCardShadows();
	m_ui->first_folder_button->SetArt(play_art, ART_GINKGO_BRANCH);
	m_ui->display_banner->SetArt(QStringLiteral(":/images/ukiyoe-display.jpg"), ART_DISPLAY_BANNER);
	m_ui->display_banner->SetBottomFade(BANNER_FADE);
	m_ui->graphics_banner->SetArt(QStringLiteral(":/images/ukiyoe-graphics.jpg"), ART_GRAPHICS_BANNER);
	m_ui->graphics_banner->SetBottomFade(BANNER_FADE);
	m_ui->shaders_banner->SetArt(QStringLiteral(":/images/ukiyoe-shaders.jpg"), ART_SHADERS_BANNER);
	m_ui->shaders_banner->SetBottomFade(BANNER_FADE);
	m_ui->advanced_banner->SetArt(QStringLiteral(":/images/ukiyoe-advanced.jpg"), ART_ADVANCED_BANNER);
	m_ui->advanced_banner->SetBottomFade(BANNER_FADE);
	m_ui->mods_banner->SetArt(QStringLiteral(":/images/ukiyoe-graphics.jpg"), ART_GRAPHICS_BANNER);
	m_ui->mods_banner->SetBottomFade(BANNER_FADE);
	m_ui->first_start_art->SetArt(QStringLiteral(":/images/ukiyoe-first-start.jpg"), ART_FIRST_START);
	m_ui->first_start_art->SetScrim(0.85, 0.70);
	m_ui->first_start_art->SetTopWash(0.95, 300);
	m_ui->first_start_art->SetCardShadows();

	SetCheck(m_ui->first_folder_check_status, m_ui->first_folder_check, m_ui->first_folder_check_description,
	         {StatusIcon::Status::Error, tr("No game found"), tr("Pick the folder that contains eboot.bin and sce_sys.")});
	m_ui->footer_version->setText(QString::fromLatin1(FORK_VERSION));
	m_ui->label_Version->setText(QStringLiteral("KytyPS5 · Yōtei fork · ") + QString::fromLatin1(FORK_VERSION));
	m_ui->row_version_description->setText(tr("by Erlou"));
	m_ui->source_code_button->setVisible(FORK_REPOSITORY_URL[0] != '\0');
	connect(m_ui->source_code_button, &QPushButton::clicked, this,
	        []() { QDesktopServices::openUrl(QUrl(QString::fromLatin1(FORK_REPOSITORY_URL))); });

	m_nav = new QButtonGroup(this);
	m_nav->addButton(m_ui->nav_play, PAGE_PLAY);
	m_nav->addButton(m_ui->nav_display, PAGE_DISPLAY);
	m_nav->addButton(m_ui->nav_graphics, PAGE_GRAPHICS);
	m_nav->addButton(m_ui->nav_shaders, PAGE_SHADERS);
	m_nav->addButton(m_ui->nav_advanced, PAGE_ADVANCED);
	m_nav->addButton(m_ui->nav_mods, PAGE_MODS);
	connect(m_nav, &QButtonGroup::idClicked, this, &MainDialogPrivate::ShowPage);
	connect(m_ui->play_button, &QPushButton::clicked, this, &MainDialogPrivate::Run);
	connect(m_ui->play_precompile_button, &QPushButton::clicked, this, [this]() {
		ShowPage(PAGE_SHADERS);
		if (m_precompile.state() == QProcess::NotRunning) {
			Precompile();
		}
	});
	connect(m_ui->cache_open_button, &QPushButton::clicked, this, [this]() {
		QDesktopServices::openUrl(QUrl::fromLocalFile(PipelineCacheDir()));
	});
	connect(m_ui->cache_clear_button, &QPushButton::clicked, this, &MainDialogPrivate::ClearCache);
	connect(m_ui->first_folder_button, &QPushButton::clicked, this, &MainDialogPrivate::ChooseGameFolder);
	connect(m_ui->game_folder_button, &QPushButton::clicked, this, &MainDialogPrivate::ChooseGameFolder);
	connect(m_ui->log_folder_button, &QPushButton::clicked, this, [this]() {
		QDir().mkpath(LogDir());
		QDesktopServices::openUrl(QUrl::fromLocalFile(LogDir()));
	});
	connect(m_ui->bug_report_button, &QPushButton::clicked, this, &MainDialogPrivate::ExportBugReport);
	{
		// v0.1.6: buttons that open the player guide at the page of the step (players skipped the guide).
		const auto find_box = [](auto&& self, QLayout* layout, QWidget* widget) -> QBoxLayout* {
			if (layout == nullptr) return nullptr;
			for (int i = 0; i < layout->count(); ++i) {
				auto* entry = layout->itemAt(i);
				if (entry->widget() == widget) return qobject_cast<QBoxLayout*>(layout);
				if (auto* found = self(self, entry->layout(), widget); found != nullptr) return found;
			}
			return nullptr;
		};
		const auto add_after = [&](QWidget* anchor, const QString& text, int page) -> QPushButton* {
			auto* button = new QPushButton(text, anchor->parentWidget());
			button->setProperty("kind", QStringLiteral("secondary"));
			button->setProperty("textStyle", QStringLiteral("Label/Button"));
			button->setCursor(Qt::PointingHandCursor);
			connect(button, &QPushButton::clicked, this, [this, page]() { OpenGuide(page); });
			if (auto* box = find_box(find_box, anchor->parentWidget()->layout(), anchor); box != nullptr) {
				box->insertWidget(box->indexOf(anchor) + 1, button, 0, Qt::AlignLeft);
			}
			return button;
		};
		m_guide_play_button = add_after(m_ui->play_title, tr("How to do this?"), GUIDE_PAGE_IMPORT_HINTS);
		m_guide_play_button->setVisible(false);
		add_after(m_ui->hints_import_button, tr("How to import hints?"), GUIDE_PAGE_IMPORT_HINTS);
		add_after(m_ui->bug_report_button, tr("Player guide"), 1);
	}
	ConnectSettingsPages();

	main_dialog->setWindowFlags(Qt::Dialog /*| Qt::MSWindowsFixedSizeDialogHint*/);

	connect(main_dialog, &MainDialog::Start, this, &MainDialogPrivate::FindInterpreter,
	        Qt::QueuedConnection);
	connect(m_ui->widget, &ConfigurationListWidget::Select, this, &MainDialogPrivate::Update);
	connect(m_ui->widget, &ConfigurationListWidget::PreviewControllerColor, this,
	        [this](const QString& color) {
		        if (m_process.state() == QProcess::NotRunning) {
			        m_lightbar.SetColor(color);
		        }
	        });
	connect(m_ui->widget, &ConfigurationListWidget::Run, this, &MainDialogPrivate::Run);
	connect(m_ui->widget, &ConfigurationListWidget::AddGameFolder, this, &MainDialogPrivate::ChooseGameFolder);
	connect(&m_game_art, &QFutureWatcher<GameArt>::finished, this, [this]() {
		// Only the picture of the game still selected (a later selection may have no picture).
		if (const auto art = m_game_art.result(); art.key == m_game_art_key) {
			ShowGameArt(art.image);
		}
	});
	connect(m_ui->check_updates_link, &QLabel::linkActivated, this,
	        [this](const QString&) { m_update_checker->Check(true); });
	connect(m_update_checker, &UpdateChecker::CheckingChanged, m_ui->check_updates_link,
	        &QLabel::setDisabled);
	connect(m_ui->check_updates_on_startup, &QCheckBox::toggled, this, [this](bool checked) {
		g_check_updates_on_startup = checked;
		m_ui->widget->WriteSettings();
	});
	connect(main_dialog, &MainDialog::Resize, [this]() {
		g_last_geometry = m_main_dialog->saveGeometry();
		m_ui->widget->WriteSettings();
	});

	connect(&m_process, &QProcess::stateChanged, this, [this](QProcess::ProcessState state) {
		if (state == QProcess::NotRunning) {
			if (m_running_item != nullptr) {
				m_running_item->SetRunning(false);
			}
			Update();
		}
	});

	m_ui->precompile_progress->setVisible(false);
	m_ui->precompile_count->setVisible(false);
	m_ui->precompile_eta->setVisible(false);
	m_ui->precompile_status->setVisible(false);
	m_ui->precompile_pause_button->setVisible(false);
	m_ui->precompile_threads->setValue(g_precompile_threads);
	m_ui->precompile_on_start->setChecked(g_precompile_on_start);
	connect(m_ui->precompile_button, &QPushButton::clicked, this, [this]() { Precompile(); });
	connect(m_ui->precompile_pause_button, &QPushButton::clicked, this, &MainDialogPrivate::PausePrecompile);
	connect(m_ui->hints_import_button, &QPushButton::clicked, this, &MainDialogPrivate::ImportHints);
	connect(m_ui->hints_export_button, &QPushButton::clicked, this, &MainDialogPrivate::ExportHints);
	connect(m_ui->precompile_on_start, &QCheckBox::toggled, this, [this](bool checked) {
		g_precompile_on_start = checked;
		m_ui->widget->WriteSettings();
		Update();
	});
	connect(m_ui->precompile_threads, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
		g_precompile_threads = value;
		m_ui->widget->WriteSettings();
		Update();
	});
	connect(&m_precompile, &QProcess::readyReadStandardOutput, this, &MainDialogPrivate::PrecompileOutput);
	connect(&m_precompile, &QProcess::finished, this, &MainDialogPrivate::PrecompileFinished);
	connect(&m_precompile, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
		if (error == QProcess::FailedToStart) {
			m_ui->precompile_progress->setVisible(false);
			m_ui->precompile_count->setVisible(false);
			m_ui->precompile_eta->setVisible(false);
			m_ui->precompile_status->setText(tr("Could not start the emulator: %1").arg(m_precompile.errorString()));
			Update();
		}
	});
	connect(&m_precompile, &QProcess::stateChanged, this, [this](QProcess::ProcessState) { Update(); });

	m_ui->label_settings_file->setText(m_ui->widget->GetSettingsFile());

	m_main_dialog->restoreGeometry(g_last_geometry);

	Update();
}

void MainDialogPrivate::ShowPage(int page) {
	// Play without any game is the First start page (no sidebar item active, as in the Figma frame).
	if (page == PAGE_PLAY && m_ui->widget->GameCount() == 0) {
		page = PAGE_FIRST_START;
	}
	if (auto* button = m_nav->button(page); button != nullptr) {
		button->setChecked(true);
	} else if (auto* checked = m_nav->checkedButton(); checked != nullptr) {
		m_nav->setExclusive(false);
		checked->setChecked(false);
		m_nav->setExclusive(true);
	}
	m_ui->pages->setCurrentIndex(page);
	if (page == PAGE_SHADERS) {
		UpdateCacheInfo();
	} else if (page == PAGE_DISPLAY || page == PAGE_GRAPHICS) {
		LoadSettingsPages();
	} else if (page == PAGE_MODS) {
		BuildModsPage();
	}
}

// One row of the Mods page in the settings pages' layout (label column 280 px, control, description, divider).
static QWidget* ModsRow(const QString& label, QWidget* control, const QString& description) {
	auto* row   = new QWidget;
	auto* outer = new QVBoxLayout(row);
	outer->setContentsMargins(0, 12, 0, 0);
	outer->setSpacing(12);
	auto* line = new QHBoxLayout;
	line->setSpacing(24);
	auto* name = new QLabel(label);
	name->setMinimumWidth(280);
	name->setMaximumWidth(280);
	name->setWordWrap(true);
	name->setProperty("textStyle", QStringLiteral("Body/Strong"));
	line->addWidget(name, 0, Qt::AlignTop);
	auto* column = new QVBoxLayout;
	column->setSpacing(8);
	column->addWidget(control, 0, Qt::AlignLeft);
	auto* text = new QLabel(description);
	text->setWordWrap(true);
	text->setProperty("textStyle", QStringLiteral("Body/Description"));
	text->setProperty("tone", QStringLiteral("secondary"));
	column->addWidget(text);
	line->addLayout(column, 1);
	outer->addLayout(line);
	auto* divider = new QFrame;
	divider->setProperty("divider", true);
	outer->addWidget(divider);
	return row;
}

// The game's mods catalog: the one of its version (:/mods/<title id>_<version>.json) when there is one, else the title's
// (:/mods/<title id>.json, whose "version" says which game version it fits). Empty when the game has none.
static QJsonObject ModsCatalog(const QString& title_id, const QString& game_version) {
	const auto title = title_id.trimmed().toUpper();
	if (title.isEmpty()) return {};
	QFile file(QStringLiteral(":/mods/%1_%2.json").arg(title, game_version));
	if (game_version.isEmpty() || !file.open(QIODevice::ReadOnly)) {
		file.setFileName(QStringLiteral(":/mods/%1.json").arg(title));
		if (!file.open(QIODevice::ReadOnly)) return {};
	}
	return QJsonDocument::fromJson(file.readAll()).object();
}

// The mods that are on until the player turns them off ("default": true in the catalog) go into a new patch plan when
// the game has none yet and its version is the catalog's (3.11.2026: "Render at 1080p" - cutscenes at up to 4K were the
// slowest part of the game, and not every player reads the Mods page).
static void WriteDefaultMods(const QString& title_id, const QString& game_version) {
	const auto path = Cheats::PlanPath(title_id);
	if (path.isEmpty() || QFileInfo::exists(path)) return;
	const auto catalog = ModsCatalog(title_id, game_version);
	if (game_version.isEmpty() || catalog.value(QStringLiteral("version")).toString() != game_version) return;
	QJsonArray mods;
	for (const auto& value: catalog.value(QStringLiteral("mods")).toArray()) {
		const auto mod = value.toObject();
		if (!mod.value(QStringLiteral("default")).toBool(false)) continue;
		QJsonObject entry {{QStringLiteral("name"), mod.value(QStringLiteral("name"))}, {QStringLiteral("enabled"), true}};
		if (mod.contains(QStringLiteral("memory"))) entry.insert(QStringLiteral("memory"), mod.value(QStringLiteral("memory")));
		if (mod.contains(QStringLiteral("env"))) entry.insert(QStringLiteral("env"), mod.value(QStringLiteral("env")));
		mods.append(entry);
	}
	if (mods.isEmpty()) return;
	(void)Cheats::SaveDocument(path,
	                           QJsonDocument(QJsonObject {{QStringLiteral("id"), title_id.trimmed().toUpper()},
	                                                      {QStringLiteral("version"), game_version},
	                                                      {QStringLiteral("process"), QStringLiteral("eboot.bin")},
	                                                      {QStringLiteral("mods"), mods}}),
	                           QByteArray());
}

void MainDialogPrivate::BuildModsPage() {
	delete m_mods_rows;
	m_mod_toggles.clear();
	m_mods_ladder = nullptr;
	m_mods_rows   = new QWidget;
	auto* layout  = new QVBoxLayout(m_mods_rows);
	layout->setContentsMargins(0, 8, 0, 0);
	layout->setSpacing(0);
	m_ui->modsRows->addWidget(m_mods_rows);
	m_mods_status = new QLabel;
	m_mods_status->setWordWrap(true);
	m_mods_status->setContentsMargins(0, 16, 0, 0);
	m_mods_status->setProperty("textStyle", QStringLiteral("Body/Description"));
	m_mods_status->setProperty("tone", QStringLiteral("secondary"));
	const auto finish = [this]() { LauncherTheme::ApplyTextStyles(m_mods_rows); };

	const auto* item = m_ui->widget->GetSelectedItem();
	m_mods_title     = item != nullptr ? item->GetInfo().title_id.trimmed().toUpper() : QString();
	m_mods_catalog = item != nullptr ? ModsCatalog(m_mods_title, item->GetInfo().gameVersion) : QJsonObject();
	if (m_mods_catalog.isEmpty()) {
		m_mods_status->setText(item == nullptr ? tr("Select a game on the Play page to see its mods.")
		                                       : tr("No mods for this game yet."));
		layout->addWidget(m_mods_status);
		finish();
		return;
	}
	const auto version      = m_mods_catalog.value(QStringLiteral("version")).toString();
	const auto game_version = item->GetInfo().gameVersion;
	const bool fits         = game_version.isEmpty() || game_version == version;
	WriteDefaultMods(m_mods_title, game_version);

	// What the plan holds now (a mod is on when an enabled entry has its name).
	m_mods_plan_data.clear();
	QSet<QString> enabled;
	if (QFile plan(Cheats::PlanPath(m_mods_title)); plan.open(QIODevice::ReadOnly)) {
		m_mods_plan_data = plan.readAll();
		for (const auto& value:
		     QJsonDocument::fromJson(m_mods_plan_data).object().value(QStringLiteral("mods")).toArray()) {
			const auto mod = value.toObject();
			if (mod.value(QStringLiteral("enabled")).toBool(true)) enabled.insert(mod.value(QStringLiteral("name")).toString());
		}
	}

	const auto mods = m_mods_catalog.value(QStringLiteral("mods")).toArray();
	// The quality ladder: one choice of its levels (they all write the same place).
	QList<QJsonObject> ladder;
	for (const auto& value: mods) {
		if (value.toObject().value(QStringLiteral("group")).toString() == QLatin1String("ladder")) ladder << value.toObject();
	}
	if (!ladder.isEmpty()) {
		m_mods_ladder = new QComboBox;
		m_mods_ladder->setProperty("select", true);
		m_mods_ladder->setProperty("textStyle", QStringLiteral("Body/Default"));
		m_mods_ladder->setMinimumSize(400, 40);
		m_mods_ladder->setMaximumSize(400, 40);
		m_mods_ladder->addItem(tr("Off (the game's full quality)"), QString());
		for (const auto& mod: ladder) {
			const auto name = mod.value(QStringLiteral("name")).toString();
			m_mods_ladder->addItem(QStringLiteral("%1 \u00b7 %2").arg(mod.value(QStringLiteral("title")).toString(),
			                                                         mod.value(QStringLiteral("gain")).toString()),
			                       name);
			if (enabled.contains(name)) m_mods_ladder->setCurrentIndex(m_mods_ladder->count() - 1);
		}
		m_mods_ladder->setEnabled(fits);
		connect(m_mods_ladder, &QComboBox::activated, this, [this](int) { SaveModsPlan(); });
		layout->addWidget(ModsRow(tr("Quality ladder"), m_mods_ladder,
		                          m_mods_catalog.value(QStringLiteral("ladder_description")).toString()));
	}
	for (const auto& value: mods) {
		const auto mod = value.toObject();
		if (mod.value(QStringLiteral("group")).toString() == QLatin1String("ladder")) continue;
		const auto name   = mod.value(QStringLiteral("name")).toString();
		auto*      toggle = new QCheckBox;
		toggle->setProperty("toggle", true);
		toggle->setChecked(enabled.contains(name));
		toggle->setEnabled(fits);
		connect(toggle, &QCheckBox::toggled, this, [this](bool) { SaveModsPlan(); });
		m_mod_toggles.insert(name, toggle);
		layout->addWidget(ModsRow(mod.value(QStringLiteral("title")).toString(), toggle,
		                          QStringLiteral("%1 \u00b7 %2").arg(mod.value(QStringLiteral("gain")).toString(),
		                                                             mod.value(QStringLiteral("description")).toString())));
	}
	m_mods_status->setText(fits ? tr("Gains measured on our test route (the village, RTX 5070 Ti); yours depend on the place "
	                                 "and your PC. Changes apply on the next Play.")
	                            : tr("These mods are for game version %1; the selected game is %2.").arg(version, game_version));
	layout->addWidget(m_mods_status);
	finish();
}

void MainDialogPrivate::SaveModsPlan() {
	if (m_mods_title.isEmpty() || m_mods_catalog.isEmpty()) return;
	const auto    path = Cheats::PlanPath(m_mods_title);
	QSet<QString> catalog_names;
	QJsonArray    mods;
	for (const auto& value: m_mods_catalog.value(QStringLiteral("mods")).toArray()) {
		const auto mod  = value.toObject();
		const auto name = mod.value(QStringLiteral("name")).toString();
		catalog_names.insert(name);
		const bool on = m_mod_toggles.contains(name)
		                    ? m_mod_toggles.value(name)->isChecked()
		                    : m_mods_ladder != nullptr && m_mods_ladder->currentData().toString() == name;
		if (on) {
			QJsonObject entry {{QStringLiteral("name"), name}, {QStringLiteral("enabled"), true}};
			if (mod.contains(QStringLiteral("memory"))) entry.insert(QStringLiteral("memory"), mod.value(QStringLiteral("memory")));
			if (mod.contains(QStringLiteral("env"))) entry.insert(QStringLiteral("env"), mod.value(QStringLiteral("env")));
			mods.append(entry);
		}
	}
	// Entries the catalog does not know (cheats from the Patches dialog) are kept as they are.
	for (const auto& value: QJsonDocument::fromJson(m_mods_plan_data).object().value(QStringLiteral("mods")).toArray()) {
		if (!catalog_names.contains(value.toObject().value(QStringLiteral("name")).toString())) mods.append(value);
	}
	const QJsonDocument document(QJsonObject {{QStringLiteral("id"), m_mods_title},
	                                          {QStringLiteral("version"), m_mods_catalog.value(QStringLiteral("version"))},
	                                          {QStringLiteral("process"), QStringLiteral("eboot.bin")},
	                                          {QStringLiteral("mods"), mods}});
	if (const auto error = Cheats::SaveDocument(path, document, m_mods_plan_data); !error.isEmpty()) {
		m_mods_status->setText(error);
		return;
	}
	m_mods_plan_data = document.toJson();
	m_mods_status->setText(mods.isEmpty() ? tr("No mods on. Changes apply on the next Play.")
	                                      : tr("%n mod(s) on. Changes apply on the next Play.", nullptr, mods.size()));
}

QString MainDialogPrivate::PipelineCacheDir() const {
	if (m_interpreter.isEmpty()) {
		return {};
	}
	return QFileInfo(m_interpreter).absoluteDir().filePath(PIPELINE_CACHE_DIR);
}

QString MainDialogPrivate::HintsDir() const {
	if (m_interpreter.isEmpty()) {
		return QApplication::applicationDirPath() + QLatin1Char('/') + QString::fromLatin1(HINTS_DIR);
	}
	return QFileInfo(m_interpreter).absoluteDir().filePath(HINTS_DIR);
}

void MainDialogPrivate::OpenGuide(int page) {
	const auto pdf =
	    QDir(QApplication::applicationDirPath()).filePath(QStringLiteral("KytyPS5-Yotei-Player-Guide.pdf"));
	if (!QFileInfo::exists(pdf)) {
		QMessageBox::information(m_main_dialog, tr("Player guide"),
		                         tr("KytyPS5-Yotei-Player-Guide.pdf is not next to launcher.exe."));
		return;
	}
	auto url = QUrl::fromLocalFile(pdf);
	if (page > 1) url.setFragment(QStringLiteral("page=%1").arg(page)); // PDF viewers that know it open that page
	QDesktopServices::openUrl(url);
}

QString MainDialogPrivate::ShippedHintsKey(const ConfigurationItem* item) const {
	if (item == nullptr || m_interpreter.isEmpty()) return {};
	const auto& info = item->GetInfo();
	if (info.title_id.isEmpty() || info.gameVersion.isEmpty()) return {};
	const auto key = info.title_id + QLatin1Char('_') + info.gameVersion;
	return QDir(HintsDir()).exists(key + QStringLiteral(".kyhints")) ? key : QString();
}

int MainDialogPrivate::ShaderSetupStep(const ConfigurationItem* item) const {
	const auto key = ShippedHintsKey(item);
	return key.isEmpty() ? 2 : g_shader_setup.value(key, 0);
}

QString MainDialogPrivate::FirstStartHints(const QString& title, const QString& version) const {
	if (title.isEmpty() || version.isEmpty() || m_interpreter.isEmpty() || m_first_start_hints.contains(title)) {
		return {};
	}
	// The hints of this game version only (hints\<title id>_<version>.kyhints): another version's shaders are mostly
	// not in the game, and the scan would read the whole game looking for them.
	const QDir hints(HintsDir());
	const auto file = title + QLatin1Char('_') + version + QStringLiteral(".kyhints");
	if (!hints.exists(file)) {
		return {};
	}
	if (!QDir(PipelineCacheDir()).entryList({title + QStringLiteral("*.journal")}, QDir::Files).isEmpty()) {
		return {};
	}
	return hints.filePath(file);
}

QString MainDialogPrivate::LogDir() const {
	if (m_interpreter.isEmpty()) {
		return {};
	}
	return QFileInfo(m_interpreter).absoluteDir().filePath(LOG_DIR);
}

// The compiled share of the recorded shaders at the last precompile: 0..100 (100 only when all are compiled), -1 unknown.
static int PrecompiledPercent(const QPair<qint64, qint64>& progress) {
	if (progress.second <= 0) {
		return -1;
	}
	return progress.first >= progress.second ? 100
	                                         : static_cast<int>(std::min<qint64>(99, progress.first * 100 / progress.second));
}

struct CacheFiles {
	QStringList files;
	qint64      bytes = 0;
};

static CacheFiles ListCacheFiles(const QString& dir) {
	CacheFiles cache;
	for (QDirIterator it(dir, QDir::Files | QDir::Hidden | QDir::System, QDirIterator::Subdirectories); it.hasNext();) {
		it.next();
		cache.files.append(it.filePath());
		cache.bytes += it.fileInfo().size();
	}
	return cache;
}

static QString ClearCacheQuestion(const QString& dir, const CacheFiles& cache) {
	return QObject::tr("Clear the shader cache?\n\nMoves %1 files (%2) from %3 to the Recycle Bin: the compiled shaders "
	                   "of every game and the list of shaders seen in play. The next start compiles shaders again "
	                   "while you play (short stutters on new effects).")
	    .arg(cache.files.size())
	    .arg(QLocale(QLocale::English).formattedDataSize(cache.bytes, 1, QLocale::DataSizeTraditionalFormat),
	         QDir::toNativeSeparators(dir));
}

// Moves the files to the Recycle Bin (nothing is deleted for good); returns the ones that could not be moved.
static QStringList MoveToTrash(const QStringList& files) {
	QStringList failed;
	for (const auto& file: files) {
		if (!QFile::moveToTrash(file)) {
			failed.append(file);
		}
	}
	return failed;
}

void MainDialogPrivate::UpdateCacheInfo() {
	const auto dir  = PipelineCacheDir();
	const bool busy = m_process.state() != QProcess::NotRunning || m_precompile.state() != QProcess::NotRunning;
	if (dir.isEmpty() || !QDir(dir).exists()) {
		m_ui->cache_size->setText(tr("No cache yet"));
		m_ui->cache_open_button->setEnabled(false);
		m_ui->cache_clear_button->setEnabled(false);
		return;
	}
	const auto cache = ListCacheFiles(dir);
	auto       text  = QLocale(QLocale::English).formattedDataSize(cache.bytes, 1, QLocale::DataSizeTraditionalFormat);
	if (const auto* item = m_ui->widget->GetSelectedItem(); item != nullptr) {
		if (const int percent = PrecompiledPercent(g_precompile_progress.value(item->GetInfo().title_id)); percent >= 0) {
			text += tr(" · %1% precompiled").arg(percent);
		}
	}
	m_ui->cache_size->setText(text);
	m_ui->cache_open_button->setEnabled(true);
	// The game and a precompile keep the cache files open.
	m_ui->cache_clear_button->setEnabled(!busy && !cache.files.isEmpty());
}

void MainDialogPrivate::ClearCache() {
	const auto dir = PipelineCacheDir();
	if (dir.isEmpty() || m_process.state() != QProcess::NotRunning || m_precompile.state() != QProcess::NotRunning) {
		return;
	}
	const auto cache = ListCacheFiles(dir);
	if (cache.files.isEmpty()) {
		UpdateCacheInfo();
		return;
	}
	if (QMessageBox::question(m_main_dialog, tr("Clear cache"), ClearCacheQuestion(dir, cache),
	                          QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes) {
		return;
	}
	const auto failed = MoveToTrash(cache.files);
	g_precompile_progress.clear();
	m_ui->widget->WriteSettings();
	if (!failed.isEmpty()) {
		QMessageBox::warning(m_main_dialog, tr("Clear cache"),
		                     tr("%1 of %2 files could not be moved to the Recycle Bin (in use, or the drive has no "
		                        "Recycle Bin). They are still in %3.")
		                         .arg(failed.size())
		                         .arg(cache.files.size())
		                         .arg(QDir::toNativeSeparators(dir)));
	}
	UpdateCacheInfo();
	Update();
}

void MainDialogPrivate::SetCheck(StatusIcon* icon, QLabel* title, QLabel* description, const Check& check) {
	icon->SetStatus(check.status);
	title->setText(check.title);
	description->setText(check.description);
	description->setVisible(!check.description.isEmpty());
}

MainDialogPrivate::Check MainDialogPrivate::CacheCheck(const ConfigurationItem* item) const {
	using S = StatusIcon::Status;
	if (item == nullptr || item->GetInfo().title_id.isEmpty()) {
		return {S::Warn, tr("Shader cache"), tr("Select a game to see how much of it is precompiled.")};
	}
	const auto& title_id = item->GetInfo().title_id;
	const int   percent  = PrecompiledPercent(g_precompile_progress.value(title_id));
	if (percent == 100) {
		return {S::Ok, tr("Shader cache: precompiled"), tr("As of the last precompile.")};
	}
	if (percent >= 0) {
		return {S::Warn, tr("Shader cache: %1% precompiled").arg(percent),
		        tr("As of the last precompile. Short stutters on new effects until it is done.")};
	}
	const auto dir = PipelineCacheDir();
	if (!dir.isEmpty() && !QDir(dir).entryList({title_id + QStringLiteral(".*")}, QDir::Files).isEmpty()) {
		return {S::Warn, tr("Shader cache: not precompiled yet"),
		        tr("Precompile shaders to avoid short stutters on new effects.")};
	}
	return {S::Warn, tr("Shader cache: empty"),
	        tr("The first start compiles shaders while you play (short stutters on new effects).")};
}

// A path that word wrap may break after its separators (one without spaces would be cut off at the card's edge).
static QString WrappablePath(const QString& path) {
	auto text = QDir::toNativeSeparators(path);
	text.replace(QLatin1Char('\\'), QStringLiteral("\\​"));
	text.replace(QLatin1Char('/'), QStringLiteral("/​"));
	return text;
}

std::optional<MainDialogPrivate::Check> MainDialogPrivate::EmulatorCheck() const {
	using S = StatusIcon::Status;
	if (m_emulator == Emulator::Missing) {
		return Check {S::Error, tr("Emulator not found"),
		              tr("%1 must be in the launcher's folder (%2) or the folder above it. Put the launcher next to "
		                 "the emulator and start it again.")
		                  .arg(QString::fromLatin1(EMULATOR_EXE), WrappablePath(QApplication::applicationDirPath()))};
	}
	if (m_emulator == Emulator::NoAnswer) {
		return Check {S::Error, tr("Emulator not responding"),
		              tr("%1 did not report its version. Replace it with a working build and start the launcher again.")
		                  .arg(WrappablePath(m_emulator_path))};
	}
	return std::nullopt;
}

std::optional<MainDialogPrivate::Check> MainDialogPrivate::VersionCheck(const ConfigurationItem* item) {
	using S = StatusIcon::Status;
	if (item == nullptr) {
		return std::nullopt;
	}
	const auto& info           = item->GetInfo();
	const auto  tested_version = QString::fromLatin1(TESTED_GAME_VERSIONS[0]);
	const auto  version        = info.gameVersion.isEmpty() ? tr("unknown") : info.gameVersion;
	const bool  tested_game    = info.title_id == QLatin1String(TESTED_TITLE_ID);
	const bool  tested_version_found =
	    std::any_of(std::begin(TESTED_GAME_VERSIONS), std::end(TESTED_GAME_VERSIONS),
	                [&](const char* v) { return info.gameVersion == QLatin1String(v); });
	if (tested_game && tested_version_found) {
		return Check {S::Ok, tr("Game version %1 · tested").arg(version), {}};
	}
	if (tested_game) {
		return Check {S::Warn, tr("Game version %1 · untested").arg(version),
		              tr("This fork is tuned for %1; it may run slower or stop with a message.").arg(tested_version)};
	}
	return Check {S::Warn, tr("Game version %1 · untested").arg(version),
	              tr("This fork is tuned for %1 (%2) %3; other games may run slower or stop with a message.")
	                  .arg(QString::fromUtf8(TESTED_GAME_NAME), QString::fromLatin1(TESTED_TITLE_ID), tested_version)};
}

// Play: the Checks card (emulator when it is missing, game folder, game version, graphics card, RAM, shader cache);
// First start: the System card.
void MainDialogPrivate::UpdateChecks(const ConfigurationItem* item) {
	using S = StatusIcon::Status;
	Check folder {S::Warn, tr("No game selected"), tr("Click a game tile to select it.")};
	int   gpu_index = m_ui->widget->EditableConfiguration()->gpu_index;
	if (item != nullptr) {
		const auto& info = item->GetInfo();
		const auto  path = WrappablePath(info.basedir);
		gpu_index        = m_ui->widget->CreateConfiguration(*item)->gpu_index;
		if (GameContent::IsArchive(info.basedir) && QFileInfo::exists(info.basedir)) {
			folder = {S::Ok, tr("Game archive found"), path};
		} else if (QDir(info.basedir).exists()) {
			const bool sce_sys = QDir(info.basedir).exists(QStringLiteral("sce_sys"));
			folder             = {sce_sys ? S::Ok : S::Warn, tr("Game folder found"),
			                      path + (sce_sys ? tr(" · eboot.bin, sce_sys") : tr(" · no sce_sys folder"))};
		} else {
			folder = {S::Error, tr("Game folder missing"), path};
		}
	}

	const auto& gpus = HostInfo::Gpus();
	Check       gpu {S::Error, tr("No Vulkan graphics card found"), tr("Install or update the graphics driver.")};
	if (const int index = HostInfo::ResolveGpuIndex(gpu_index); index >= 0) {
		gpu = {S::Ok, HostInfo::GpuText(gpus.at(index)), {}};
		if (gpu_index >= gpus.size()) {
			gpu.status      = S::Warn;
			gpu.description = tr("The chosen card %1 is not found; the emulator picks this one.").arg(gpu_index + 1);
		}
	}
	const Check ram {S::Ok, tr("%1 of RAM").arg(HostInfo::Gigabytes(HostInfo::TotalRamBytes())), {}};

	SetCheck(m_ui->check_folder_status, m_ui->check_folder, m_ui->check_folder_description, folder);
	SetCheck(m_ui->check_gpu_status, m_ui->check_gpu, m_ui->check_gpu_description, gpu);
	SetCheck(m_ui->check_ram_status, m_ui->check_ram, m_ui->check_ram_description, ram);
	SetCheck(m_ui->check_cache_status, m_ui->check_cache, m_ui->check_cache_description, CacheCheck(item));
	SetCheck(m_ui->system_gpu_status, m_ui->system_gpu, m_ui->system_gpu_description, gpu);
	SetCheck(m_ui->system_ram_status, m_ui->system_ram, m_ui->system_ram_description, ram);

	const auto version = VersionCheck(item);
	m_ui->check_version_row->setVisible(version.has_value());
	if (version.has_value()) {
		SetCheck(m_ui->check_version_status, m_ui->check_version, m_ui->check_version_description, *version);
	}

	const auto emulator = EmulatorCheck();
	m_ui->check_emulator_row->setVisible(emulator.has_value());
	m_ui->system_emulator_row->setVisible(emulator.has_value());
	if (emulator.has_value()) {
		SetCheck(m_ui->check_emulator_status, m_ui->check_emulator, m_ui->check_emulator_description, *emulator);
		SetCheck(m_ui->system_emulator_status, m_ui->system_emulator, m_ui->system_emulator_description, *emulator);
	}
}

void MainDialogPrivate::UpdateGameArt(const ConfigurationItem* item) {
	const auto key = item != nullptr ? item->GetInfo().basedir : QString();
	if (key == m_game_art_key) {
		return;
	}
	m_game_art_key = key;
	// Read here (archives keep their reader on this thread), decode and scale off the UI thread.
	QByteArray data;
	for (const auto& name: {QStringLiteral("sce_sys/pic0.png"), QStringLiteral("sce_sys/pic1.png")}) {
		if (!key.isEmpty() && data.isEmpty()) {
			data = GameContent::ReadFile(key, name, GameContent::MaxImageSize);
		}
	}
	if (data.isEmpty()) {
		ShowGameArt({});
		return;
	}
	m_game_art.setFuture(QtConcurrent::run([key, data]() {
		QImage image;
		if (image.loadFromData(data) && image.width() > GAME_ART_WIDTH) {
			image = image.scaledToWidth(GAME_ART_WIDTH, Qt::SmoothTransformation);
		}
		return GameArt {key, image};
	}));
}

void MainDialogPrivate::ShowGameArt(const QImage& image) {
	m_ui->play_art->SetPicture(QPixmap::fromImage(image), GAME_ART_INK);
	SetPlayDark(!image.isNull());
}

void MainDialogPrivate::SetPlayDark(bool dark) {
	if (m_ui->play_art->property("dark").toBool() == dark) {
		return;
	}
	m_ui->play_art->setProperty("dark", dark);
	for (auto* widget: m_ui->play_art->findChildren<QWidget*>()) {
		widget->style()->unpolish(widget);
		widget->style()->polish(widget);
		widget->update();
	}
	m_ui->widget->SetDark(dark);
}

// The Play page: the selected game, its checks and the status line of the bottom bar.
void MainDialogPrivate::UpdatePlaySummary(const ConfigurationItem* item) {
	UpdateChecks(item);
	UpdateGameArt(item);
	m_ui->label_game_folder->setText(item != nullptr ? QDir::toNativeSeparators(item->GetInfo().basedir)
	                                                 : tr("No game selected"));
	if (item == nullptr) {
		m_ui->play_eyebrow->setText(tr("No game selected").toUpper());
		m_ui->play_title->setText(tr("Select your game"));
		m_ui->play_status->setText(tr("No game selected"));
		return;
	}
	const auto& game_info = item->GetInfo();
	const bool  folder_ok =
	    !game_info.basedir.isEmpty() && (QDir(game_info.basedir).exists() || GameContent::IsArchive(game_info.basedir));
	const auto game = game_info.title_id.isEmpty() ? game_info.name : game_info.title_id;
	m_ui->play_eyebrow->setText((game + (folder_ok ? tr(" · found") : tr(" · folder missing"))).toUpper());
	const int setup = ShaderSetupStep(item);
	if (m_guide_play_button != nullptr) {
		m_guide_play_button->setVisible(folder_ok && !item->IsRunning() && setup < 2);
		m_guide_play_button->disconnect();
		const int page = setup == 0 ? GUIDE_PAGE_IMPORT_HINTS : GUIDE_PAGE_PRECOMPILE;
		connect(m_guide_play_button, &QPushButton::clicked, this, [this, page]() { OpenGuide(page); });
	}
	if (const auto emulator = EmulatorCheck(); emulator.has_value()) {
		m_ui->play_title->setText(emulator->title);
	} else if (!item->IsRunning() && folder_ok && setup == 0) {
		m_ui->play_title->setText(tr("Import shader hints first"));
	} else if (!item->IsRunning() && folder_ok && setup == 1) {
		m_ui->play_title->setText(tr("Precompile shaders first"));
	} else {
		m_ui->play_title->setText(item->IsRunning() ? tr("Playing") : (folder_ok ? tr("Ready to play") : tr("Game folder missing")));
	}

	// The settings the game starts with (its own, or the global ones).
	const auto info       = m_ui->widget->CreateConfiguration(*item);
	const auto resolution = EnumToText(info->screen_resolution).replace(QLatin1Char('x'), QChar(0x00D7));
	auto       status     = resolution + QStringLiteral(" · ") + (info->fullscreen_enabled ? tr("full screen") : tr("window")) +
	                 (info->tessellation_enabled ? tr(" · tessellation on") : tr(" · tessellation off"));
	if (info->overlay_enabled) {
		status += tr(" · overlay");
	}
	if (const int percent = PrecompiledPercent(g_precompile_progress.value(game_info.title_id)); percent >= 0) {
		status += tr(" · cache %1%").arg(percent);
	}
	m_ui->play_status->setText(status);
}

// Display and Graphics: the selected game's own settings when it has them, else the global ones.
void MainDialogPrivate::LoadSettingsPages() {
	const auto* item    = m_ui->widget->GetSelectedItem();
	const auto* info    = m_ui->widget->EditableConfiguration();
	auto        eyebrow = tr("Settings · all games");
	if (item != nullptr && m_ui->widget->EditsGameSettings()) {
		const auto game = item->GetInfo().title_id.isEmpty() ? item->GetInfo().name : item->GetInfo().title_id;
		eyebrow         = tr("Settings · %1 only").arg(game);
	}
	m_ui->display_eyebrow->setText(eyebrow.toUpper());
	m_ui->graphics_eyebrow->setText(eyebrow.toUpper());

	const QSignalBlocker block_resolution(m_ui->display_resolution);
	const QSignalBlocker block_fullscreen(m_ui->display_fullscreen);
	const QSignalBlocker block_present(m_ui->display_present);
	const QSignalBlocker block_overlay(m_ui->display_overlay);
	const QSignalBlocker block_tessellation(m_ui->graphics_tessellation);
	const QSignalBlocker block_gpu(m_ui->graphics_gpu);

	m_ui->display_resolution->clear();
	for (const auto& resolution: EnumToList<Configuration::Resolution>()) {
		m_ui->display_resolution->addItem(QString(resolution).replace(QLatin1Char('x'), QChar(0x00D7)), resolution);
	}
	m_ui->display_resolution->setCurrentIndex(m_ui->display_resolution->findData(EnumToText(info->screen_resolution)));
	m_ui->display_fullscreen->setChecked(info->fullscreen_enabled);
	m_ui->display_present->clear();
	m_ui->display_present->addItems(EnumToList<Configuration::PresentMode>());
	m_ui->display_present->setCurrentText(EnumToText(info->present_mode));
	m_ui->display_overlay->clear();
	m_ui->display_overlay->addItem(tr("Off"), false);
	m_ui->display_overlay->addItem(tr("On"), true);
	m_ui->display_overlay->setCurrentIndex(info->overlay_enabled ? 1 : 0);

	m_ui->graphics_tessellation->setChecked(info->tessellation_enabled);
	const auto& gpus = HostInfo::Gpus();
	m_ui->graphics_gpu->clear();
	const int automatic = HostInfo::ResolveGpuIndex(-1);
	m_ui->graphics_gpu->addItem(automatic >= 0 ? tr("Auto (%1)").arg(HostInfo::GpuText(gpus.at(automatic))) : tr("Auto"), -1);
	for (int i = 0; i < gpus.size(); i++) {
		m_ui->graphics_gpu->addItem(HostInfo::GpuText(gpus.at(i)), i);
	}
	if (info->gpu_index >= gpus.size()) {
		m_ui->graphics_gpu->addItem(tr("Card %1 (not found)").arg(info->gpu_index + 1), info->gpu_index);
	}
	m_ui->graphics_gpu->setCurrentIndex(std::max(0, m_ui->graphics_gpu->findData(info->gpu_index)));
}

void MainDialogPrivate::ConnectSettingsPages() {
	// Changes go straight into the settings file; the next start uses them.
	const auto apply = [this](auto change) {
		change(*m_ui->widget->EditableConfiguration());
		m_ui->widget->CommitSettings();
	};
	connect(m_ui->display_resolution, &QComboBox::activated, this, [this, apply](int) {
		apply([this](Configuration& info) {
			info.screen_resolution =
			    TextToEnum<Configuration::Resolution>(m_ui->display_resolution->currentData().toString());
		});
	});
	connect(m_ui->display_fullscreen, &QCheckBox::toggled, this,
	        [apply](bool checked) { apply([checked](Configuration& info) { info.fullscreen_enabled = checked; }); });
	connect(m_ui->display_present, &QComboBox::activated, this, [this, apply](int) {
		apply([this](Configuration& info) {
			info.present_mode = TextToEnum<Configuration::PresentMode>(m_ui->display_present->currentText());
		});
	});
	connect(m_ui->display_overlay, &QComboBox::activated, this, [this, apply](int) {
		apply([this](Configuration& info) { info.overlay_enabled = m_ui->display_overlay->currentData().toBool(); });
	});
	connect(m_ui->graphics_tessellation, &QCheckBox::toggled, this,
	        [apply](bool checked) { apply([checked](Configuration& info) { info.tessellation_enabled = checked; }); });
	connect(m_ui->graphics_gpu, &QComboBox::activated, this, [this, apply](int) {
		apply([this](Configuration& info) { info.gpu_index = m_ui->graphics_gpu->currentData().toInt(); });
	});
}

// First start "Choose game folder" and Advanced "Game folder: Change...".
void MainDialogPrivate::ChooseGameFolder() {
	const auto* item  = m_ui->widget->GetSelectedItem();
	const auto  start = item != nullptr ? QFileInfo(item->GetInfo().basedir).absolutePath() : QDir::homePath();
	const auto  dir   = QFileDialog::getExistingDirectory(m_main_dialog, tr("Choose game folder"), start);
	if (dir.isEmpty()) {
		return;
	}
	const bool first_start = m_ui->pages->currentIndex() == PAGE_FIRST_START;
	if (m_ui->widget->AddGameDirectory(dir)) {
		m_ui->first_folder_path->setProperty("error", false);
		if (first_start) {
			ShowPage(PAGE_PLAY);
		}
	} else {
		const auto native = QDir::toNativeSeparators(dir);
		SetCheck(m_ui->first_folder_check_status, m_ui->first_folder_check, m_ui->first_folder_check_description,
		         {StatusIcon::Status::Error, tr("No game found"),
		          tr("No eboot.bin in this folder or below it. Pick the folder that contains eboot.bin and sce_sys.")});
		m_ui->first_folder_path->setText(native);
		m_ui->first_folder_path->setProperty("error", true);
		if (!first_start) {
			QMessageBox::warning(m_main_dialog, tr("Choose game folder"),
			                     tr("No game found in %1: pick the folder that contains eboot.bin and sce_sys.").arg(native));
		}
	}
	m_ui->first_folder_path->style()->unpolish(m_ui->first_folder_path);
	m_ui->first_folder_path->style()->polish(m_ui->first_folder_path);
	Update();
}

QList<BugReport::File> MainDialogPrivate::BugReportFiles() const {
	QString     system;
	QTextStream s(&system);
	s << "KytyPS5 Yōtei fork " << FORK_VERSION << " - bug report, "
	  << QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm")) << "\n";
	s << "Emulator build: " << (m_emulator_version.isEmpty() ? tr("(emulator not found)") : m_emulator_version) << "\n";
	s << "OS: " << QSysInfo::prettyProductName() << " (kernel " << QSysInfo::kernelVersion() << ", "
	  << QSysInfo::currentCpuArchitecture() << ")\n";
	s << "CPU: " << QString::fromStdString(Common::GetSystemInfo().ProcessorName) << "\n";
	s << "RAM: " << HostInfo::Gigabytes(HostInfo::TotalRamBytes()) << "\n";
	s << "Graphics cards (the emulator's --gpu order):\n";
	const auto& gpus = HostInfo::Gpus();
	for (int i = 0; i < gpus.size(); i++) {
		s << "  " << i << ": " << HostInfo::GpuText(gpus.at(i)) << ", driver " << gpus.at(i).driver << "\n";
	}
	if (const auto* item = m_ui->widget->GetSelectedItem(); item != nullptr) {
		const auto& game = item->GetInfo();
		const auto  info = m_ui->widget->CreateConfiguration(*item);
		s << "\nGame: " << game.name << " (" << game.title_id << "), version " << game.gameVersion << ", firmware "
		  << game.firmwareVer << "\n";
		s << "Game folder: " << QDir::toNativeSeparators(game.basedir) << "\n";
		s << "Settings (" << (game.custom_settings ? "the game's own" : "global") << "): "
		  << EnumToText(info->screen_resolution) << ", " << (info->fullscreen_enabled ? "full screen" : "window")
		  << ", tessellation " << (info->tessellation_enabled ? "on" : "off") << ", overlay "
		  << (info->overlay_enabled ? "on" : "off") << ", present mode " << EnumToText(info->present_mode) << ", gpu "
		  << (info->gpu_index < 0 ? QStringLiteral("auto") : QString::number(info->gpu_index)) << "\n";
		const auto progress = g_precompile_progress.value(game.title_id);
		s << "Shaders: precompile on start " << (g_precompile_on_start ? "on" : "off") << ", threads "
		  << (g_precompile_threads > 0 ? QString::number(g_precompile_threads) : QStringLiteral("default"))
		  << ", last precompile " << progress.first << " / " << progress.second << "\n";
	} else {
		s << "\nGame: none selected\n";
	}
	s.flush();

	QList<BugReport::File> files {{QStringLiteral("report/system.txt"), {}, system}};
	if (const auto settings = m_ui->widget->GetSettingsFile(); QFileInfo::exists(settings)) {
		files.append({QStringLiteral("launcher/") + QFileInfo(settings).fileName(), settings, {}});
	}
	if (const auto logs = LogDir(); !logs.isEmpty()) {
		const auto runs = QDir(logs).entryInfoList({QString::fromLatin1(LOG_PREFIX) + QStringLiteral("*.log")},
		                                           QDir::Files, QDir::Time);
		for (qsizetype i = 0; i < std::min<qsizetype>(runs.size(), BUG_REPORT_LOGS); i++) {
			files.append({QStringLiteral("logs/") + runs.at(i).fileName(), runs.at(i).absoluteFilePath(), {}});
		}
	}
	if (!m_interpreter.isEmpty()) {
		const auto log = QFileInfo(m_interpreter).absoluteDir().filePath(PRECOMPILE_LOG_FILE);
		if (QFileInfo::exists(log)) {
			files.append({QStringLiteral("logs/") + QString::fromLatin1(PRECOMPILE_LOG_FILE), log, {}});
		}
	}
	return files;
}

void MainDialogPrivate::ExportBugReport() {
	BugReport::Scrubber scrubber;
	for (const auto& name: m_ui->widget->UserNames()) {
		scrubber.AddUserName(name);
	}
	BugReport::Dialog dialog(BugReportFiles(), scrubber, m_main_dialog);
	dialog.exec();
}

// KYTY_LAUNCHER_SCREENSHOT_DIR: each page at the Figma frame size (1280x800), plus Shaders with a running precompile
// fed through the same PRECOMPILE_* parser as the emulator's output, the bug report dialog on a sample log (saved as
// bug-report-sample.zip) and, when <dir>/test-cache/_PipelineCache exists (a copy made for the test), Clear cache on
// that copy (clear-cache-test.txt). The emulator file is looked for (Checks) but never started, and its own cache is
// never touched here.
void MainDialogPrivate::SaveScreenshots(const QString& dir) {
	QDir().mkpath(dir);
	m_main_dialog->resize(1280, 800);
	const auto grab = [&dir](QWidget* widget, const QString& name) {
		QApplication::processEvents();
		QImage image(widget->size(), QImage::Format_ARGB32_Premultiplied);
		image.setDevicePixelRatio(1.0); // design pixels, whatever the display scale
		widget->render(&image);
		image.save(QDir(dir).filePath(QStringLiteral("launcher-%1.png").arg(name)));
	};
	const auto save = [this, &grab](int page, const QString& name) {
		ShowPage(page);
		m_ui->pages->setCurrentIndex(page); // also Play when no game is found
		if (auto* button = m_nav->button(page); button != nullptr) {
			button->setChecked(true);
		}
		if (page == PAGE_PLAY) {
			m_game_art.waitForFinished(); // the selected game's picture
			QApplication::processEvents();
		}
		grab(m_main_dialog, name);
	};
	Update();
	save(PAGE_PLAY, QStringLiteral("play"));
	if (auto* tiles = m_ui->widget->findChild<GameTileView*>(QStringLiteral("tiles"));
	    tiles != nullptr && tiles->model() != nullptr && tiles->model()->rowCount() > 1) {
		// The second tile under the mouse (its hover look).
		const auto centre = QPointF(tiles->visualRect(tiles->model()->index(1, 0)).center());
		QHoverEvent hover(QEvent::HoverMove, centre, tiles->viewport()->mapToGlobal(centre), QPointF(-1, -1));
		QApplication::sendEvent(tiles->viewport(), &hover);
		grab(m_main_dialog, QStringLiteral("play-hover"));
	}
	save(PAGE_DISPLAY, QStringLiteral("display"));
	save(PAGE_GRAPHICS, QStringLiteral("graphics"));
	save(PAGE_SHADERS, QStringLiteral("shaders"));
	m_ui->precompile_progress->setVisible(true);
	m_ui->precompile_pause_button->setVisible(true);
	m_ui->precompile_button->setText(tr("Continue precompiling"));
	PrecompileLine(QStringLiteral("PRECOMPILE_START total=28700 cached=0 old_journals=0"));
	PrecompileLine(QStringLiteral("PRECOMPILE_PROGRESS done=17840 total=28700 compute=9120 graphics=8720 eta_s=1860 phase=compile"));
	save(PAGE_SHADERS, QStringLiteral("shaders-precompiling"));
	save(PAGE_ADVANCED, QStringLiteral("advanced"));
	save(PAGE_FIRST_START, QStringLiteral("first-start"));

	// The bug report on a sample log with this PC's real user, computer and temporary paths.
	const QDir sample(QDir(dir).filePath(QStringLiteral("bug-report-sample")));
	sample.mkpath(QStringLiteral("."));
	const auto sample_log = sample.filePath(QStringLiteral("kyty-20261023-170000.log"));
	if (QFile file(sample_log); file.open(QIODevice::WriteOnly | QIODevice::Text)) {
		const auto home = QDir::toNativeSeparators(QDir::homePath());
		QTextStream(&file) << "KytyPS5 Yōtei fork " << FORK_VERSION << " launcher, 2026-10-23T17:00:00\n"
		                   << "Emulator build: sample\n"
		                   << "Command: \"" << home << "\\Games\\KytyPS5\\kyty_emulator.exe\" \"--screen-width\" "
		                   << "\"3840\" \"--screen-height\" \"2160\" \"--user-name\" \"Erlou\" \"--user-id\" \"1000\" "
		                   << "\"--game\" \"D:\\Games\\PPSA26344\\eboot.bin\"\n\n"
		                   << "ENV_DEFAULT KYTY_SHADER_PRECOMPILE=1 source=code kind=config\n"
		                   << "Vulkan device: NVIDIA GeForce RTX 5070 Ti\n"
		                   << "pipeline cache: " << home << "\\Games\\KytyPS5\\_PipelineCache\\PPSA26344.bin\n"
		                   << "temp: " << QDir::toNativeSeparators(QDir::tempPath()) << "\\kyty_spirv.tmp\n"
		                   << "TEMP variable: " << qEnvironmentVariable("TEMP") << "\n"
		                   << "host: " << QSysInfo::machineHostName() << ", user Erlou signed in\n";
	}
	BugReport::Scrubber scrubber;
	scrubber.AddUserName(QStringLiteral("Erlou"));
	for (const auto& name: m_ui->widget->UserNames()) {
		scrubber.AddUserName(name);
	}
	auto files = BugReportFiles();
	files.append({QStringLiteral("logs/kyty-20261023-170000.log"), sample_log, {}});
	{
		BugReport::Dialog dialog(files, scrubber, m_main_dialog);
		dialog.show();
		if (auto* list = dialog.findChild<QTreeWidget*>(QStringLiteral("bug_report_files")); list != nullptr) {
			list->setCurrentItem(list->topLevelItem(list->topLevelItemCount() - 1)); // the sample log
		}
		grab(&dialog, QStringLiteral("bug-report"));
		dialog.SaveTo(QDir(dir).filePath(QStringLiteral("bug-report-sample.zip")));
		dialog.close();
	}

	const auto test_cache = QDir(dir).filePath(QStringLiteral("test-cache/_PipelineCache"));
	if (QDir(test_cache).exists()) {
		const auto cache = ListCacheFiles(test_cache);
		{
			QMessageBox box(QMessageBox::Question, tr("Clear cache"), ClearCacheQuestion(test_cache, cache),
			                QMessageBox::Yes | QMessageBox::No, m_main_dialog);
			box.show();
			grab(&box, QStringLiteral("clear-cache"));
		}
		const auto failed = MoveToTrash(cache.files);
		const auto left   = ListCacheFiles(test_cache);
		QFile      report(QDir(dir).filePath(QStringLiteral("clear-cache-test.txt")));
		if (report.open(QIODevice::WriteOnly | QIODevice::Text)) {
			QTextStream(&report) << "folder " << QDir::toNativeSeparators(test_cache) << "\nfiles " << cache.files.size()
			                     << " (" << cache.bytes << " bytes)\nmoved to the Recycle Bin "
			                     << cache.files.size() - failed.size() << "\nfailed " << failed.size() << "\nleft "
			                     << left.files.size() << "\n";
		}
	}
	QTimer::singleShot(0, qApp, &QApplication::quit);
}

void MainDialogPrivate::FindInterpreter() {
	// The picture mode looks for the emulator file the same way, but never starts it.
	const auto screenshot_dir = qEnvironmentVariable(SCREENSHOT_DIR_ENV);

	QDir search_dir(QApplication::applicationDirPath());
	m_emulator_path = search_dir.absoluteFilePath(EMULATOR_EXE);
	if (!QFile::exists(m_emulator_path)) {
		search_dir.cdUp();
		m_emulator_path = search_dir.absoluteFilePath(EMULATOR_EXE);
	}
	m_emulator = QFile::exists(m_emulator_path) ? Emulator::Found : Emulator::Missing;

	if (m_emulator == Emulator::Found && screenshot_dir.isEmpty()) {
		QProcess test;
		test.setProgram(m_emulator_path);
		test.start();
		test.waitForFinished();

		auto output = QString(test.readAllStandardOutput());
		auto lines  = output.split(QRegularExpression("[\r\n]"), Qt::SkipEmptyParts);

		if (lines.count() >= 2) {
			m_emulator_version = lines.at(0).startsWith("exe_name") ? lines.at(1) : lines.at(0);
			m_ui->row_version_description->setText(tr("by Erlou · emulator build %1").arg(m_emulator_version));
		} else {
			m_emulator = Emulator::NoAnswer;
		}
	}

	// Without a working emulator the launcher stays open: Checks says what is wrong, PLAY and Precompile are off.
	const bool found = m_emulator == Emulator::Found;
	m_interpreter    = found ? m_emulator_path : QString();
	m_ui->widget->SetRuntimeDirectory(found ? QFileInfo(m_emulator_path).absolutePath() : QApplication::applicationDirPath());
	m_ui->label_Interpreter->setText(found ? QDir::toNativeSeparators(m_emulator_path) : EmulatorCheck()->title);
	m_ui->label_settings_file->setText(m_ui->widget->GetSettingsFile());
	m_ui->label_log_folder->setText(found ? QDir::toNativeSeparators(LogDir()) : QStringLiteral("—"));
	m_ui->log_folder_button->setEnabled(found);

	if (!screenshot_dir.isEmpty()) {
		SaveScreenshots(screenshot_dir);
		return;
	}

	// Without any game the First start page asks for the game folder.
	ShowPage(PAGE_PLAY);
	Update();
	if (m_ui->check_updates_on_startup->isChecked()) {
		m_update_checker->Check(false);
	}
}

static QString BoolArg(bool value) {
	return value ? QStringLiteral("true") : QStringLiteral("false");
}

static QStringList CreateEmulatorArgs(const Configuration& info) {
	QStringList args;
	auto        r = EnumToText(info.screen_resolution).split('x');

	if (r.size() != 2) {
		return {};
	}

	args << "--screen-width" << r.at(0);
	args << "--screen-height" << r.at(1);
	args << "--user-name" << info.user_name;
	args << "--user-id" << QString::number(info.user_id);
	if (!info.audio_input_device.isEmpty()) {
		args << "--mic" << info.audio_input_device;
	}
	if (!info.controller.color.isEmpty()) {
		args << "--controller-color" << info.controller.color;
	}
	args << "--controller-volume" << QString::number(info.controller.speaker_volume);
	args << "--controller-vibration" << QString::number(info.controller.vibration_intensity);
	args << "--present-mode" << EnumToText(info.present_mode);
	if (info.gpu_index >= 0) {
		args << "--gpu" << QString::number(info.gpu_index);
	}
	if (info.fullscreen_enabled) {
		args << "--fullscreen";
	}
	if (info.hide_cursor_enabled) {
		args << "--hide-cursor";
	}
	if (info.overlay_enabled) {
		args << "--perf-overlay";
	}
	args << "--readback-linear-images" << BoolArg(info.readback_linear_images);
	// (This fork's emulator: trophy toasts on by default, "--no-trophy-notifications" turns them off.)
	if (!info.trophy_enabled) {
		args << "--no-trophy-notifications";
	}
	// Both ways: the emulator's default is on, so "off" must be said.
	args << (info.tessellation_enabled ? "--tessellation" : "--no-tessellation");
	args << "--vblank-frequency" << QString::number(info.vblank_frequency);
	args << "--console-language" << QString::number(info.console_language);
	args << "--vulkan-validation" << BoolArg(info.vulkan_validation_enabled);
	args << "--shader-validation" << BoolArg(info.shader_validation_enabled);
	args << "--shader-optimization-type" << EnumToText(info.shader_optimization_type);
	args << "--shader-log-direction" << EnumToText(info.shader_log_direction);
	args << "--shader-log-folder" << info.shader_log_folder;
	args << "--command-buffer-dump" << BoolArg(info.command_buffer_dump_enabled);
	args << "--command-buffer-dump-folder" << info.command_buffer_dump_folder;
	args << "--printf-direction" << EnumToText(info.printf_direction);
	args << "--printf-output-file" << info.printf_output_file;
	if (info.profiler_enabled) {
		args << "--profile";
	}
	args << "--spirv-debug-printf" << "false";
	if (info.amd_cpu_enabled) {
		args << "--amd-cpu";
	}
#if defined(_WIN32)
	if (info.red_zone_protection_enabled) {
		args << "--redzone";
	}
#endif
	for (const auto& binding: info.host_input_mapping) {
		args << "--keymap" << binding;
	}
	if (info.renderdoc_enabled) {
		args << "--rd";
	}

	QString game = info.basedir;
	if (!info.elf.isEmpty() && !GameContent::IsArchive(info.basedir)) {
		game = QDir(info.basedir).filePath(info.elf);
	}
	args << "--game" << game;

	WriteDefaultMods(info.title_id, info.gameVersion);
	const auto patch_plan = Cheats::PlanPath(info.title_id);
	if (QFileInfo::exists(patch_plan)) {
		args << "--game-patch" << patch_plan;
	}

	return args;
}

#ifdef __linux__
static QString BashQuote(QString value) {
	value.replace('\'', "'\\''");
	return QStringLiteral("'") + value + QStringLiteral("'");
}

static bool CreateBashScript(const QString& interpreter, const QStringList& args,
                             const QString& file_name, const QString& log_file) {
	QFile file(file_name);
	if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
		QTextStream s(&file);

		s << "#!/bin/bash\n";
		s << BashQuote(interpreter);
		for (const auto& arg: args) {
			s << " " << BashQuote(arg);
		}
		if (!log_file.isEmpty()) {
			s << " 2>> " << BashQuote(log_file);
		}
		s << "\n";
		s << "echo Press any key...\n";
		s << "read -n1\n";

		file.close();

		return file.setPermissions(file.permissions() | QFile::ExeUser | QFile::ExeOwner |
		                           QFile::ExeGroup);
	}
	return false;
}

// Find a terminal and its command separator.
static bool FindTerminal(QString* program, QStringList* prefix) {
	struct TerminalSpec {
		const char* executable;
		const char* separator; // nullptr when the command follows immediately
	};

	static const TerminalSpec candidates[] = {
	    {"x-terminal-emulator", "-e"},
	    {"gnome-terminal", "--"},
	    {"konsole", "-e"},
	    {"xfce4-terminal", "-x"},
	    {"mate-terminal", "--"},
	    {"tilix", "-e"},
	    {"alacritty", "-e"},
	    {"kitty", nullptr},
	    {"foot", nullptr},
	    {"wezterm", "-e"},
	    {"urxvt", "-e"},
	    {"xterm", "-e"},
	};

	const auto try_candidate = [program, prefix](const QString& executable, const char* separator) {
		const auto resolved = QStandardPaths::findExecutable(executable);
		if (resolved.isEmpty()) {
			return false;
		}
		*program = resolved;
		prefix->clear();
		if (separator != nullptr) {
			*prefix << QString::fromLatin1(separator);
		}
		return true;
	};

	if (const auto from_env = qEnvironmentVariable("TERMINAL"); !from_env.isEmpty()) {
		// Reuse the known separator for an explicit terminal.
		const auto  env_name  = QFileInfo(from_env).fileName();
		const char* separator = "-e";
		for (const auto& candidate: candidates) {
			if (env_name == QLatin1String(candidate.executable)) {
				separator = candidate.separator;
				break;
			}
		}
		if (try_candidate(from_env, separator)) {
			return true;
		}
	}

	for (const auto& candidate: candidates) {
		if (try_candidate(QString::fromLatin1(candidate.executable), candidate.separator)) {
			return true;
		}
	}

	return false;
}
#endif

#if defined(_WIN32)
// Quote one token for cmd.exe so paths with spaces survive /K parsing.
static QString WinCmdQuote(QString value) {
	value.replace(QLatin1Char('"'), QStringLiteral("\\\""));
	return QLatin1Char('"') + value + QLatin1Char('"');
}

static QString BuildWinCmdKCommand(const QString& interpreter, const QStringList& args) {
	QString command = WinCmdQuote(QDir::toNativeSeparators(interpreter));
	for (const auto& arg: args) {
		command += QLatin1Char(' ');
		command += WinCmdQuote(arg);
	}
	return command;
}
#endif

void MainDialog::RunInterpreter(QProcess* process, const Configuration& info) {
	const auto& interpreter = m_p->GetInterpreter();

	QFileInfo f(interpreter);
	auto      dir = f.absoluteDir();

	auto args = CreateEmulatorArgs(info);
	if (args.isEmpty()) {
		QMessageBox::critical(this, tr("Error"), tr("Invalid emulator configuration"));
		QApplication::quit();
		return;
	}

	QString command = QDir::toNativeSeparators(interpreter);
	for (const auto& arg: args) {
		command += QStringLiteral(" \"") + arg + QLatin1Char('"');
	}
	const auto log_file = m_p->StartRunLog(command);
	process->setProcessEnvironment(m_p->GameEnvironment(info));

#ifdef __linux__
	auto bash_file_name = dir.filePath(KYTY_BASH_FILE);
	if (!CreateBashScript(interpreter, args, bash_file_name, log_file)) {
		QMessageBox::critical(this, tr("Error"), tr("Can't create file:\n") + bash_file_name);
		QApplication::quit();
		return;
	}

	{
		QString     terminal;
		QStringList terminal_prefix;
		// Pass the script as a file argument (not bash -c) so paths with spaces work.
		if (FindTerminal(&terminal, &terminal_prefix)) {
			process->setProgram(terminal);
			process->setArguments(terminal_prefix + QStringList {"bash", bash_file_name});
		} else {
			// Run without a terminal as a fallback.
			process->setProgram(QStringLiteral("bash"));
			process->setArguments({bash_file_name});
		}
	}
#elif defined(_WIN32)
	{
		// Use nativeArguments so Qt does not re-quote the /K command string.
		process->setProgram(CMD_EXE);
		process->setArguments({});
		// The emulator's stderr goes to the run log (its stdout stays in the console window).
		const auto redirect = log_file.isEmpty() ? QString() : QStringLiteral(" 2>>") + WinCmdQuote(QDir::toNativeSeparators(log_file));
		process->setNativeArguments(QStringLiteral("/K \"") + BuildWinCmdKCommand(interpreter, args) + redirect +
		                            QLatin1Char('"'));
	}
#else
	process->setProgram(interpreter);
	process->setArguments(args);
	if (!log_file.isEmpty()) {
		process->setStandardErrorFile(log_file, QIODevice::Append);
	}
#endif
	process->setWorkingDirectory(dir.path());
#if defined(_WIN32)
	process->setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* args) {
		args->flags |= static_cast<uint32_t>(CREATE_NEW_CONSOLE);
		args->startupInfo->dwFlags &= ~static_cast<DWORD>(STARTF_USESTDHANDLES);
		args->startupInfo->dwFlags |= static_cast<DWORD>(STARTF_USECOUNTCHARS);
		args->startupInfo->dwXCountChars = CMD_X_CHARS;
		args->startupInfo->dwYCountChars = CMD_Y_CHARS;
		// args->startupInfo->dwFlags |= static_cast<DWORD>(STARTF_USEFILLATTRIBUTE);
		// args->startupInfo->dwFillAttribute =
		//     static_cast<DWORD>(BACKGROUND_BLUE) | static_cast<DWORD>(FOREGROUND_RED) |
		//     static_cast<DWORD>(FOREGROUND_INTENSITY);
	});
#endif
	process->start();
#if !defined(_WIN32)
	// Report immediate launch failures.
	if (!process->waitForStarted(5000)) {
		QMessageBox::critical(
		    this, tr("Error"),
		    tr("Failed to start:\n%1\n\n%2").arg(process->program(), process->errorString()));
		return;
	}
#endif
	process->waitForFinished(100);
}

QProcessEnvironment MainDialogPrivate::GameEnvironment(const Configuration& info) const {
	auto env = QProcessEnvironment::systemEnvironment();
	if (!g_precompile_on_start) {
		// Off: no background compiles and no shader journals (Common::EnvDefaults has it on).
		env.insert(QStringLiteral("KYTY_SHADER_PRECOMPILE"), QStringLiteral("0"));
	}
	if (g_precompile_threads > 0) {
		env.insert(QStringLiteral("KYTY_SHADER_PRECOMPILE_THREADS"), QString::number(g_precompile_threads));
	}
	// The Mods page's emulator-setting mods: an enabled plan entry with an "env" object (no memory writes).
	if (QFile plan(Cheats::PlanPath(info.title_id)); plan.open(QIODevice::ReadOnly)) {
		for (const auto& value: QJsonDocument::fromJson(plan.readAll()).object().value(QStringLiteral("mods")).toArray()) {
			const auto mod = value.toObject();
			if (!mod.value(QStringLiteral("enabled")).toBool(true)) continue;
			const auto vars = mod.value(QStringLiteral("env")).toObject();
			for (auto it = vars.begin(); it != vars.end(); ++it) env.insert(it.key(), it.value().toString());
		}
	}
	return env;
}

QString MainDialogPrivate::StartRunLog(const QString& command) const {
	const auto dir_path = LogDir();
	if (dir_path.isEmpty() || !QDir().mkpath(dir_path)) {
		return {};
	}
	const QDir dir(dir_path);
	const auto pattern = QString::fromLatin1(LOG_PREFIX) + QStringLiteral("*.log");
	const auto old     = dir.entryInfoList({pattern}, QDir::Files, QDir::Time);
	for (qsizetype i = LOG_KEEP - 1; i < old.size(); i++) {
		QFile::remove(old.at(i).absoluteFilePath());
	}
	const auto now  = QDateTime::currentDateTime();
	const auto path = dir.filePath(QString::fromLatin1(LOG_PREFIX) + now.toString(QStringLiteral("yyyyMMdd-HHmmss")) +
	                               QStringLiteral(".log"));
	QFile file(path);
	if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
		return {};
	}
	QTextStream(&file) << "KytyPS5 Yōtei fork " << FORK_VERSION << " launcher, " << now.toString(Qt::ISODate) << "\n"
	                   << "Emulator build: " << m_emulator_version << "\n"
	                   << "Command: " << command << "\n\n";
	return path;
}

void MainDialog::WriteSettings(QSettings& s) {
	MainDialogPrivate::WriteSettings(s);
}

void MainDialog::ReadSettings(QSettings& s) {
	MainDialogPrivate::ReadSettings(s);
}

void MainDialog::resizeEvent(QResizeEvent* event) {
	emit Resize();
	QDialog::resizeEvent(event);
}

void MainDialogPrivate::WriteSettings(QSettings& s) {
	s.beginGroup(SETTINGS_MAIN_DIALOG);

	if (!g_last_geometry.isEmpty()) {
		s.setValue(SETTINGS_MAIN_LAST_GEOMETRY, g_last_geometry);
	}
	s.setValue(SETTINGS_CHECK_UPDATES, g_check_updates_on_startup);
	s.setValue(SETTINGS_PRECOMPILE_THREADS, g_precompile_threads);
	s.setValue(SETTINGS_PRECOMPILE_ON_START, g_precompile_on_start);
	s.remove(SETTINGS_PRECOMPILE_PROGRESS);
	s.beginGroup(SETTINGS_PRECOMPILE_PROGRESS);
	for (auto it = g_precompile_progress.constBegin(); it != g_precompile_progress.constEnd(); ++it) {
		s.setValue(it.key(), QStringLiteral("%1 %2").arg(it.value().first).arg(it.value().second));
	}
	s.endGroup();
	s.beginGroup(QStringLiteral("ShaderSetup"));
	for (auto it = g_shader_setup.constBegin(); it != g_shader_setup.constEnd(); ++it) {
		s.setValue(it.key(), it.value());
	}
	s.endGroup();

	s.endGroup();
}

void MainDialogPrivate::ReadSettings(QSettings& s) {
	s.beginGroup(SETTINGS_MAIN_DIALOG);

	g_last_geometry = s.value(SETTINGS_MAIN_LAST_GEOMETRY, g_last_geometry).toByteArray();
	g_check_updates_on_startup = s.value(SETTINGS_CHECK_UPDATES, true).toBool();
	g_precompile_threads       = std::clamp(s.value(SETTINGS_PRECOMPILE_THREADS, 0).toInt(), 0, 64);
	g_precompile_on_start      = s.value(SETTINGS_PRECOMPILE_ON_START, true).toBool();
	g_precompile_progress.clear();
	s.beginGroup(SETTINGS_PRECOMPILE_PROGRESS);
	for (const auto& title_id: s.childKeys()) {
		const auto parts = s.value(title_id).toString().split(QLatin1Char(' '));
		if (parts.size() == 2) {
			g_precompile_progress.insert(title_id, {parts.at(0).toLongLong(), parts.at(1).toLongLong()});
		}
	}
	s.endGroup();
	g_shader_setup.clear();
	s.beginGroup(QStringLiteral("ShaderSetup"));
	for (const auto& key: s.childKeys()) {
		g_shader_setup.insert(key, std::clamp(s.value(key).toInt(), 0, 2));
	}
	s.endGroup();

	s.endGroup();
}

void MainDialogPrivate::Run() {
	m_running_item = m_ui->widget->GetSelectedItem();
	if (m_running_item == nullptr || m_interpreter.isEmpty()) {
		return;
	}

	// The first start of a game whose hints the package ships: its shaders are prepared from them first (the game and
	// a precompile use the same cache files), then the game starts by itself.
	const auto& running = m_running_item->GetInfo();
	if (const auto hints = FirstStartHints(running.title_id, running.gameVersion); !hints.isEmpty()) {
		m_first_start_hints.insert(m_running_item->GetInfo().title_id);
		m_running_item          = nullptr;
		m_play_after_precompile = true;
		ShowPage(PAGE_SHADERS);
		m_hints_file = hints;
		Precompile({QStringLiteral("--import-hints"), QDir::toNativeSeparators(hints)});
		return;
	}

	m_running_item->SetRunning(true);
	m_lightbar.Stop();

	auto info = m_ui->widget->CreateConfiguration(*m_running_item);
	m_main_dialog->RunInterpreter(&m_process, *info);

	Update();
}

// "1:02:03" or "12:34".
static QString FormatDuration(qint64 seconds) {
	seconds = std::max<qint64>(seconds, 0);
	const auto hours   = seconds / 3600;
	const auto minutes = (seconds / 60) % 60;
	const auto rest    = seconds % 60;
	if (hours > 0) {
		return QStringLiteral("%1:%2:%3")
		    .arg(hours)
		    .arg(minutes, 2, 10, QLatin1Char('0'))
		    .arg(rest, 2, 10, QLatin1Char('0'));
	}
	return QStringLiteral("%1:%2").arg(minutes).arg(rest, 2, 10, QLatin1Char('0'));
}

// The key=value fields of a PRECOMPILE_* line (kyty_emulator --precompile-only).
static QMap<QString, QString> PrecompileFields(const QString& line) {
	QMap<QString, QString> fields;
	for (const auto& part: line.split(QLatin1Char(' '), Qt::SkipEmptyParts)) {
		const auto separator = part.indexOf(QLatin1Char('='));
		if (separator > 0) {
			fields.insert(part.left(separator), part.mid(separator + 1));
		}
	}
	return fields;
}

// "17 840" (groups of three, as in the Figma progress line).
static QString GroupDigits(qint64 value) {
	auto text = QString::number(value);
	for (auto i = text.size() - 3; i > (value < 0 ? 1 : 0); i -= 3) {
		text.insert(i, QLatin1Char(' '));
	}
	return text;
}

// "~31 min left", "~1 h 5 min left", "< 1 min left"; negative: not known yet.
static QString EtaText(qint64 seconds) {
	if (seconds < 0) {
		return QObject::tr("estimating the time left");
	}
	const auto minutes = (seconds + 59) / 60;
	if (minutes <= 1) {
		return QObject::tr("< 1 min left");
	}
	if (minutes < 60) {
		return QObject::tr("~%1 min left").arg(minutes);
	}
	return QObject::tr("~%1 h %2 min left").arg(minutes / 60).arg(minutes % 60);
}

void MainDialogPrivate::ImportHints() {
	if (m_precompile.state() != QProcess::NotRunning || m_ui->widget->GetSelectedItem() == nullptr) {
		return;
	}
	const auto file = QFileDialog::getOpenFileName(m_main_dialog, tr("Import shader hints"),
	                                               QDir(HintsDir()).exists() ? HintsDir() : QDir::homePath(),
	                                               tr("Shader hints (*.kyhints);;All files (*)"));
	if (file.isEmpty()) {
		return;
	}
	m_hints_file = file;
	Precompile({QStringLiteral("--import-hints"), QDir::toNativeSeparators(file)});
}

void MainDialogPrivate::ExportHints() {
	const auto* item = m_ui->widget->GetSelectedItem();
	if (m_precompile.state() != QProcess::NotRunning || item == nullptr) {
		return;
	}
	const auto& info = item->GetInfo();
	QDir().mkpath(HintsDir());
	const auto suggested = QDir(HintsDir()).filePath(
        QStringLiteral("%1_%2.kyhints").arg(info.title_id.isEmpty() ? QStringLiteral("game") : info.title_id,
                                             info.gameVersion.isEmpty() ? QStringLiteral("shaders") : info.gameVersion));
	const auto file = QFileDialog::getSaveFileName(m_main_dialog, tr("Export shader hints"), suggested,
	                                               tr("Shader hints (*.kyhints)"));
	if (file.isEmpty()) {
		return;
	}
	m_hints_file = file;
	Precompile({QStringLiteral("--export-hints"), QDir::toNativeSeparators(file)});
}

void MainDialogPrivate::Precompile(const QStringList& extra) {
	if (m_precompile.state() != QProcess::NotRunning) {
		return;
	}
	if (extra.isEmpty()) {
		m_hints_file.clear();
	}
	m_precompile_hints.clear();

	auto* item = m_ui->widget->GetSelectedItem();
	if (item == nullptr) {
		return;
	}
	// "Precompile shaders" on a first start: the shipped hints are imported with it.
	if (const auto hints = FirstStartHints(item->GetInfo().title_id, item->GetInfo().gameVersion); extra.isEmpty() && !hints.isEmpty()) {
		m_first_start_hints.insert(item->GetInfo().title_id);
		m_hints_file = hints;
		Precompile({QStringLiteral("--import-hints"), QDir::toNativeSeparators(hints)});
		return;
	}
	auto info = m_ui->widget->CreateConfiguration(*item);
	auto args = CreateEmulatorArgs(*info);
	if (args.isEmpty()) {
		QMessageBox::critical(m_main_dialog, tr("Error"), tr("Invalid emulator configuration"));
		return;
	}
	// The same arguments as the game's run (the same device and settings, so the same pipelines), without the game.
	args << QStringLiteral("--precompile-only") << extra;
	if (g_precompile_threads > 0) {
		args << QStringLiteral("--precompile-threads") << QString::number(g_precompile_threads);
	}

	const QDir runtime_dir = QFileInfo(m_interpreter).absoluteDir();
	m_precompile.setProgram(m_interpreter);
	m_precompile.setArguments(args);
	m_precompile.setWorkingDirectory(runtime_dir.path());
	m_precompile.setStandardErrorFile(runtime_dir.filePath(PRECOMPILE_LOG_FILE));
#if defined(_WIN32)
	m_precompile.setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* process_args) {
		process_args->flags |= static_cast<uint32_t>(CREATE_NO_WINDOW);
	});
#endif

	m_precompile_stop_sent = false;
	m_precompile_title     = info->title_id;
	m_precompile_cached    = 0;
	m_precompile_old       = 0;
	m_precompile_done.clear();
	m_ui->precompile_progress->setRange(0, 0); // busy until PRECOMPILE_START
	m_ui->precompile_progress->setVisible(true);
	m_ui->precompile_count->setVisible(false);
	m_ui->precompile_eta->setVisible(false);
	m_ui->precompile_status->setVisible(true);
	m_ui->precompile_status->setText(
	    m_play_after_precompile
	        ? tr("First start: preparing this game's shaders from the included hints (a few minutes). The game starts "
	             "when they are ready.")
	    : extra.contains(QStringLiteral("--import-hints"))
	        ? tr("Importing hints: looking for their shaders in the game files (can take a few minutes)...")
	    : extra.contains(QStringLiteral("--export-hints")) ? tr("Exporting hints...")
	                                                       : tr("Starting: loading the pipeline cache and the shader journals..."));
	m_ui->precompile_pause_button->setText(tr("Pause"));
	m_precompile.start();
	Update();
}

void MainDialogPrivate::PausePrecompile() {
	if (m_precompile.state() == QProcess::NotRunning) {
		return;
	}
	if (!m_precompile_stop_sent) {
		// The emulator finishes the shaders it is compiling, saves the pipeline cache and exits.
		m_precompile_stop_sent = true;
		m_precompile.write("quit\n");
		m_ui->precompile_pause_button->setText(tr("Force stop"));
		m_ui->precompile_status->setVisible(true);
		m_ui->precompile_status->setText(
		    tr("Pausing: finishing the shaders being compiled (up to about a minute), then saving..."));
	} else {
		m_precompile.kill();
	}
}

void MainDialogPrivate::PrecompileOutput() {
	while (m_precompile.canReadLine()) {
		const auto line = QString::fromUtf8(m_precompile.readLine()).trimmed();
		if (line.startsWith(QLatin1String("PRECOMPILE_"))) {
			PrecompileLine(line);
		}
	}
}

void MainDialogPrivate::PrecompileLine(const QString& line) {
	const auto kind   = line.section(QLatin1Char(' '), 0, 0);
	const auto fields = PrecompileFields(line);
	if (kind == QLatin1String("PRECOMPILE_START")) {
		const int total     = fields.value(QStringLiteral("total")).toInt();
		m_precompile_cached = fields.value(QStringLiteral("cached")).toInt();
		m_precompile_old    = fields.value(QStringLiteral("old_journals")).toInt();
		m_ui->precompile_progress->setRange(0, std::max(total, 1));
		m_ui->precompile_progress->setValue(0);
		g_precompile_progress[m_precompile_title] = {m_precompile_cached, m_precompile_cached + total};
	} else if (kind == QLatin1String("PRECOMPILE_PROGRESS")) {
		const int  done  = fields.value(QStringLiteral("done")).toInt();
		const int  total = fields.value(QStringLiteral("total")).toInt();
		const auto phase = fields.value(QStringLiteral("phase"));
		const auto eta   = fields.value(QStringLiteral("eta_s")).toLongLong();
		m_ui->precompile_progress->setRange(0, std::max(total, 1));
		m_ui->precompile_progress->setValue(total == 0 ? 1 : done);
		m_ui->precompile_count->setText(tr("%1 / %2 shaders").arg(GroupDigits(done), GroupDigits(total)));
		m_ui->precompile_count->setVisible(true);
		m_ui->precompile_eta->setText(EtaText(eta));
		m_ui->precompile_eta->setVisible(phase == QLatin1String("compile"));
		g_precompile_progress[m_precompile_title] = {m_precompile_cached + done, m_precompile_cached + total};
		QStringList notes;
		if (phase == QLatin1String("stopping")) {
			notes << tr("Pausing: finishing the shaders being compiled (up to about a minute), then saving...");
		} else if (phase == QLatin1String("save")) {
			notes << tr("Saving the pipeline cache...");
		}
		if (m_precompile_cached > 0) {
			notes << tr("%1 compiled before.").arg(GroupDigits(m_precompile_cached));
		}
		if (m_precompile_old > 0) {
			notes << tr("%1 shader journal(s) from an older emulator version skipped.").arg(m_precompile_old);
		}
		m_ui->precompile_status->setText(notes.join(QLatin1Char(' ')));
		m_ui->precompile_status->setVisible(!notes.isEmpty());
	} else if (kind == QLatin1String("PRECOMPILE_DONE")) {
		m_precompile_done = fields;
	} else if (kind == QLatin1String("PRECOMPILE_HINTS")) {
		m_precompile_hints = fields;
	}
}

// The PRECOMPILE_HINTS line in words: what an import found, or why it was not used.
static QString HintsText(const QMap<QString, QString>& hints) {
	if (hints.isEmpty() || hints.value(QStringLiteral("kind")) != QLatin1String("import")) {
		return {};
	}
	if (hints.value(QStringLiteral("status")) != QLatin1String("ok")) {
		const auto reason = hints.value(QStringLiteral("reason"));
		if (reason == QLatin1String("other_emulator_version")) {
			return QObject::tr("Hints not used: they were made by another emulator version.");
		}
		if (reason == QLatin1String("other_shader_mode")) {
			return QObject::tr("Hints not used: they were made with other shader settings.");
		}
		if (reason == QLatin1String("other_game")) {
			return QObject::tr("Hints not used: they are for another game.");
		}
		if (reason == QLatin1String("other_version")) {
			return QObject::tr("Hints not used: they are for game version %1, your game is %2.")
			    .arg(hints.value(QStringLiteral("hints_version")), hints.value(QStringLiteral("game_version")));
		}
		return QObject::tr("Hints not used: this is not a shader hints file.");
	}
	return QObject::tr("Hints: %1 of %2 shaders found in your game files, %3 new.")
	    .arg(hints.value(QStringLiteral("found")), hints.value(QStringLiteral("hints")))
	    .arg(hints.value(QStringLiteral("added_compute")).toInt() + hints.value(QStringLiteral("added_stages")).toInt());
}

void MainDialogPrivate::PrecompileFinished(int exit_code, QProcess::ExitStatus exit_status) {
	PrecompileOutput();
	const auto status   = m_precompile_done.value(QStringLiteral("status"));
	const auto done     = m_precompile_done.value(QStringLiteral("done")).toInt();
	const auto total    = m_precompile_done.value(QStringLiteral("total")).toInt();
	const auto elapsed  = m_precompile_done.value(QStringLiteral("elapsed_s")).toLongLong();
	const bool save_bad = m_precompile_done.value(QStringLiteral("saved")) == QLatin1String("0");
	const auto log_path = QFileInfo(m_interpreter).absoluteDir().filePath(PRECOMPILE_LOG_FILE);
	QString    text;
	m_ui->precompile_eta->setVisible(false);
	if (status == QLatin1String("exported") || status == QLatin1String("export_failed")) {
		const auto& hints = m_precompile_hints;
		if (status == QLatin1String("exported")) {
			text = tr("Hints saved to %1: %2 shaders, %3 pipelines. The file holds no game code.")
			           .arg(QDir::toNativeSeparators(m_hints_file))
			           .arg(hints.value(QStringLiteral("compute")).toInt() + hints.value(QStringLiteral("stages")).toInt())
			           .arg(hints.value(QStringLiteral("pipelines")).toInt());
		} else if (hints.value(QStringLiteral("reason")) == QLatin1String("no_journal")) {
			text = tr("No hints to save yet: play the game first, the shaders it uses are recorded then.");
		} else {
			text = tr("Hints not saved: the file could not be written. Log: %1").arg(log_path);
		}
		m_ui->precompile_progress->setVisible(false);
		m_ui->precompile_count->setVisible(false);
		m_ui->precompile_status->setText(text);
		m_ui->precompile_status->setVisible(true);
		Update();
		return;
	}
	if (status == QLatin1String("complete")) {
		text = total == 0 ? tr("Shaders ready: nothing new to compile (%1 compiled before).").arg(m_precompile_cached)
		                  : tr("Shaders ready: %1 compiled in %2.").arg(total).arg(FormatDuration(elapsed));
		m_ui->precompile_progress->setRange(0, 1);
		m_ui->precompile_progress->setValue(1);
		g_precompile_progress[m_precompile_title] = {m_precompile_cached + total, m_precompile_cached + total};
	} else if (status == QLatin1String("stopped")) {
		text = tr("Paused: %1 of %2 done and saved. Continue precompiling goes on from there.").arg(done).arg(total);
		g_precompile_progress[m_precompile_title] = {m_precompile_cached + done, m_precompile_cached + total};
	} else if (status == QLatin1String("unavailable")) {
		text = m_precompile_done.value(QStringLiteral("reason")) == QLatin1String("precompile_off")
		           ? tr("Not available: shader precompile is off (KYTY_SHADER_PRECOMPILE=0).")
		           : tr("Not available: this emulator build keeps no pipeline cache (a debug or uncommitted build, "
		                "or the game has no title id).");
		m_ui->precompile_progress->setVisible(false);
		m_ui->precompile_count->setVisible(false);
	} else {
		text = exit_status == QProcess::CrashExit || m_precompile_stop_sent
		           ? tr("Precompile ended early; what was saved is kept. Log: %1").arg(log_path)
		           : tr("Precompile ended without finishing (exit code %1). Log: %2").arg(exit_code).arg(log_path);
	}
	if (save_bad) {
		text += tr(" Saving the pipeline cache failed. Log: %1").arg(log_path);
	}
	if (const auto hints = HintsText(m_precompile_hints); !hints.isEmpty()) {
		text = hints + QLatin1Char(' ') + text;
	}
	m_ui->precompile_status->setText(text);
	m_ui->precompile_status->setVisible(true);
	// v0.1.6 first-start steps: an import that found the hints usable, then a precompile that finished.
	if (const auto key = ShippedHintsKey(m_ui->widget->GetSelectedItem()); !key.isEmpty()) {
		int step = g_shader_setup.value(key, 0);
		if (m_precompile_hints.value(QStringLiteral("kind")) == QLatin1String("import") &&
		    m_precompile_hints.value(QStringLiteral("status")) == QLatin1String("ok")) {
			step = std::max(step, 1);
		}
		if (step >= 1 && status == QLatin1String("complete")) step = 2;
		g_shader_setup[key] = step;
	}
	m_ui->widget->WriteSettings();
	Update();
	// Play on a first start: the game now, once its shaders are ready (not after a pause or a failure).
	if (std::exchange(m_play_after_precompile, false) && status == QLatin1String("complete")) {
		Run();
	}
}

void MainDialogPrivate::Update() {
	const auto* item = m_ui->widget->GetSelectedItem();

	bool run_enabled = (m_process.state() == QProcess::NotRunning && item != nullptr && !m_interpreter.isEmpty());

	if (run_enabled) {
		const auto& info = item->GetInfo();
		auto        dir  = info.basedir;
		run_enabled      = !dir.isEmpty() && (QDir(dir).exists() || GameContent::IsArchive(dir));
	}

	// The game and a shader precompile use the same pipeline cache files: one at a time.
	const bool precompiling = m_precompile.state() != QProcess::NotRunning;
	if (!precompiling) {
		const auto progress = item != nullptr ? g_precompile_progress.value(item->GetInfo().title_id) : QPair<qint64, qint64> {};
		m_ui->precompile_button->setText(progress.first > 0 && progress.first < progress.second ? tr("Continue precompiling")
		                                                                                        : tr("Precompile shaders"));
	}
	// v0.1.6: with shipped hints, Precompile waits for their import and Play for a finished precompile.
	const int setup = ShaderSetupStep(item);
	{
		// v0.1.6: red frame until the shipped hints' shaders are prepared to the end, green when done.
		const QString state = ShippedHintsKey(item).isEmpty() ? QString() : (setup >= 2 ? QStringLiteral("done") : QStringLiteral("todo"));
		for (QWidget* button: {static_cast<QWidget*>(m_ui->precompile_button), static_cast<QWidget*>(m_ui->play_precompile_button)}) {
			if (button->property("shaderState").toString() != state) {
				button->setProperty("shaderState", state);
				button->style()->unpolish(button);
				button->style()->polish(button);
				button->update();
			}
		}
	}
	m_ui->precompile_button->setEnabled(!precompiling && run_enabled && setup >= 1);
	m_ui->hints_import_button->setEnabled(!precompiling && run_enabled);
	m_ui->hints_export_button->setEnabled(!precompiling && run_enabled);
	m_ui->precompile_pause_button->setVisible(precompiling);
	m_ui->play_precompile_button->setEnabled(precompiling || (run_enabled && setup >= 1));
	m_ui->precompile_threads->setEnabled(!precompiling);
	run_enabled = run_enabled && !precompiling && setup >= 2;

	m_ui->widget->SetRunEnabled(run_enabled);
	m_ui->play_button->setEnabled(run_enabled);
	UpdatePlaySummary(item);
	if (const int page = m_ui->pages->currentIndex(); page == PAGE_SHADERS) {
		UpdateCacheInfo();
	} else if (page == PAGE_DISPLAY || page == PAGE_GRAPHICS) {
		LoadSettingsPages();
	}
	if (m_process.state() != QProcess::NotRunning) {
		m_lightbar.Stop();
		return;
	}
	m_lightbar.SetColor(m_ui->widget->GetGlobalControllerColor());
}

#include "mainDialog.moc"
