#ifndef LAUNCHER_INCLUDE_FORK_INFO_H_
#define LAUNCHER_INCLUDE_FORK_INFO_H_

// The Yōtei fork's own version and addresses (sidebar footer, About, bug report, update check).

constexpr char FORK_VERSION[] = "v0.1.4";

// The fork's GitHub repository (no trailing slash): the About "Source code" button opens it and the bug report opens
// <address>/issues/new after saving the zip. Empty: the button is hidden and the bug report only saves the zip.
constexpr char FORK_REPOSITORY_URL[] = "https://github.com/Erlou93/KytyPS5-Yotei";

// The GitHub API address of the fork's latest release, "https://api.github.com/repos/<owner>/<repo>/releases/latest".
// Empty: the update check is off (the upstream KytyPS5 feeds announce upstream releases, not the fork's). Fill it in
// together with FORK_REPOSITORY_URL once the fork publishes releases whose tags match KYTY_RELEASE_TAG.
constexpr char FORK_RELEASES_FEED_URL[] = "";

// The game and game versions the fork is tested and tuned on (Play checks: "Game version ... · tested"; any other
// version or game gets a yellow "untested" note, PLAY stays on). When the fork moves to a new game patch, add the
// version here (as written in the game's sce_sys/param.json "appVersion" / "contentVersion"); the first is the main one.
constexpr char        TESTED_TITLE_ID[]       = "PPSA26344";
constexpr char        TESTED_GAME_NAME[]      = "Ghost of Yōtei";
constexpr const char* TESTED_GAME_VERSIONS[] = {"02.002.000"};

#endif // LAUNCHER_INCLUDE_FORK_INFO_H_
