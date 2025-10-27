/*
 * PQS Framework Launcher
 * Compiled binary that launches the Python app
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <limits.h>

int main(int argc, char *argv[]) {
    char path[PATH_MAX];
    char resources[PATH_MAX];
    char lib_path[PATH_MAX];
    char python_path_env[PATH_MAX * 2];
    
    // Get the directory of the executable
    if (realpath(argv[0], path) == NULL) {
        system("osascript -e 'display alert \"Error\" message \"Cannot determine executable path\" as critical'");
        return 1;
    }
    
    // Get the Resources directory
    char *dir = dirname(dirname(path));
    snprintf(resources, sizeof(resources), "%s/Resources", dir);
    snprintf(lib_path, sizeof(lib_path), "%s/lib", resources);
    
    // Find Python
    const char *python_paths[] = {
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
        NULL
    };
    
    const char *python = NULL;
    for (int i = 0; python_paths[i] != NULL; i++) {
        if (access(python_paths[i], X_OK) == 0) {
            python = python_paths[i];
            break;
        }
    }
    
    if (python == NULL) {
        system("osascript -e 'display alert \"Python Required\" message \"PQS Framework requires Python 3.\\n\\nInstall with:\\nbrew install python3\" as critical'");
        return 1;
    }
    
    // Set environment with bundled libraries
    snprintf(python_path_env, sizeof(python_path_env), "%s:%s", resources, lib_path);
    setenv("PYTHONPATH", python_path_env, 1);
    setenv("PQS_RESOURCES", resources, 1);
    setenv("PYTHONNOUSERSITE", "1", 1);
    
    // Change to resources directory
    if (chdir(resources) != 0) {
        system("osascript -e 'display alert \"Error\" message \"Cannot access Resources directory\" as critical'");
        return 1;
    }
    
    // Use execl to replace this process with Python
    // This ensures the app stays running properly
    execl(python, python, "universal_pqs_app.py", NULL);
    
    // If execl returns, there was an error
    system("osascript -e 'display alert \"Launch Error\" message \"Failed to start Python application\" as critical'");
    return 1;
}
