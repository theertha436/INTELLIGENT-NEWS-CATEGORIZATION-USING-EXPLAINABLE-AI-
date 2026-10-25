
import argostranslate.package
import argostranslate.translate

LANGUAGES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "ml": "Malayalam",
    "kn": "Kannada",
    "te": "Telugu",
    "bn": "Bengali",
}

print("Refreshing Argos package index...")
try:
    argostranslate.package.update_package_index()
except Exception as exc:
    print(f"Could not refresh package index: {exc}")
    print("Check your internet connection and try again.")
    raise SystemExit(1)

installed = argostranslate.translate.get_installed_languages()
installed_codes = {lang.code for lang in installed}

if "en" not in installed_codes:
    print("English is not installed. Install a language pair containing English first.")

for code, name in LANGUAGES.items():
    print(f"\nChecking {name} ({code}) -> English...")

    try:
        packages = argostranslate.package.get_available_packages()

        package = next(
            (
                p for p in packages
                if p.from_code == code and p.to_code == "en"
            ),
            None,
        )

        if package is None:
            print(f"No direct {name} -> English package found.")
            continue

        print(
            f"Downloading {name} -> English "
            "translation model. This can take several minutes..."
        )

        path = package.download()
        argostranslate.package.install_from_path(path)

        print(f"Successfully installed {name} -> English.")

    except KeyboardInterrupt:
        print(f"\nDownload of {name} was interrupted.")
        print("Run this installer again when ready.")
        break

    except Exception as exc:
        print(f"Could not install {name} -> English: {exc}")
        print("Continuing to the next language...")

print("\nInstalled languages:")
for lang in argostranslate.translate.get_installed_languages():
    print(f"- {lang.name} ({lang.code})")

print("\nInstallation check complete.")
