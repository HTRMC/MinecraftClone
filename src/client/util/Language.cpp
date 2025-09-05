#include "Language.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

std::unique_ptr<Language> Language::instance = nullptr;

Language& Language::getInstance() {
    if (!instance) {
        instance = loadFromFile("./assets/minecraft/lang/en_us.json");
        if (!instance) {
            instance = std::unique_ptr<Language>(new Language());
            std::cerr << "Warning: no language file loaded, translations unavailable.\n";
        }
    }
    return *instance;
}

void Language::setInstance(std::unique_ptr<Language> lang) {
    instance = std::move(lang);
}

std::unique_ptr<Language> Language::loadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open language file: " << path << "\n";
        return nullptr;
    }

    json j;
    file >> j;

    auto lang = std::unique_ptr<Language>(new Language());
    for (auto it = j.begin(); it != j.end(); ++it) {
        lang->translations[it.key()] = it.value().get<std::string>();
    }

    return lang;
}

std::string Language::get(const std::string& key, const std::string& fallback) const {
    auto it = translations.find(key);
    if (it != translations.end()) {
        return it->second;
    }
    return fallback;
}

bool Language::hasTranslation(const std::string& key) const {
    return translations.find(key) != translations.end();
}
