#include "Language.hpp"
#include <fstream>
#include <simdjson.h>
#include <iostream>

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

    // Read entire file into string for simdjson
    std::string json_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    auto parse_error = parser.parse(json_content).get(doc);
    if (parse_error) {
        std::cerr << "Failed to parse JSON from file: " << path << " - " << parse_error << "\n";
        return nullptr;
    }

    auto lang = std::unique_ptr<Language>(new Language());
    
    // Iterate through all key-value pairs in the JSON object
    for (auto [key, value] : doc.get_object()) {
        std::string_view key_view = key;
        std::string_view value_view;
        if (value.get_string().get(value_view) == simdjson::SUCCESS) {
            lang->translations[std::string(key_view)] = std::string(value_view);
        }
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
