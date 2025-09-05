#pragma once
#include <string>
#include <unordered_map>
#include <memory>

class Language {
public:
    static constexpr const char* DEFAULT_LANGUAGE = "en_us";

    static Language& getInstance();
    static void setInstance(std::unique_ptr<Language> lang);

    std::string get(const std::string& key, const std::string& fallback) const;
    bool hasTranslation(const std::string& key) const;

    static std::unique_ptr<Language> loadFromFile(const std::string& path);

private:
    Language() = default;
    std::unordered_map<std::string, std::string> translations;

    static std::unique_ptr<Language> instance;
};
