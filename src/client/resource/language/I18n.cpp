#include "I18n.hpp"

#include <format>

#include "client/util/Language.hpp"

Language* I18n::language = nullptr;

std::string I18n::translate(const std::string& key, const std::vector<std::string>& args) {
    auto& lang = Language::getInstance();
    std::string pattern = lang.get(key, key);

    std::string result = pattern;
    for (size_t i = 0; i < args.size(); i++) {
        auto pos = result.find("%s");
        if (pos != std::string::npos) {
            result.replace(pos, 2, args[i]);
        }
    }
    return result;
}

bool I18n::hasTranslation(const std::string& key) {
    return Language::getInstance().hasTranslation(key);
}

void I18n::setLanguage(Language* lang) {
    language = lang;
}
