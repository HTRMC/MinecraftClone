#pragma once
#include <string>
#include <vector>

#include "client/util/Language.hpp"

class I18n {
public:
    static std::string translate(const std::string& key, const std::vector<std::string>& args = {});

    static bool hasTranslation(const std::string& key);

    static void setLanguage(Language* lang);

private:
    static Language* language;
};
