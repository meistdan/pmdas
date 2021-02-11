#include <fstream>
#include <iostream>
#include <sstream>

#include "Environment.h"

Environment* Environment::instance = nullptr;

bool Environment::filterValue(const std::string& value, std::string& filteredValue, OptType type) {
    bool valid = true;
    std::stringstream ss(value);
    if (type == OPT_INT) {
        int val;
        ss >> val;
        filteredValue = std::to_string(val);
    }
    else if (type == OPT_FLOAT) {
        float val;
        ss >> val;
        filteredValue = std::to_string(val);
    }
    else if (type == OPT_BOOL) {
        if (value == "true" || value == "yes" || value == "on" || value == "1" ||
            value == "TRUE" || value == "YES" || value == "ON") {
            filteredValue = "1";
        }
        else if (value == "false" || value == "no" || value == "off" || value == "0" ||
            value == "FALSE" || value == "NO" || value == "OFF") {
            filteredValue = "0";
        }
        else {
            valid = false;
        }
    }
    else if (type == OPT_VECTOR3) {
        float val0, val1, val2;
        ss >> val0;
        ss >> val1;
        ss >> val2;
        filteredValue = std::to_string(val0) + " " + std::to_string(val1) + " " + std::to_string(val2);
    }
    else if (type == OPT_VECTOR4) {
        float val0, val1, val2, val3;
        ss >> val0;
        ss >> val1;
        ss >> val2;
        ss >> val3;
        filteredValue = std::to_string(val0) + " " + std::to_string(val1) + " " 
                      + std::to_string(val2) + " " + std::to_string(val3);
    }
    else if (type == OPT_FRAME) {
        Frame value;
        ss >> value.scale.x;
        ss >> value.scale.y;
        ss >> value.scale.z;
        ss >> value.translate.x;
        ss >> value.translate.y;
        ss >> value.translate.z;
        ss >> value.rotate.w;
        ss >> value.rotate.x;
        ss >> value.rotate.y;
        ss >> value.rotate.z;
        filteredValue = std::to_string(value.scale.x) + " " + std::to_string(value.scale.y) + " " + std::to_string(value.scale.z) + " "
            + std::to_string(value.translate.x) + " " + std::to_string(value.translate.y) + " " + std::to_string(value.translate.z) + " "
            + std::to_string(value.rotate.w) + " " + std::to_string(value.rotate.x) + " " + std::to_string(value.rotate.y) + " " + std::to_string(value.rotate.z);
    }
    else {
        filteredValue = value;
        if (value.empty()) valid = false;
    }
    return valid;

}

bool Environment::findOption(const std::string& name, Option& option) {
    std::vector<Option>::iterator i;
    for (i = options.begin(); i != options.end(); ++i) {
        if (i->name == name) {
            option = *i;
            return true;
        }
    }
    return false;
}

void Environment::registerOption(const std::string& name, const std::string defaultValue, OptType type) {
    Option opt;
    if (!filterValue(defaultValue, opt.defaultValue, type)) {
        std::cerr << "Invalid default value for option '" << name << "'." << std::endl;
        exit(EXIT_FAILURE);
    }
    opt.name = name;
    opt.type = type;
    options.push_back(opt);
}

void Environment::registerOption(const std::string& name, OptType type) {
    Option opt;
    opt.name = name;
    opt.type = type;
    options.push_back(opt);
}

Environment* Environment::getInstance() {
    if (!instance)
        std::cerr << "Environment is not allocated." << std::endl;
    return instance;
}

void Environment::deleteInstance() {
    if (instance) {
        delete instance;
        instance = nullptr;
    }
}

void Environment::setInstance(Environment* instance) {
    Environment::instance = instance;
}

Environment::Environment() {
}

Environment::~Environment() {
}

bool Environment::getIntValue(const std::string& name, int& value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        std::stringstream ss(opt.values.front());
        ss >> value;
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        std::stringstream ss(opt.defaultValue);
        ss >> value;
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getFloatValue(const std::string& name, float& value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        std::stringstream ss(opt.values.front());
        ss >> value;
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        std::stringstream ss(opt.defaultValue);
        ss >> value;
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getBoolValue(const std::string& name, bool& value) {
    Option opt;
    int ivalue;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        std::stringstream ss(opt.values.front());
        ss >> ivalue;
        value = static_cast<bool>(ivalue);
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        std::stringstream ss(opt.defaultValue);
        ss >> ivalue;
        value = static_cast<bool>(ivalue);
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getVector3Value(const std::string& name, float3& value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        std::stringstream ss(opt.values.front());
        ss >> value.x;
        ss >> value.y;
        ss >> value.z;
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        std::stringstream ss(opt.defaultValue);
        ss >> value.x;
        ss >> value.y;
        ss >> value.z;
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getVector4Value(const std::string& name, float4& value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        std::stringstream ss(opt.values.front());
        ss >> value.x;
        ss >> value.y;
        ss >> value.z;
        ss >> value.w;
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        std::stringstream ss(opt.defaultValue);
        ss >> value.x;
        ss >> value.y;
        ss >> value.z;
        ss >> value.z;
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getFrameValue(const std::string& name, Frame& value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        std::stringstream ss(opt.values.front());
        ss >> value.scale.x;
        ss >> value.scale.y;
        ss >> value.scale.z;
        ss >> value.translate.x;
        ss >> value.translate.y;
        ss >> value.translate.z;
        ss >> value.rotate.w;
        ss >> value.rotate.x;
        ss >> value.rotate.y;
        ss >> value.rotate.z;
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        std::stringstream ss(opt.defaultValue);
        ss >> value.scale.x;
        ss >> value.scale.y;
        ss >> value.scale.z;
        ss >> value.translate.x;
        ss >> value.translate.y;
        ss >> value.translate.z;
        ss >> value.rotate.w;
        ss >> value.rotate.x;
        ss >> value.rotate.y;
        ss >> value.rotate.z;
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getStringValue(const std::string& name, std::string& value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        value = opt.values.front();
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        value = opt.defaultValue;
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getIntValues(const std::string& name, std::vector<int>& values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        values.clear();
        std::vector<std::string>::iterator i;
        for (i = opt.values.begin(); i != opt.values.end(); ++i) {
            int value;
            std::stringstream ss(*i);
            ss >> value;
            values.push_back(value);
        }
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        int value;
        std::stringstream ss(opt.defaultValue);
        ss >> value;
        values.push_back(value);
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getFloatValues(const std::string& name, std::vector<float>& values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        values.clear();
        std::vector<std::string>::iterator i;
        for (i = opt.values.begin(); i != opt.values.end(); ++i) {
            float value;
            std::stringstream ss(*i);
            ss >> value;
            values.push_back(value);
        }
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        float value;
        std::stringstream ss(opt.defaultValue);
        ss >> value;
        values.push_back(value);
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getBoolValues(const std::string& name, std::vector<bool>& values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        values.clear();
        std::vector<std::string>::iterator i;
        for (i = opt.values.begin(); i != opt.values.end(); ++i) {
            int value;
            std::stringstream ss(*i);
            ss >> value;
            values.push_back(static_cast<bool>(value));
        }
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        int value;
        std::stringstream ss(opt.defaultValue);
        ss >> value;
        values.push_back(static_cast<bool>(value));
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getVector3Values(const std::string& name, std::vector<float3>& values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        values.clear();
        std::vector<std::string>::iterator i;
        for (i = opt.values.begin(); i != opt.values.end(); ++i) {
            float3 value;
            std::stringstream ss(*i);
            ss >> value.x;
            ss >> value.y;
            ss >> value.z;
            values.push_back(value);
        }
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        float3 value;
        std::stringstream ss(opt.defaultValue);
        ss >> value.x;
        ss >> value.y;
        ss >> value.z;
        values.push_back(value);
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getVector4Values(const std::string& name, std::vector<float4>& values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        values.clear();
        std::vector<std::string>::iterator i;
        for (i = opt.values.begin(); i != opt.values.end(); ++i) {
            float4 value;
            std::stringstream ss(*i);
            ss >> value.x;
            ss >> value.y;
            ss >> value.z;
            ss >> value.w;
            values.push_back(value);
        }
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        float4 value;
        std::stringstream ss(opt.defaultValue);
        ss >> value.x;
        ss >> value.y;
        ss >> value.z;
        ss >> value.z;
        values.push_back(value);
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getFrameValues(const std::string& name, std::vector<Frame>& values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        values.clear();
        std::vector<std::string>::iterator i;
        for (i = opt.values.begin(); i != opt.values.end(); ++i) {
            Frame value;
            std::stringstream ss(*i);
            ss >> value.scale.x;
            ss >> value.scale.y;
            ss >> value.scale.z;
            ss >> value.translate.x;
            ss >> value.translate.y;
            ss >> value.translate.z;
            ss >> value.rotate.w;
            ss >> value.rotate.x;
            ss >> value.rotate.y;
            ss >> value.rotate.z;
            values.push_back(value);
        }
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        Frame value;
        std::stringstream ss(opt.defaultValue);
        ss >> value.scale.x;
        ss >> value.scale.y;
        ss >> value.scale.z;
        ss >> value.translate.x;
        ss >> value.translate.y;
        ss >> value.translate.z;
        ss >> value.rotate.w;
        ss >> value.rotate.x;
        ss >> value.rotate.y;
        ss >> value.rotate.z;
        values.push_back(value);
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getStringValues(const std::string& name, std::vector<std::string>& values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.front().empty()) {
        values = opt.values;
        return true;
    }
    else if (!opt.defaultValue.empty()) {
        values.push_back(opt.defaultValue);
        return true;
    }
    else {
        return false;
    }
}

bool Environment::parse(int argc, char** argv) {
    std::string envFilename;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        int index = arg.find_last_of(".env");
        if (index != -1) {
            envFilename = argv[i];
            break;
        }
    }
    if (envFilename.empty())
        envFilename = "default.env";
    return readEnvFile(envFilename);
}

bool Environment::readEnvFile(const std::string& filename) {

    std::ifstream in(filename);

    if (!in.good()) {
        std::cerr << "Cannot open file '" << filename << "'!" << std::endl;;
        return false;
    }

    std::string blockName;
    std::string last;
    std::string value;
    std::string line;
    bool block = false;
    int lineNumber = 0;

    // Read line
    while (std::getline(in, line)) {

        ++lineNumber;

        // Get rid of comments
        int commentIndex = line.find("#");
        if (commentIndex != -1) {
            line = line.substr(0, commentIndex);
        }

        // Split line
        std::stringstream ss(line);
        std::vector<std::string> words{ std::istream_iterator<std::string>{ss},
                      std::istream_iterator<std::string>{} };

        // Process words
        for (int i = 0; i < words.size(); ++i) {

            if (words[i] == "{") {
                blockName = last;
                block = true;
            }

            else if (words[i] == "}") {

                if (blockName.empty()) {
                    std::cerr << "Unpaired } in '" << filename << "' (line " << line << ")." << std::endl;
                    in.close();
                    return false;
                }
                block = false;
                blockName.clear();
            }

            else if (block) {

                bool found = false;
                std::string optionName = blockName + "." + words[i];
                std::vector<Option>::iterator j;
                for (j = options.begin(); j != options.end(); ++j) {
                    if (optionName == j->name) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    std::cerr << "Unknown option '" <<
                        optionName << "' in environment file '" << filename << "' (line " << lineNumber << ")." << std::endl;
                    std::cout << words[i-1] << " " << line << std::endl;
                    in.close();
                    return false;
                }
                else {

                    switch (j->type) {

                    case OPT_INT: {
                        if (i + 1 >= words.size() || !filterValue(words[++i], value, OPT_INT)) {
                            std::cerr << "Mismatch in int variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ")." << std::endl;
                            in.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        break;
                    }

                    case OPT_FLOAT: {
                        if (i + 1 >= words.size() || !filterValue(words[++i], value, OPT_FLOAT)) {
                            std::cerr << "Mismatch in float variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ")." << std::endl;
                            in.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        break;
                    }

                    case OPT_BOOL: {
                        if (i + 1 >= words.size() || !filterValue(words[++i], value, OPT_BOOL)) {
                            std::cerr << "Mismatch in bool variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ")." << std::endl;
                            in.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        break;
                    }

                    case OPT_VECTOR3: {
                        if (i + 3 >= words.size() || !filterValue(words[i + 1] + " " + words[i + 2] + " " + words[i + 3], value, OPT_VECTOR3)) {
                            std::cerr << "Mismatch in vector variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ")." << std::endl;
                            in.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        i += 3;
                        break;
                    }

                    case OPT_VECTOR4: {
                        if (i + 4 >= words.size() || !filterValue(words[i + 1] + " " + words[i + 2] + " " + words[i + 3] + " " + words[i + 4], value, OPT_VECTOR4)) {
                            std::cerr << "Mismatch in vector variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ")." << std::endl;
                            in.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        i += 4;
                        break;
                    }

                    case OPT_FRAME: {
                        if (i + 10 >= words.size() || !filterValue(words[i + 1] + " " + words[i + 2] + " " + words[i + 3] + " " + 
                            words[i + 4] + " " + words[i + 5] + " " + words[i + 6] + " " + words[i + 7] + " " + words[i + 8] + " " + words[i + 9] + " " + words[i + 10], value, OPT_FRAME)) {
                            std::cerr << "Mismatch in vector variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ")." << std::endl;
                            in.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        i += 10;
                        break;
                    }

                    case OPT_STRING: {
                        if (i + 1 >= words.size() || !filterValue(words[++i], value, OPT_STRING)) {
                            std::cerr << "Mismatch in string variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ")." << std::endl;
                            in.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        break;
                    }

                    }

                }

            }

            last = words[i];

        }

    }

    in.close();
    return true;

}
