#ifndef _ENVIRONMENT_H_
#define _ENVIRONMENT_H_

#include <vector>
#include <string>

#include <sutil/Frame.h>
#include <sutil/sutilapi.h>

enum OptType {
    OPT_INT,
    OPT_FLOAT,
    OPT_BOOL,
    OPT_VECTOR3,
    OPT_VECTOR4,
    OPT_FRAME,
    OPT_STRING
};

struct Option {
    OptType type;
    std::string name;
    std::vector<std::string> values;
    std::string defaultValue;
};

class Environment {

private:

    static Environment* instance;

    std::vector<Option> options;
    int numberOfOptions;

    SUTILAPI bool filterValue(const std::string& value, std::string& filteredValue, OptType type);
    SUTILAPI bool findOption(const std::string& name, Option& option);

protected:

    SUTILAPI void registerOption(const std::string& name, const std::string defaultValue, OptType type);
    SUTILAPI void registerOption(const std::string& name, OptType type);
    SUTILAPI virtual void registerOptions(void) = 0;

public:

    SUTILAPI static Environment* getInstance(void);
    SUTILAPI static void deleteInstance(void);
    SUTILAPI static void setInstance(Environment* instance);

    SUTILAPI Environment(void);
    SUTILAPI virtual ~Environment(void);

    SUTILAPI bool getIntValue(const std::string& name, int& value);
    SUTILAPI bool getFloatValue(const std::string& name, float& value);
    SUTILAPI bool getBoolValue(const std::string& name, bool& value);
    SUTILAPI bool getVector3Value(const std::string& name, float3& value);
    SUTILAPI bool getVector4Value(const std::string& name, float4& value);
    SUTILAPI bool getFrameValue(const std::string& name, Frame& value);
    SUTILAPI bool getStringValue(const std::string& name, std::string& value);

    SUTILAPI bool getIntValues(const std::string& name, std::vector<int>& values);
    SUTILAPI bool getFloatValues(const std::string& name, std::vector<float>& values);
    SUTILAPI bool getBoolValues(const std::string& name, std::vector<bool>& values);
    SUTILAPI bool getVector3Values(const std::string& name, std::vector<float3>& values);
    SUTILAPI bool getVector4Values(const std::string& name, std::vector<float4>& values);
    SUTILAPI bool getFrameValues(const std::string& name, std::vector<Frame>& values);
    SUTILAPI bool getStringValues(const std::string& name, std::vector<std::string>& values);

    SUTILAPI bool parse(int argc, char** argv);
    SUTILAPI bool readEnvFile(const std::string& filename);

};

#endif /* _ENVIRONMENT_H_ */
