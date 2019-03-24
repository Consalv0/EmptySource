#include "../../include/Utility/DeviceFunctions.h"

#ifdef __APPLE__
static io_connect_t conn;

typedef char              SMCBytes_t[32];
typedef char              UInt32Char_t[5];

typedef struct {
    UInt32Char_t            key;
    UInt32                  dataSize;
    UInt32Char_t            dataType;
    SMCBytes_t              bytes;
} SMCVal_t;

typedef struct {
    char                  major;
    char                  minor;
    char                  build;
    char                  reserved[1];
    UInt16                release;
} SMCKeyData_vers_t;

typedef struct {
    UInt16                version;
    UInt16                length;
    UInt32                cpuPLimit;
    UInt32                gpuPLimit;
    UInt32                memPLimit;
} SMCKeyData_pLimitData_t;

typedef struct {
    UInt32                dataSize;
    UInt32                dataType;
    char                  dataAttributes;
} SMCKeyData_keyInfo_t;

typedef struct {
    UInt32                  key;
    SMCKeyData_vers_t       vers;
    SMCKeyData_pLimitData_t pLimitData;
    SMCKeyData_keyInfo_t    keyInfo;
    char                    result;
    char                    status;
    char                    data8;
    UInt32                  data32;
    SMCBytes_t              bytes;
} SMCKeyData_t;

kern_return_t SMCCall(int index, SMCKeyData_t *inputStructure, SMCKeyData_t *outputStructure) {
    size_t   structureInputSize;
    size_t   structureOutputSize;
    
    structureInputSize = sizeof(SMCKeyData_t);
    structureOutputSize = sizeof(SMCKeyData_t);
    
#if MAC_OS_X_VERSION_10_5
    return IOConnectCallStructMethod( conn, index,
                                     // inputStructure
                                     inputStructure, structureInputSize,
                                     // ouputStructure
                                     outputStructure, &structureOutputSize );
#else
    return IOConnectMethodStructureIStructureO( conn, index,
                                               structureInputSize, /* structureInputSize */
                                               &structureOutputSize,   /* structureOutputSize */
                                               inputStructure,        /* inputStructure */
                                               outputStructure);       /* ouputStructure */
#endif
    
}

kern_return_t SMCReadKey(UInt32Char_t key, SMCVal_t *val) {
    kern_return_t result;
    SMCKeyData_t  inputStructure;
    SMCKeyData_t  outputStructure;
    
    memset(&inputStructure, 0, sizeof(SMCKeyData_t));
    memset(&outputStructure, 0, sizeof(SMCKeyData_t));
    memset(val, 0, sizeof(SMCVal_t));
    
    UInt32 total = 0;
    for (int i = 0; i < 4; i++) {
        total += key[i] << (4 - 1 - i) * 8;
    }
    inputStructure.key = total;
    // --- Read key info
    inputStructure.data8 = 9;
    
    result = SMCCall(2, &inputStructure, &outputStructure);
    if (result != kIOReturnSuccess)
        return result;
    
    val->dataSize = outputStructure.keyInfo.dataSize;
    val->dataType[0] = '\0';
    sprintf(val->dataType, "%c%c%c%c",
            (unsigned int)outputStructure.keyInfo.dataType >> 24,
            (unsigned int)outputStructure.keyInfo.dataType >> 16,
            (unsigned int)outputStructure.keyInfo.dataType >> 8,
            (unsigned int)outputStructure.keyInfo.dataType);
    inputStructure.keyInfo.dataSize = val->dataSize;
    // --- Read data
    inputStructure.data8 = 5;
    
    result = SMCCall(2, &inputStructure, &outputStructure);
    if (result != kIOReturnSuccess)
        return result;
    
    memcpy(val->bytes, outputStructure.bytes, sizeof(outputStructure.bytes));
    
    return kIOReturnSuccess;
}
#endif

bool Debug::InitializeDeviceFunctions() {
#ifdef WIN32
    nvmlReturn_t DeviceResult = nvmlInit();
    
    if (NVML_SUCCESS != DeviceResult) {
        Debug::Log(Debug::LogError, L"NVML :: Failed to initialize: %ls", CharToWChar(nvmlErrorString(DeviceResult)));
        return false;
    }
#elif __APPLE__
    kern_return_t result;
    io_iterator_t iterator;
    io_object_t   device;
    
    CFMutableDictionaryRef matchingDictionary = IOServiceMatching("AppleSMC");
    result = IOServiceGetMatchingServices(kIOMasterPortDefault, matchingDictionary, &iterator);
    if (result != kIOReturnSuccess) {
        Debug::Log(Debug::LogError, L"Device Functions error : IOServiceGetMatchingServices() = %08x", result);
        return false;
    }
    
    device = IOIteratorNext(iterator);
    IOObjectRelease(iterator);
    if (device == 0) {
        Debug::Log(Debug::LogError, L"Device Functions error : no SMC found");
        return false;
    }
    
    result = IOServiceOpen(device, mach_task_self(), 0, &conn);
    IOObjectRelease(device);
    if (result != kIOReturnSuccess) {
        Debug::Log(Debug::LogError, L"Device Functions error : IOServiceOpen() = %08x", result);
        return false;
    }
    
    return true;
#endif
}

bool Debug::CloseDeviceFunctions() {
#ifdef WIN32
    nvmlReturn_t DeviceResult = nvmlShutdown();
    
    if (NVML_SUCCESS != DeviceResult) {
        Debug::Log(Debug::LogError, L"NVML :: Failed to initialize: %ls", CharToWChar(nvmlErrorString(DeviceResult)));
        return false;
    }
#elif __APPLE__
    kern_return_t result;
    
    result = IOServiceClose(conn);
    
    if (result != kIOReturnSuccess) {
        Debug::Log(Debug::LogError, L"Device Functions error : IOServiceClose() = %08x", result);
        return false;
    }
    
    return true;
#endif
}

float Debug::GetDeviceTemperature(const int & DeviceIndex) {
#ifdef WIN32
    nvmlDevice_t Device;
    nvmlReturn_t DeviceResult;
    unsigned int DeviceTemperature = 0;
    
    DeviceResult = nvmlDeviceGetHandleByIndex(DeviceIndex, &Device);
    if (NVML_SUCCESS != DeviceResult) {
        Debug::Log(Debug::LogError, L"NVML :: Failed to get handle for device %i: %s", DeviceIndex, CharToWChar(nvmlErrorString(DeviceResult)));
        return 0;
    }
    
    DeviceResult = nvmlDeviceGetTemperature(Device, NVML_TEMPERATURE_GPU, &DeviceTemperature);
    if (NVML_SUCCESS != DeviceResult) {
        Debug::Log(Debug::LogError, L"NVML :: Failed to get temperature of device %i: %s", DeviceIndex, CharToWChar(nvmlErrorString(DeviceResult)));
    }
    
    return (float)DeviceTemperature;
    
#elif __APPLE__
    SMCVal_t val;
    kern_return_t result;
    char * DeviceTempKey = (char *)"TG0P";
    
    result = SMCReadKey(DeviceTempKey, &val);
    if (result == kIOReturnSuccess) {
        // --- Read succeeded - check returned value
        if (val.dataSize > 0) {
            if (strcmp(val.dataType, "sp78") == 0) {
                // --- Convert sp78 value to temperature
                int intValue = val.bytes[0] * 256 + (unsigned char)val.bytes[1];
                return intValue / 256.F;
            }
        }
    }
    
    return 0.F;
#endif
}
