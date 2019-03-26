
#include "../../include/Utility/DeviceFunctions.h"

#ifdef __APPLE__
#include <IOKit/IOKitLib.h>
#elif WIN32
#include <nvml.h>
#endif

#ifdef __APPLE__
static io_connect_t IOConnection;

typedef char              SMCBytes_t[32];
typedef char              UInt32Char_t[5];

typedef struct {
    UInt32Char_t          Key;
    UInt32                DataSize;
    UInt32Char_t          DataType;
    SMCBytes_t            Bytes;
} SMCVal_t;

typedef struct {
    char                  Major;
    char                  Minor;
    char                  Build;
    char                  Reserved[1];
    UInt16                Release;
} SMCKeyData_vers_t;

typedef struct {
    UInt16                Version;
    UInt16                Length;
    UInt32                CPUpLimit;
    UInt32                GPUpLimit;
    UInt32                mempLimit;
} SMCKeyData_pLimits_t;

typedef struct {
    UInt32                DataSize;
    UInt32                DataType;
    char                  DataAttributes;
} SMCKeyData_Info_t;

typedef struct {
    UInt32                  Key;
    SMCKeyData_vers_t       Version;
    SMCKeyData_pLimits_t    pLimits;
    SMCKeyData_Info_t       Info;
    char                    Result;
    char                    Status;
    char                    Data8;
    UInt32                  Data32;
    SMCBytes_t              Bytes;
} SMCKeyData_t;

inline kern_return_t SMCCall(int Index, SMCKeyData_t *InputStructure, SMCKeyData_t *OutputStructure) {
    size_t InputSize = sizeof(SMCKeyData_t);
    size_t OutputSize = sizeof(SMCKeyData_t);
    
#if MAC_OS_X_VERSION_10_5
    return IOConnectCallStructMethod( IOConnection, Index,
                                      InputStructure, InputSize,     
                                      OutputStructure, &OutputSize );
#else
    return IOConnectMethodStructureIStructureO( IOConnection, Index,
                                                InputSize, &OutputSize,      
                                                InputStructure, OutputStructure );
#endif
    
}

inline kern_return_t SMCReadKey(UInt32Char_t Key, SMCVal_t *Value) {
    kern_return_t Result;
    SMCKeyData_t  InputStructure;
    SMCKeyData_t  OutputStructure;
    
    memset(&InputStructure, 0, sizeof(SMCKeyData_t));
    memset(&OutputStructure, 0, sizeof(SMCKeyData_t));
    memset(Value, 0, sizeof(SMCVal_t));
    
    UInt32 total = 0;
    for (int i = 0; i < 4; i++) {
        total += Key[i] << (4 - 1 - i) * 8;
    }
    InputStructure.Key = total;
    // --- Read Key info
    InputStructure.Data8 = 9;
    
    Result = SMCCall(2, &InputStructure, &OutputStructure);
    if (Result != kIOReturnSuccess)
        return Result;
    
    Value->DataSize = OutputStructure.Info.DataSize;
    Value->DataType[0] = '\0';
    sprintf(Value->DataType, "%c%c%c%c",
            (unsigned int)OutputStructure.Info.DataType >> 24,
            (unsigned int)OutputStructure.Info.DataType >> 16,
            (unsigned int)OutputStructure.Info.DataType >> 8,
            (unsigned int)OutputStructure.Info.DataType);
    InputStructure.Info.DataSize = Value->DataSize;
    // --- Read data
    InputStructure.Data8 = 5;
    
    Result = SMCCall(2, &InputStructure, &OutputStructure);
    if (Result != kIOReturnSuccess)
        return Result;
    
    memcpy(Value->Bytes, OutputStructure.Bytes, sizeof(OutputStructure.Bytes));
    
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

	return true;
#elif __APPLE__
    kern_return_t Result;
    io_iterator_t Iterator;
    io_object_t   Device;
    
    CFMutableDictionaryRef matchingDictionary = IOServiceMatching("AppleSMC");
    Result = IOServiceGetMatchingServices(kIOMasterPortDefault, matchingDictionary, &Iterator);
    if (Result != kIOReturnSuccess) {
        Debug::Log(Debug::LogError, L"ASMC :: IOServiceGetMatchingServices() = %08x", Result);
        return false;
    }
    
    Device = IOIteratorNext(Iterator);
    IOObjectRelease(Iterator);
    if (Device == 0) {
        Debug::Log(Debug::LogError, L"ASMC :: No SMC Found");
        return false;
    }
    
    Result = IOServiceOpen(Device, mach_task_self(), 0, &IOConnection);
    IOObjectRelease(Device);
    if (Result != kIOReturnSuccess) {
        Debug::Log(Debug::LogError, L"ASMC :: IOServiceOpen() = %08x", Result);
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

	return true;
#elif __APPLE__
    kern_return_t Result;
    
    Result = IOServiceClose(IOConnection);
    
    if (Result != kIOReturnSuccess) {
        Debug::Log(Debug::LogError, L"ASMC :: IOServiceClose() = %08x", Result);
        return false;
    }
    
    return true;
#endif
}

float Debug::GetDeviceTemperature(const int & DeviceIndex) {
#ifdef WIN32
    nvmlDevice_t Device;
    nvmlReturn_t DeviceResult;
    unsigned int DeviceTemperature = 0.F;
    
    DeviceResult = nvmlDeviceGetHandleByIndex(DeviceIndex, &Device);
    if (NVML_SUCCESS != DeviceResult) {
        // Debug::Log(Debug::LogError, L"NVML :: Failed to get handle for device %i: %ls", DeviceIndex, CharToWChar(nvmlErrorString(DeviceResult)));
        return (float)DeviceTemperature;
    }
    
    DeviceResult = nvmlDeviceGetTemperature(Device, NVML_TEMPERATURE_GPU, &DeviceTemperature);
    // if (NVML_SUCCESS != DeviceResult) {
    //     Debug::Log(Debug::LogError, L"NVML :: Failed to get temperature of device %i: %ls", DeviceIndex, CharToWChar(nvmlErrorString(DeviceResult)));
    // }
    
    return (float)DeviceTemperature;
    
#elif __APPLE__
    SMCVal_t Value;
    kern_return_t Result;
    std::string DeviceTempKey = "TG" + std::to_string(DeviceIndex) + "P";
    char * DeviceTempKeyChars = (char *)DeviceTempKey.c_str();
    
    Result = SMCReadKey(DeviceTempKeyChars, &Value);
    if (Result == kIOReturnSuccess) {
        // --- Read succeeded - check returned value
        if (Value.DataSize > 0) {
            if (strcmp(Value.DataType, "sp78") == 0) {
                // --- Convert sp78 value to temperature
                int intValue = Value.Bytes[0] * 256 + (unsigned char)Value.Bytes[1];
                return intValue / 256.F;
            }
        }
    }
    
    return 0.F;
#endif
}
