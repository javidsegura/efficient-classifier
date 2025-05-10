BOUNDS = {
    # --- Memory (size in KB) --------------------------------
    "Memory_PssClean"          : (0, 150_000),
    "Memory_SharedDirty"       : (0, 200_000),
    "Memory_PrivateDirty"      : (0, 300_000),
    "Memory_SwapPssDirty"      : (0,  50_000),   # swap should be small
    # Heap
    "Memory_HeapAlloc"         : (0, 200_000),
    "Memory_HeapFree"          : (0, 200_000),

    # --- Memory object counts --------------------------------
    "Memory_Views"             : (0,   5_000),
    "Memory_ViewRootImpl"      : (0,     100),
    "Memory_Activities"        : (0,     100),
    "Memory_LocalBinders"      : (0,   1_000),
    "Memory_ParcelCount"       : (0,   2_000),

    # --- API counters (calls per execution trace) ------------
    **{f"API_{name}": (0, 1_000) for name in [
        "Process_android.os.Process_start",
        "Process_android.os.Process_killProcess",
        "Command_java.lang.Runtime_exec",
        "JavaNativeInterface_java.lang.Runtime_loadLibrary",
        "WebView_android.webkit.WebView_loadUrl",
        "Database_android.database.sqlite.SQLiteDatabase_query",
        "DeviceInfo_android.telephony.TelephonyManager_getDeviceId",
        # add the rest as needed
    ]},

    # --- Network ---------------------------------------------
    "Network_TotalReceivedBytes"     : (0, 5_000_000_000), # ~5 GB
    "Network_TotalTransmittedBytes"  : (0, 5_000_000_000),
    "Network_TotalReceivedPackets"   : (0,   2_000_000),
    "Network_TotalTransmittedPackets": (0,   2_000_000),

    # --- Battery & process -----------------------------------
    "Battery_wakelock"          : (0, 10_000),
    "Process_total"             : (0,     500),

    # --- Logcat ----------------------------------------------
    "Logcat_verbose"            : (0, 50_000),
    "Logcat_debug"              : (0, 30_000),
    "Logcat_info"               : (0, 20_000),
    "Logcat_warning"            : (0, 10_000),
    "Logcat_error"              : (0,  5_000),
}