# 📱 AI Monk Attendance - Native Android (Kotlin) Client

High-performance native Android Edge AI client for continuous real-time face detection, IoU trajectory tracking, and GPU recognition.

---

## ⚡ Performance Highlights
* **$1.5\text{–}3.0\text{ ms}$ Face Detection**: Powered by **Microsoft ONNX Runtime Mobile** with native **Android NNAPI** hardware acceleration (Qualcomm Hexagon / MediaTek NPU / Samsung NPU).
* **Zero-Copy Camera Pipeline**: Built on **AndroidX CameraX** with direct hardware buffer memory mapping.
* **Locked 60–120 FPS Fluid Rendering**: Custom hardware-accelerated canvas overlay with velocity dead-reckoning motion interpolation.
* **Micro-Payload Cloud Dispatches**: Sends throttled $224 \times 224$ face crops to the GPU server (`https://49.206.228.75:9001`) via persistent OkHttp3 HTTP/2 connection pooling.

---

## 📂 Project Structure
```text
android_client/
├── app/
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/
│       │   └── version-RFB-320.onnx           <-- 1.1MB Ultra-Light ONNX model
│       ├── java/com/aimonk/attendance/
│       │   ├── MainActivity.kt               <-- CameraX Lifecycle & Main Orchestrator
│       │   ├── engine/
│       │   │   ├── UltraLightDetector.kt     <-- ONNX Runtime Mobile + NNAPI Acceleration
│       │   │   ├── IoUTracker.kt             <-- Multi-Face Tracking & Dead-Reckoning
│       │   │   ├── CropDispatcher.kt         <-- 120px Gate & Async GPU Cloud Dispatcher
│       │   │   └── OverlayView.kt            <-- 60 FPS Custom Canvas Drawing Engine
│       │   ├── network/
│       │   │   └── ApiService.kt             <-- OkHttp3 Fast Binary Client
│       │   └── model/
│       │       └── AttendanceModels.kt
│       └── res/layout/
│           └── activity_main.xml
├── build.gradle.kts
└── settings.gradle.kts
```

---

## 🚀 How to Build & Run

### Method 1: In Android Studio (Recommended)
1. Launch **Android Studio**.
2. Click **Open** and select the [`/h3/anas/ABLBL_Attendance/android_client`](file:///h3/anas/ABLBL_Attendance/android_client) folder.
3. Wait for Gradle Sync to complete.
4. Connect an Android phone / tablet via USB (or start an Android Virtual Device).
5. Click **Run (`Shift + F10`)**.

### Method 2: Build APK via Terminal / Gradle CLI
```bash
cd /h3/anas/ABLBL_Attendance/android_client
./gradlew assembleDebug
```
The output APK will be generated at:
`app/build/outputs/apk/debug/app-debug.apk`

---

## ⚙️ Server Configuration
To point the app to a different GPU server, update the `baseUrl` in [`MainActivity.kt`](file:///h3/anas/ABLBL_Attendance/android_client/app/src/main/java/com/aimonk/attendance/MainActivity.kt):
```kotlin
val apiService = ApiService("https://YOUR_SERVER_IP:9001")
```
