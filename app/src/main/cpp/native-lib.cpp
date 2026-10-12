#include <jni.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <vector>
#include <algorithm>

#include "net.h"
#include "gpu.h"

#define TAG "NCNN"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static ncnn::Net g_net;

// ================= JNI LIFECYCLE =================
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    ncnn::create_gpu_instance();
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    ncnn::destroy_gpu_instance();
}

// ================= CONFIG =================
static const int INPUT_SIZE = 640;
static const int NUM_CLASSES = 36;
//static const float CONF_THRESH = 0.01f;

// ================= JNI CACHED =================
static jclass objCls;
static jmethodID ctorId;
static jfieldID xId, yId, wId, hId, probId, labelId;

// ================= LABELS =================
static const char* class_names[] = {
        "0","1","2","3","4","5","6","7","8","9",
        "A","B","C","D","E","F","G","H","I","J",
        "K","L","M","N","O","P","Q","R","S","T",
        "U","V","W","X","Y","Z"
};

struct Detection {
    float x, y, w, h;
    int label;
    float prob;
};

// ================= NMS HELPERS =================
static inline float intersection_area(const Detection& a, const Detection& b) {
    float inter_x = std::max(a.x, b.x);
    float inter_y = std::max(a.y, b.y);
    float inter_w = std::min(a.x + a.w, b.x + b.w) - inter_x;
    float inter_h = std::min(a.y + a.h, b.y + b.h) - inter_y;
    if (inter_w <= 0 || inter_h <= 0) return 0.f;
    return inter_w * inter_h;
}

static void nms_sorted_bboxes(std::vector<Detection>& objects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = objects[i].w * objects[i].h;

    for (int i = 0; i < n; i++) {
        const Detection& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Detection& b = objects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold) {
                keep = 0;
                break;
            }
        }
        if (keep) picked.push_back(i);
    }
}

// ================= INIT =================
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_sign_1learn_1ncnn_YoloV5ncnn_Init(
        JNIEnv* env, jobject thiz, jobject assetManager)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) return JNI_FALSE;

    g_net.opt.lightmode = true;
    g_net.opt.num_threads = 4;

    g_net.opt.use_vulkan_compute = false;

    //if (ncnn::get_gpu_count() > 0)
      //  g_net.opt.use_vulkan_compute = true;

    if (g_net.load_param(mgr, "sign_ncnn.param") != 0) {
        LOGE("load_param failed");
        return JNI_FALSE;
    }

    if (g_net.load_model(mgr, "sign_ncnn.bin") != 0) {
        LOGE("load_model failed");
        return JNI_FALSE;
    }

    // Cache Java Obj class
    jclass local = env->FindClass(
            "com/example/sign_learn_ncnn/YoloV5ncnn$Obj");
    objCls = (jclass) env->NewGlobalRef(local);

    ctorId = env->GetMethodID(objCls, "<init>", "()V");
    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    probId = env->GetFieldID(objCls, "prob", "F");
    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");

    LOGI("Model initialized");
    return JNI_TRUE;
}

// ================= DETECT =================
extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_sign_1learn_1ncnn_YoloV5ncnn_Detect(
        JNIEnv* env, jobject, jobject bitmap, jboolean use_gpu)
{
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    // Convert bitmap → NCNN Mat
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(
            env, bitmap,
            ncnn::Mat::PIXEL_RGB,
            INPUT_SIZE, INPUT_SIZE);

    const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(nullptr, norm);

    ncnn::Extractor ex = g_net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);

    LOGI("out0: w=%d h=%d c=%d", out.w, out.h, out.c);

    std::vector<Detection> objects;

    for(int i=0;i<out.h;i++)
    {
        const float* r = out.row(i);

        float obj = r[4];

        if(obj < 0.05f)
            continue;

        int label = 0;
        float cls_conf = 0.f;

        for(int c=0;c<NUM_CLASSES;c++)
        {
            if(r[5+c] > cls_conf)
            {
                cls_conf = r[5+c];
                label = c;
            }
        }

        float score = obj * cls_conf;

        if(score < 0.05f)
            continue;

        float cx = r[0];
        float cy = r[1];
        float bw = r[2];
        float bh = r[3];

        if(bw <= 0.f)
            continue;

        if(bh <= 0.f)
            continue;

        Detection d;

        d.x = (cx - bw * 0.5f) * info.width / INPUT_SIZE;

        d.y = (cy - bh * 0.5f) * info.height / INPUT_SIZE;

        d.w = bw * info.width / INPUT_SIZE;

        d.h = bh * info.height / INPUT_SIZE;

        d.label = label;

        d.prob = score;

        objects.push_back(d);
    }
    // ===== DECODE (FINAL, CORRECT) =====

    LOGI("Detections before NMS: %zu", objects.size());

    // sort by prob
    std::sort(objects.begin(), objects.end(), [](const Detection& a, const Detection& b) {
        return a.prob > b.prob;
    });

    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, 0.45f);

    int count = picked.size();
    LOGI("Detections after NMS: %d", count);

    jobjectArray arr = env->NewObjectArray(count, objCls, nullptr);
    for (int i = 0; i < count; i++)
    {
        const Detection& d = objects[picked[i]];
        jobject o = env->NewObject(objCls, ctorId);
        env->SetFloatField(o, xId, d.x);
        env->SetFloatField(o, yId, d.y);
        env->SetFloatField(o, wId, d.w);
        env->SetFloatField(o, hId, d.h);
        env->SetFloatField(o, probId, d.prob);
        env->SetObjectField(o, labelId,
                            env->NewStringUTF(class_names[d.label]));
        env->SetObjectArrayElement(arr, i, o);
        env->DeleteLocalRef(o);
    }

    return arr;
}
