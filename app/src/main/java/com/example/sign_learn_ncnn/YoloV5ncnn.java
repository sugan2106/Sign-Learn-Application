package com.example.sign_learn_ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class YoloV5ncnn {

    static {
        System.loadLibrary("yolov5ncnn");
    }

    public native boolean Init(AssetManager mgr);
    public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);

    public static class Obj {
        public float x;
        public float y;
        public float w;
        public float h;
        public float prob;
        public String label;
    }
}
