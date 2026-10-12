package com.example.sign_learn_ncnn

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private val detector = YoloV5ncnn()
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: ImageView
    private val executor = Executors.newSingleThreadExecutor()

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.GREEN
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("DEBUG", "=== onCreate started ===")
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.viewFinder)
        overlayView = findViewById(R.id.overlayView)
        Log.d("DEBUG", "Views initialized")

        if (!detector.Init(assets)) {
            Log.e("DEBUG", "Model init FAILED")
            Toast.makeText(this, "Model load failed", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        Log.d("DEBUG", "Model loaded successfully!")

        if (hasPermission()) {
            Log.d("DEBUG", "Permission already granted, starting camera")
            startCamera()
        } else {
            Log.d("DEBUG", "Requesting camera permission")
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), 10
            )
        }
    }

    private fun startCamera() {
        Log.d("DEBUG", "=== startCamera() called ===")

        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                Log.d("DEBUG", "Camera provider listener triggered")
                val provider = providerFuture.get()

                val preview = Preview.Builder().build()
                preview.setSurfaceProvider(previewView.surfaceProvider)
                Log.d("DEBUG", "Preview surface provider set")

                val analysis = ImageAnalysis.Builder()
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                analysis.setAnalyzer(executor) { image ->
                    try {
                        val originalBmp = image.toBitmap()
                        val matrix = Matrix()
                        matrix.postRotate(image.imageInfo.rotationDegrees.toFloat())
                        
                        val bmp = Bitmap.createBitmap(
                            originalBmp, 0, 0,
                            originalBmp.width, originalBmp.height,
                            matrix, true
                        )

                        val objs = detector.Detect(bmp, true)

                        runOnUiThread {
                            draw(objs, bmp.width, bmp.height)
                        }
                    } catch (e: Exception) {
                        Log.e("DEBUG", "Error during frame analysis", e)
                    } finally {
                        image.close()
                    }
                }
                Log.d("DEBUG", "Analyzer configured")

                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                )
                Log.d("DEBUG", "*** Camera bound to lifecycle SUCCESS ***")

            } catch (e: Exception) {
                Log.e("DEBUG", "*** Camera setup FAILED ***", e)
                runOnUiThread {
                    Toast.makeText(this, "Camera error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun draw(objects: Array<YoloV5ncnn.Obj>?, w: Int, h: Int) {
        Log.d("DEBUG", "draw() called with ${objects?.size ?: 0} objects")
        if (objects == null) return

        val bmp = Bitmap.createBitmap(
            previewView.width,
            previewView.height,
            Bitmap.Config.ARGB_8888
        )
        val canvas = Canvas(bmp)

        val sx = previewView.width.toFloat() / w
        val sy = previewView.height.toFloat() / h
        Log.d("DEBUG", "Scale factors: sx=$sx, sy=$sy")

        for (o in objects) {
            val l = o.x * sx
            val t = o.y * sy
            val r = (o.x + o.w) * sx
            val b = (o.y + o.h) * sy
            Log.d("DEBUG", "Drawing box: ${o.label} at ($l, $t, $r, $b)")
            canvas.drawRect(l, t, r, b, boxPaint)
            canvas.drawText(o.label, l, t - 8, textPaint)
        }

        overlayView.setImageBitmap(bmp)
        Log.d("DEBUG", "Overlay bitmap set")
    }

    // old imageToBitmap logic is fully replaced by image.toBitmap()

    private fun hasPermission(): Boolean =
        ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        Log.d("DEBUG", "Permission result received: ${grantResults.firstOrNull()}")

        if (requestCode == 10 && grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.d("DEBUG", "Permission GRANTED - starting camera")
            startCamera()
        } else {
            Log.e("DEBUG", "Permission DENIED")
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
        }
    }
}
