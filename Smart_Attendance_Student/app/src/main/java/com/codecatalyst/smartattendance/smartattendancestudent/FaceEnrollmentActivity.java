package com.codecatalyst.smartattendance.smartattendancestudent;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.graphics.drawable.AnimationDrawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.codecatalyst.smartattendance.smartattendancestudent.ml.BitmapPreprocessor;
import com.codecatalyst.smartattendance.smartattendancestudent.ml.FaceNetModel;
import com.codecatalyst.smartattendance.smartattendancestudent.storage.FaceStorage;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FaceEnrollmentActivity extends AppCompatActivity {

    private static final String TAG = "FaceEnroll";
    private static final int CAMERA_PERMISSION_CODE = 1001;

    // ✅ Configurable number of images to capture
    public static final int NUM_IMAGES_REQUIRED = 5;

    private PreviewView previewView;
    private Button btnCapture;
    private View faceBorderOverlay;

    private String studentEmailId;
    private FaceDetector faceDetector;
    private FaceNetModel faceNetModel;
    private ExecutorService cameraExecutor;

    private boolean faceAligned = false;
    private AnimationDrawable redGlowAnim;
    private AnimationDrawable greenGlowAnim;
    private AnimationDrawable currentAnim;

    private Bitmap lastDetectedFaceBitmap = null; // store last valid frame

    // Store multiple embeddings
    private final List<float[]> capturedEmbeddings = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face_enrollment);

        previewView = findViewById(R.id.previewView);
        btnCapture = findViewById(R.id.btnCapture);
        faceBorderOverlay = findViewById(R.id.faceBorderOverlay);

        faceBorderOverlay.setBackgroundResource(R.drawable.anim_red_glow);
        redGlowAnim = (AnimationDrawable) faceBorderOverlay.getBackground();
        redGlowAnim.start();
        currentAnim = redGlowAnim;

        faceNetModel = new FaceNetModel(this);

        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .build();

        faceDetector = FaceDetection.getClient(options);
        cameraExecutor = Executors.newSingleThreadExecutor();

        studentEmailId = getIntent().getStringExtra("STUDENT_EMAIL");

        // Initial button text with remaining count
        updateCaptureButtonText();

        // Show instruction dialog once
        showEnrollmentInstructions();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_CODE
            );
        } else {
            startCamera();
        }

        btnCapture.setOnClickListener(v -> captureFace());
    }

    private void showEnrollmentInstructions() {
        new AlertDialog.Builder(this)
                .setTitle("Face Enrollment")
                .setMessage(
                        "For better accuracy, we will capture "
                                + NUM_IMAGES_REQUIRED
                                + " images of your face.\n\n" +
                                "Align your face inside the circle until it turns green, " +
                                "then press the Capture button. Repeat until all images are captured."
                )
                .setPositiveButton("OK", null)
                .show();
    }

    private void updateCaptureButtonText() {
        int captured = capturedEmbeddings.size();
        int remaining = NUM_IMAGES_REQUIRED - captured;
        if (remaining < 0) remaining = 0;
        btnCapture.setText("Capture (" + remaining + " left)");
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;

                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageAnalysis);

                Log.d(TAG, "✅ Camera started with live analysis.");

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "❌ Camera init error: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    private void analyzeFrame(@NonNull ImageProxy imageProxy) {
        android.media.Image mediaImage = imageProxy.getImage();
        if (mediaImage == null) {
            imageProxy.close();
            return;
        }

        InputImage image = InputImage.fromMediaImage(
                mediaImage,
                imageProxy.getImageInfo().getRotationDegrees()
        );

        faceDetector.process(image)
                .addOnSuccessListener(faces -> {
                    if (faces != null && !faces.isEmpty()) {
                        Face face = faces.get(0);
                        Rect bounds = face.getBoundingBox();

                        Bitmap frameBitmap = imageProxyToBitmap(imageProxy);
                        if (frameBitmap == null) {
                            lastDetectedFaceBitmap = null;
                            return;
                        }
                        Bitmap mirrored = mirrorBitmap(frameBitmap); // mirror front camera
                        Bitmap cropped = cropFace(mirrored, bounds);

                        lastDetectedFaceBitmap = cropped; // store valid frame

                        if (!faceAligned) runOnUiThread(this::showGreenGlow);
                        faceAligned = true;
                    } else {
                        if (faceAligned) runOnUiThread(this::showRedGlow);
                        faceAligned = false;
                        lastDetectedFaceBitmap = null;
                    }
                })
                .addOnFailureListener(e ->
                        Log.e(TAG, "Face detection error: " + e.getMessage()))
                .addOnCompleteListener(t -> imageProxy.close());
    }

    private void showRedGlow() {
        if (currentAnim == redGlowAnim) return;
        faceBorderOverlay.setBackgroundResource(R.drawable.circle_border);
        redGlowAnim = (AnimationDrawable) faceBorderOverlay.getBackground();
        redGlowAnim.start();
        currentAnim = redGlowAnim;
    }

    private void showGreenGlow() {
        if (currentAnim == greenGlowAnim) return;
        faceBorderOverlay.setBackgroundResource(R.drawable.anim_green_glow);
        greenGlowAnim = (AnimationDrawable) faceBorderOverlay.getBackground();
        greenGlowAnim.start();
        currentAnim = greenGlowAnim;
    }

    private void captureFace() {
        if (lastDetectedFaceBitmap == null) {
            Toast.makeText(this,
                    "Align your face properly (green glow) before capturing.",
                    Toast.LENGTH_SHORT).show();
            return;
        }

        if (capturedEmbeddings.size() >= NUM_IMAGES_REQUIRED) {
            // Just in case user taps extra
            Toast.makeText(this,
                    "Already captured all required images.",
                    Toast.LENGTH_SHORT).show();
            return;
        }

        // Use the last valid frame to compute embedding
        float[] input = BitmapPreprocessor.preprocess(lastDetectedFaceBitmap);
        float[] embedding = faceNetModel.getEmbedding(input);

        capturedEmbeddings.add(embedding);

        int captured = capturedEmbeddings.size();
        int remaining = NUM_IMAGES_REQUIRED - captured;

        if (remaining > 0) {
            Toast.makeText(
                    this,
                    "Captured " + captured + " of " + NUM_IMAGES_REQUIRED,
                    Toast.LENGTH_SHORT
            ).show();
            updateCaptureButtonText();
        } else {
            // All images captured – save them
            float[][] allEmbeddings = new float[capturedEmbeddings.size()][];
            for (int i = 0; i < capturedEmbeddings.size(); i++) {
                allEmbeddings[i] = capturedEmbeddings.get(i);
            }

            FaceStorage.saveEmbeddings(this, allEmbeddings);
            Toast.makeText(this,
                    "✅ Face registered successfully!",
                    Toast.LENGTH_LONG).show();
            getSharedPreferences("EnrolledPrefs", MODE_PRIVATE)
                    .edit()
                    .putString("ENROLLED_STUDENT_EMAIL", studentEmailId)
                    .apply();
            Intent intent = new Intent(FaceEnrollmentActivity.this, MainActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
            finish();
        }
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    private Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
        android.media.Image image = imageProxy.getImage();

        if (image == null) return null;

        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();

        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

        byte[] y = new byte[yBuffer.remaining()];
        byte[] u = new byte[uBuffer.remaining()];
        byte[] v = new byte[vBuffer.remaining()];

        yBuffer.get(y);
        uBuffer.get(u);
        vBuffer.get(v);

        // NV21 format for ML Kit compatibility
        byte[] nv21 = new byte[y.length + u.length + v.length];

        // Copy Y
        System.arraycopy(y, 0, nv21, 0, y.length);

        // Copy VU (swap order)
        for (int i = 0; i < u.length; i += 2) {
            nv21[y.length + i] = v[i];
            nv21[y.length + i + 1] = u[i];
        }

        YuvImage yuvImage = new YuvImage(
                nv21,
                android.graphics.ImageFormat.NV21,
                width,
                height,
                null
        );
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, width, height), 90, out);
        byte[] jpegBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);
    }

    private Bitmap mirrorBitmap(Bitmap original) {
        Matrix matrix = new Matrix();
        matrix.preScale(-1.0f, 1.0f);
        return Bitmap.createBitmap(
                original,
                0, 0,
                original.getWidth(),
                original.getHeight(),
                matrix,
                true
        );
    }

    private Bitmap cropFace(Bitmap src, Rect rect) {
        int left = Math.max(rect.left, 0);
        int top = Math.max(rect.top, 0);
        int width = Math.min(rect.width(), src.getWidth() - left);
        int height = Math.min(rect.height(), src.getHeight() - top);
        try {
            return Bitmap.createBitmap(src, left, top, width, height);
        } catch (Exception e) {
            Log.e(TAG, "Crop failed, using full image: " + e.getMessage());
            return src;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 &&
                    grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this,
                        "Camera permission required",
                        Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) cameraExecutor.shutdown();
    }
}
