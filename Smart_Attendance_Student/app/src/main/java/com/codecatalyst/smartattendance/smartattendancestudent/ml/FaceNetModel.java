package com.codecatalyst.smartattendance.smartattendancestudent.ml;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class FaceNetModel {

    private static final String TAG = "FaceNetModel";
    private static final int EMBEDDING_SIZE = 512; // 🔴 IMPORTANT: matches your facenet.tflite output [1, 512]

    private Interpreter interpreter;

    public FaceNetModel(Context context) {
        try {
            MappedByteBuffer modelBuffer = loadModelFile(context, "facenet.tflite");
            interpreter = new Interpreter(modelBuffer);
            Log.d(TAG, "✅ FaceNet TFLite model loaded");
        } catch (IOException e) {
            Log.e(TAG, "❌ Error loading model: " + e.getMessage());
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public float[] getEmbedding(float[] input) {
        // input expected shape: [1, 160, 160, 3] → we flatten and pass as buffer
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(input.length * 4);
        inputBuffer.order(ByteOrder.nativeOrder());
        for (float v : input) {
            inputBuffer.putFloat(v);
        }

        float[][] output = new float[1][EMBEDDING_SIZE];
        interpreter.run(inputBuffer, output);
        return output[0];
    }
}
