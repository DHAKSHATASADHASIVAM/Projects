package com.codecatalyst.smartattendance.smartattendancestudent.ml;

import android.graphics.Bitmap;

public class BitmapPreprocessor {

    private static final int INPUT_SIZE = 160;  // 160x160x3

    public static float[] preprocess(Bitmap bitmap) {
        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        float[] input = new float[INPUT_SIZE * INPUT_SIZE * 3];
        int idx = 0;

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];

            // Normalize to [0,1] — this is fine as long as we use the same for enroll & verify
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;

            input[idx++] = r;
            input[idx++] = g;
            input[idx++] = b;
        }

        return input;
    }
}
