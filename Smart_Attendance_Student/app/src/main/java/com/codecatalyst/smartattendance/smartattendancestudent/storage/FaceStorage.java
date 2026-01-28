package com.codecatalyst.smartattendance.smartattendancestudent.storage;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Base64;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class FaceStorage {

    private static final String PREF_NAME = "face_embeddings";

    // We now store multiple embeddings
    private static final String KEY_EMBEDDING_PREFIX = "embedding_vector_";
    private static final String KEY_EMBEDDING_COUNT  = "embedding_count";

    private static final String TAG = "FaceStorage";

    // Save multiple embeddings
    public static void saveEmbeddings(Context context, float[][] embeddings) {
        if (embeddings == null || embeddings.length == 0) {
            Log.w(TAG, "saveEmbeddings called with empty array");
            return;
        }

        SharedPreferences prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();

        int count = embeddings.length;
        editor.putInt(KEY_EMBEDDING_COUNT, count);

        for (int i = 0; i < count; i++) {
            float[] emb = embeddings[i];
            if (emb == null) continue;
            String base64 = floatArrayToBase64(emb);
            editor.putString(KEY_EMBEDDING_PREFIX + i, base64);
        }

        editor.apply();
        Log.d(TAG, "✅ Saved " + count + " face embeddings");
    }

    // Load all stored embeddings
    public static float[][] loadEmbeddings(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
        int count = prefs.getInt(KEY_EMBEDDING_COUNT, 0);
        if (count <= 0) {
            Log.d(TAG, "No embeddings found");
            return null;
        }

        float[][] result = new float[count][];
        for (int i = 0; i < count; i++) {
            String base64 = prefs.getString(KEY_EMBEDDING_PREFIX + i, null);
            if (base64 == null) {
                Log.w(TAG, "Missing embedding at index " + i);
                result[i] = null;
            } else {
                result[i] = base64ToFloatArray(base64);
            }
        }

        Log.d(TAG, "✅ Loaded " + count + " embeddings");
        return result;
    }

    public static boolean hasEmbedding(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
        int count = prefs.getInt(KEY_EMBEDDING_COUNT, 0);
        return count > 0;
    }

    public static void clearEmbedding(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
        prefs.edit().clear().apply();
        Log.d(TAG, "🗑️ All face embeddings cleared");
    }

    // --- helpers ---

    private static String floatArrayToBase64(float[] embedding) {
        ByteBuffer buffer = ByteBuffer.allocate(embedding.length * 4);
        buffer.order(ByteOrder.nativeOrder());
        for (float v : embedding) buffer.putFloat(v);
        return Base64.encodeToString(buffer.array(), Base64.DEFAULT);
    }

    private static float[] base64ToFloatArray(String base64) {
        byte[] bytes = Base64.decode(base64, Base64.DEFAULT);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        float[] embedding = new float[bytes.length / 4];
        for (int i = 0; i < embedding.length; i++) {
            embedding[i] = buffer.getFloat();
        }
        return embedding;
    }
}
