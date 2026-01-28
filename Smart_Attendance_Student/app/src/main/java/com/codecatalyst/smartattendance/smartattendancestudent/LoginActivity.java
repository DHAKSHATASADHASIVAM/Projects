package com.codecatalyst.smartattendance.smartattendancestudent;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Toast;



import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.codecatalyst.smartattendance.smartattendancestudent.storage.FaceStorage;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.QueryDocumentSnapshot;

public class LoginActivity extends AppCompatActivity {

    private EditText etEmail, etPassword;
    private Button btnLogin;
    private Button btnSignup;
    private ProgressBar progressBar;
    private float[][] storedEmbeddings;
    private FirebaseFirestore db;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_login);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        db = FirebaseFirestore.getInstance();

        etEmail = findViewById(R.id.etEmail);
        etPassword = findViewById(R.id.etPassword);
        btnLogin = findViewById(R.id.btnLogin);
        btnSignup = findViewById(R.id.btnSignup);
        progressBar = findViewById(R.id.progressBar);

        btnLogin.setOnClickListener(v -> attemptLogin());
        btnSignup.setOnClickListener(v -> attemptSignup());
    }

    private void attemptSignup() {
        Intent intent = new Intent(LoginActivity.this, SignupActivity.class);
        startActivity(intent);
        finish();
    }

    private void attemptLogin() {
        String email = etEmail.getText().toString().trim();
        String password = etPassword.getText().toString().trim();

        if (email.isEmpty() || password.isEmpty()) {
            Toast.makeText(this, "Please fill in both fields", Toast.LENGTH_SHORT).show();
            return;
        }
        SharedPreferences enrolledPrefs = getSharedPreferences("EnrolledPrefs", MODE_PRIVATE);
        String enrolledEmail = enrolledPrefs.getString("ENROLLED_STUDENT_EMAIL", null);

        storedEmbeddings = FaceStorage.loadEmbeddings(this);
        if (storedEmbeddings != null && storedEmbeddings.length != 0 &&
                enrolledEmail != null && !enrolledEmail.equals(email)) {
            Toast.makeText(this, "This device is already enrolled with another email, Please contact Admin!", Toast.LENGTH_LONG).show();
            return;
        }
        progressBar.setVisibility(View.VISIBLE);
        btnLogin.setEnabled(false);

        db.collection("student")
                .whereEqualTo("Email", email)
                .whereEqualTo("Password", password)
                .get()
                .addOnCompleteListener(task -> {
                    progressBar.setVisibility(View.GONE);
                    btnLogin.setEnabled(true);

                    if (task.isSuccessful()) {
                        if (!task.getResult().isEmpty()) {
                            for (QueryDocumentSnapshot doc : task.getResult()) {
                                String studentId = doc.getId();
                                String firstName = doc.getString("FirstName");
                                String lastName = doc.getString("LastName");
                                String fullName = firstName + " " + lastName;

                                Toast.makeText(this, "Welcome, " + firstName, Toast.LENGTH_SHORT).show();

                                // ✅ Save student ID persistently
                                getSharedPreferences("StudentPrefs", MODE_PRIVATE)
                                        .edit()
                                        .putString("STUDENT_ID", studentId)
                                        .putString("STUDENT_NAME", fullName)
                                        .putString("STUDENT_EMAIL", email)
                                        .apply();

                                // 🔹 Launch MainActivity
                                Intent intent = new Intent(LoginActivity.this, MainActivity.class);
                                intent.putExtra("STUDENT_ID", studentId);
                                intent.putExtra("STUDENT_NAME", fullName);
                                intent.putExtra("STUDENT_EMAIL", email);
                                startActivity(intent);
                                finish();
                                break;
                            }
                        } else {
                            Toast.makeText(this, "Invalid email or password", Toast.LENGTH_SHORT).show();
                        }
                    } else {
                        Toast.makeText(this, "Error: " + task.getException().getMessage(), Toast.LENGTH_LONG).show();
                    }
                });
    }
}
