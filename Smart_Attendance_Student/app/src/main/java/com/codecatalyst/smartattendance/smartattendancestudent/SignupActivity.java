package com.codecatalyst.smartattendance.smartattendancestudent;
import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FieldValue;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.QueryDocumentSnapshot;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SignupActivity extends AppCompatActivity {
    private Spinner spinnerEmail;
    private String selectedEmail;
    private EditText etCurrentPassword;
    private EditText etFirstName;
    private EditText etLastName;
    private Spinner spinnerCourse;
    private Button btnSignup;
    private ProgressBar progressBar;
    private FirebaseFirestore db;
    private final ArrayList<String> courseList = new ArrayList<>();

    private final ArrayList<String> EmailList = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_signup);

        // Initialize views
        spinnerEmail = findViewById(R.id.spinnerEmail);
        etCurrentPassword = findViewById(R.id.etCurrentPassword);
        etFirstName = findViewById(R.id.etFirstName);
        etLastName = findViewById(R.id.etLastName);
        spinnerCourse = findViewById(R.id.spinnerCourse);
        btnSignup = findViewById(R.id.btnSignup);
        progressBar = findViewById(R.id.progressBar);

        var btnBack = findViewById(R.id.btnBack);


        // Initialize Firestore
        db = FirebaseFirestore.getInstance();

        setupTextWatchers();

        // Fetch courses from Firestore
        fetchEmailFromFirestore();

        // Signup button click
        btnSignup.setOnClickListener(v -> attemptSignup());
        btnBack.setOnClickListener(v -> goback()); // closes the activity

        spinnerEmail.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                // Avoid triggering for the default hint item if you have one
                if (position == 0) return;

                // Get the selected email
                selectedEmail = parent.getItemAtPosition(position).toString();

                // Call your function here
                fetchStudentDetails(selectedEmail);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // Optional: do something when nothing is selected
            }
        });
    }

    private void fetchStudentDetails(String email) {
        progressBar.setVisibility(View.VISIBLE);

        db.collection("student")
                .whereEqualTo("Email", email)
                .get()
                .addOnSuccessListener(querySnapshot -> {
                    progressBar.setVisibility(View.GONE);

                    if (!querySnapshot.isEmpty()) {
                        DocumentSnapshot doc = querySnapshot.getDocuments().get(0);
                        String studentId = doc.getId();
                        String firstName = doc.getString("FirstName");
                        String lastName = doc.getString("LastName");

                        etFirstName.setText(firstName);
                        etLastName.setText(lastName);

                        // Example: save locally or use directly
                        Log.d("StudentInfo", "ID: " + studentId + ", Name: " + firstName + " " + lastName);

                        // You can now use this info to fetch scheduled courses for this student
                        fetchScheduledCoursesForStudent(studentId);

                    } else {
                        Toast.makeText(this, "No student found with this email", Toast.LENGTH_SHORT).show();
                    }
                })
                .addOnFailureListener(e -> {
                    progressBar.setVisibility(View.GONE);
                    Toast.makeText(this, "Error fetching student: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                });
    }

    private void fetchScheduledCoursesForStudent(String studentId) {
        progressBar.setVisibility(View.VISIBLE);

        db.collection("Courses")
                .get()
                .addOnCompleteListener(task -> {
                    progressBar.setVisibility(View.GONE);

                    if (task.isSuccessful()) {
                        courseList.clear();
                        courseList.add("Enrolled Courses"); // default option

                        for (QueryDocumentSnapshot courseDoc : task.getResult()) {
                            String courseId = courseDoc.getId();
                            String courseName = courseDoc.getString("CourseName");

                            // Go to schedule subcollection
                            db.collection("Courses")
                                    .document(courseId)
                                    .collection("Schedule")
                                    .get()
                                    .addOnSuccessListener(scheduleQuery -> {
                                        for (QueryDocumentSnapshot scheduleDoc : scheduleQuery) {
                                            List<String> enrolledList = (List<String>) scheduleDoc.get("StudentsEnrolled");

                                            if (enrolledList != null && enrolledList.contains(studentId)) {
                                                // Add course if not already added
                                                if (courseName != null && !courseList.contains(courseName)) {
                                                    courseList.add(courseName);
                                                }
                                            }
                                        }

                                        // Update spinner adapter
                                        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                                                this,
                                                android.R.layout.simple_spinner_dropdown_item,
                                                courseList
                                        );
                                        spinnerCourse.setAdapter(adapter);
                                    });
                        }

                    } else {
                        Toast.makeText(this,
                                "Failed to fetch courses: " + task.getException().getMessage(),
                                Toast.LENGTH_SHORT).show();
                    }
                });
    }



    private void fetchEmailFromFirestore() {
        progressBar.setVisibility(View.VISIBLE);

        db.collection("student")
                .get()
                .addOnCompleteListener(task -> {
                    progressBar.setVisibility(View.GONE);
                    if (task.isSuccessful()) {
                        EmailList.clear();
                        EmailList.add("Select Email"); // default option
                        for (QueryDocumentSnapshot doc : task.getResult()) {
                            String Email = doc.getString("Email");
                            if (Email != null) {
                                EmailList.add(Email);
                            }
                        }
                        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                                android.R.layout.simple_spinner_dropdown_item, EmailList);
                        spinnerEmail.setAdapter(adapter);
                    } else {
                        Toast.makeText(this, "Failed to fetch courses: " +
                                task.getException().getMessage(), Toast.LENGTH_SHORT).show();
                    }
                });
    }

    // Handle signup logic
    private void attemptSignup() {
            String password = etCurrentPassword.getText().toString().trim();

            progressBar.setVisibility(View.VISIBLE);

            // Fetch student document by ID
        db.collection("student")
                .whereEqualTo("Email", selectedEmail) // query by email field
                .get()
                .addOnSuccessListener(querySnapshot -> {
                    progressBar.setVisibility(View.GONE);

                    if (!querySnapshot.isEmpty()) {
                        DocumentSnapshot doc = querySnapshot.getDocuments().get(0);

                        // Set or overwrite the Password field
                        doc.getReference().update("Password", password)
                                .addOnSuccessListener(aVoid -> {
                                    Toast.makeText(this, "Password saved successfully!", Toast.LENGTH_SHORT).show();
                                    Intent intent = new Intent(SignupActivity.this, LoginActivity.class);
                                    startActivity(intent);
                                    finish();
                                })
                                .addOnFailureListener(e ->
                                        Toast.makeText(this, "Failed to save password: " + e.getMessage(), Toast.LENGTH_SHORT).show()
                                );

                    } else {
                        Toast.makeText(this, "Student not found", Toast.LENGTH_SHORT).show();
                    }
                })
                .addOnFailureListener(e -> {
                    progressBar.setVisibility(View.GONE);
                    Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                });


    }


    private void setupTextWatchers() {
        TextWatcher watcher = new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) { checkAllFields(); }
            @Override public void afterTextChanged(Editable s) {}
        };

        etCurrentPassword.addTextChangedListener(watcher);

        spinnerEmail.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                checkAllFields();
                if (position > 0) fetchStudentDetails(parent.getItemAtPosition(position).toString());
            }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });

        spinnerCourse.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) { checkAllFields(); }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });
    }

    private void checkAllFields() {
        boolean enabled = spinnerEmail.getSelectedItemPosition() > 0 &&
                !etFirstName.getText().toString().isEmpty() &&
                !etLastName.getText().toString().isEmpty() &&
                !etCurrentPassword.getText().toString().isEmpty();
        btnSignup.setEnabled(enabled);
    }

    private void goback(){
        Intent intent = new Intent(SignupActivity.this, LoginActivity.class);
        startActivity(intent);
        finish();
    }


}
