package com.codecatalyst.smartattendance.smartattendancestudent;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.firebase.Timestamp;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FirebaseFirestore;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class AttendanceHistoryActivity extends AppCompatActivity {

    private static final String TAG = "AttendanceHistory";

    private FirebaseFirestore db;
    private String studentId;

    // UI
    private ImageButton btnBackHistory;
    private Spinner spSemester, spCourse;
    private LinearLayout cardCourse, cardHistory;
    private RecyclerView rvHistory;
    private ProgressBar progressHistory;
    private TextView tvNoData, tvHistoryHeader;
    private LinearLayout tableHeader;     // header row: Date | Attendance Status

    // Semester data
    private final List<String> semesterNames = new ArrayList<>();
    private final List<String> semesterIds = new ArrayList<>();

    // Course data
    private final List<String> courseNames = new ArrayList<>();
    private final List<String> courseIds = new ArrayList<>();
    // courseId -> list of scheduleIds in that semester where student is enrolled
    private final Map<String, List<String>> courseSchedulesByCourseId = new HashMap<>();

    // History data
    private final List<AttendanceRow> historyRows = new ArrayList<>();
    private HistoryAdapter historyAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_attendance_history);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        db = FirebaseFirestore.getInstance();

        // Get studentId from intent or shared prefs
        Intent intent = getIntent();
        studentId = intent.getStringExtra("STUDENT_ID");
        if (studentId == null || studentId.isEmpty()) {
            studentId = getSharedPreferences("StudentPrefs", MODE_PRIVATE)
                    .getString("STUDENT_ID", null);
        }

        // Bind views
        btnBackHistory = findViewById(R.id.btnBackHistory);
        spSemester = findViewById(R.id.spSemester);
        spCourse = findViewById(R.id.spCourse);
        cardCourse = findViewById(R.id.cardCourse);
        cardHistory = findViewById(R.id.cardHistory);
        rvHistory = findViewById(R.id.rvAttendanceHistory);
        progressHistory = findViewById(R.id.progressHistory);
        tvNoData = findViewById(R.id.tvNoData);
        tvHistoryHeader = findViewById(R.id.tvHistoryHeader);
        tableHeader = findViewById(R.id.tableHeader);

        cardCourse.setVisibility(View.GONE);
        cardHistory.setVisibility(View.GONE);
        tableHeader.setVisibility(View.GONE);

        btnBackHistory.setOnClickListener(v -> finish());

        // RecyclerView
        rvHistory.setLayoutManager(new LinearLayoutManager(this));
        historyAdapter = new HistoryAdapter(historyRows);
        rvHistory.setAdapter(historyAdapter);

        setupSemesterSpinner();
    }

    // ----------------------------------------------------
    // SEMESTER SPINNER
    // ----------------------------------------------------
    private void setupSemesterSpinner() {
        semesterNames.clear();
        semesterIds.clear();

        semesterNames.add("Select semester");
        semesterIds.add(null);

        db.collection("Semester")
                .get()
                .addOnSuccessListener(snap -> {
                    for (DocumentSnapshot doc : snap.getDocuments()) {
                        String id = doc.getId();
                        String name = doc.getString("Name");
                        if (name == null || name.isEmpty()) name = id;
                        semesterIds.add(id);
                        semesterNames.add(name);
                    }

                    ArrayAdapter<String> adapter = new ArrayAdapter<>(
                            this,
                            android.R.layout.simple_spinner_item,
                            semesterNames
                    );
                    adapter.setDropDownViewResource(
                            android.R.layout.simple_spinner_dropdown_item);
                    spSemester.setAdapter(adapter);
                })
                .addOnFailureListener(e ->
                        Log.e(TAG, "Failed to load semesters: " + e.getMessage()));

        spSemester.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent,
                                       View view,
                                       int position,
                                       long id) {

                if (position == 0) {
                    cardCourse.setVisibility(View.GONE);
                    cardHistory.setVisibility(View.GONE);
                    tableHeader.setVisibility(View.GONE);
                    tvNoData.setVisibility(View.GONE);
                    historyRows.clear();
                    historyAdapter.notifyDataSetChanged();
                    return;
                }

                String semesterId = semesterIds.get(position);
                loadCoursesForSemester(semesterId);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // no-op
            }
        });
    }

    // ----------------------------------------------------
    // LOAD COURSES FOR STUDENT & SEMESTER
    // ----------------------------------------------------
    private void loadCoursesForSemester(String semesterId) {
        cardCourse.setVisibility(View.GONE);
        cardHistory.setVisibility(View.GONE);
        tableHeader.setVisibility(View.GONE);
        tvNoData.setVisibility(View.GONE);

        courseNames.clear();
        courseIds.clear();
        courseSchedulesByCourseId.clear();

        courseNames.add("Select course");
        courseIds.add(null);

        db.collection("Courses")
                .get()
                .addOnSuccessListener(coursesSnap -> {
                    if (coursesSnap.isEmpty()) {
                        updateCourseSpinner();
                        return;
                    }

                    for (DocumentSnapshot courseDoc : coursesSnap.getDocuments()) {
                        String courseId = courseDoc.getId();
                        String courseName = courseDoc.getString("CourseName");
                        if (courseName == null || courseName.isEmpty()) {
                            courseName = courseId;
                        }

                        String finalCourseName = courseName;
                        courseDoc.getReference()
                                .collection("Schedule")
                                .whereEqualTo("Semester", semesterId)
                                .whereArrayContains("StudentsEnrolled", studentId)
                                .get()
                                .addOnSuccessListener(scheduleSnap -> {
                                    if (!scheduleSnap.isEmpty()) {

                                        if (!courseIds.contains(courseId)) {
                                            courseIds.add(courseId);
                                            courseNames.add(finalCourseName);
                                        }

                                        List<String> schedList =
                                                courseSchedulesByCourseId.get(courseId);
                                        if (schedList == null) {
                                            schedList = new ArrayList<>();
                                            courseSchedulesByCourseId.put(courseId, schedList);
                                        }

                                        for (DocumentSnapshot sDoc : scheduleSnap) {
                                            String schedId = sDoc.getId();
                                            if (!schedList.contains(schedId)) {
                                                schedList.add(schedId);
                                            }
                                        }
                                    }
                                    updateCourseSpinner();
                                })
                                .addOnFailureListener(e ->
                                        Log.e(TAG,
                                                "Schedule query failed: " + e.getMessage()));
                    }
                })
                .addOnFailureListener(e ->
                        Log.e(TAG, "Courses query failed: " + e.getMessage()));
    }

    private void updateCourseSpinner() {
        if (courseIds.size() <= 1) {
            // Only placeholder
            cardCourse.setVisibility(View.GONE);
            cardHistory.setVisibility(View.GONE);
            tableHeader.setVisibility(View.GONE);
            return;
        }

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_item,
                courseNames
        );
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spCourse.setAdapter(adapter);
        cardCourse.setVisibility(View.VISIBLE);

        spCourse.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent,
                                       View view,
                                       int position,
                                       long id) {
                if (position == 0) {
                    cardHistory.setVisibility(View.GONE);
                    tableHeader.setVisibility(View.GONE);
                    tvNoData.setVisibility(View.GONE);
                    historyRows.clear();
                    historyAdapter.notifyDataSetChanged();
                    return;
                }

                String courseId = courseIds.get(position);
                List<String> scheduleIds = courseSchedulesByCourseId.get(courseId);
                if (scheduleIds == null || scheduleIds.isEmpty()) {
                    cardHistory.setVisibility(View.GONE);
                    tableHeader.setVisibility(View.GONE);
                    tvNoData.setVisibility(View.VISIBLE);
                    historyRows.clear();
                    historyAdapter.notifyDataSetChanged();
                    return;
                }

                String courseName = courseNames.get(position);
                tvHistoryHeader.setText("Attendance – " + courseName);

                loadAttendanceHistory(courseId, scheduleIds);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // no-op
            }
        });
    }

    // ----------------------------------------------------
    // LOAD ATTENDANCE HISTORY
    // ----------------------------------------------------
    private void loadAttendanceHistory(String courseId, List<String> scheduleIds) {
        cardHistory.setVisibility(View.VISIBLE);
        progressHistory.setVisibility(View.VISIBLE);
        tableHeader.setVisibility(View.GONE);
        tvNoData.setVisibility(View.GONE);
        historyRows.clear();
        historyAdapter.notifyDataSetChanged();

        final int[] pending = {scheduleIds.size()};

        for (String schedId : scheduleIds) {
            db.collection("Courses")
                    .document(courseId)
                    .collection("Schedule")
                    .document(schedId)
                    .collection("Attendance")
                    .get()
                    .addOnSuccessListener(attSnap -> {
                        for (DocumentSnapshot dateDoc : attSnap.getDocuments()) {
                            String date = dateDoc.getId(); // e.g., "2025-11-15"
                            Map<String, Object> sessionsMap = dateDoc.getData();
                            if (sessionsMap == null) continue;

                            for (Map.Entry<String, Object> entry : sessionsMap.entrySet()) {
                                if (!(entry.getValue() instanceof Map)) continue;

                                Map<String, Object> sessionData =
                                        (Map<String, Object>) entry.getValue();

                                Map<String, Object> studentMap = null;
                                Object saObj = sessionData.get("StudentAttendanceData");
                                if (saObj instanceof Map) {
                                    Map<String, Object> allStudents =
                                            (Map<String, Object>) saObj;
                                    Object stuObj = allStudents.get(studentId);
                                    if (stuObj instanceof Map) {
                                        studentMap = (Map<String, Object>) stuObj;
                                    }
                                }

                                if (studentMap == null) continue;

                                String status = (String) studentMap.get("status");
                                Timestamp ts = null;
                                Object tsObj = studentMap.get("timestamp");
                                if (tsObj instanceof Timestamp) {
                                    ts = (Timestamp) tsObj;
                                }

                                String timeStr = "";
                                if (ts != null) {
                                    SimpleDateFormat df = new SimpleDateFormat(
                                            "hh:mm a", Locale.getDefault());
                                    timeStr = df.format(ts.toDate());
                                }

                                // Build final display status WITHOUT UUID/session id
                                String displayStatus = (status == null) ? "" : status;
                                if (!timeStr.isEmpty()) {
                                    displayStatus = displayStatus + " (" + timeStr + ")";
                                }

                                historyRows.add(new AttendanceRow(date, displayStatus));
                            }
                        }

                        pending[0]--;
                        if (pending[0] <= 0) {
                            finalizeHistory();
                        }
                    })
                    .addOnFailureListener(e -> {
                        Log.e(TAG, "Attendance query failed: " + e.getMessage());
                        pending[0]--;
                        if (pending[0] <= 0) {
                            finalizeHistory();
                        }
                    });
        }
    }

    private void finalizeHistory() {
        progressHistory.setVisibility(View.GONE);

        if (historyRows.isEmpty()) {
            tvNoData.setVisibility(View.VISIBLE);
            tableHeader.setVisibility(View.GONE);
        } else {
            tvNoData.setVisibility(View.GONE);
            tableHeader.setVisibility(View.VISIBLE);
        }

        historyAdapter.notifyDataSetChanged();
    }

    // ----------------------------------------------------
    // MODEL + ADAPTER  (Date + Status ONLY)
    // ----------------------------------------------------
    private static class AttendanceRow {
        final String date;
        final String status;

        AttendanceRow(String date, String status) {
            this.date = date;
            this.status = status == null ? "" : status;
        }
    }

    private static class HistoryAdapter
            extends RecyclerView.Adapter<HistoryAdapter.HistoryViewHolder> {

        private final List<AttendanceRow> items;

        HistoryAdapter(List<AttendanceRow> items) {
            this.items = items;
        }

        @NonNull
        @Override
        public HistoryViewHolder onCreateViewHolder(@NonNull android.view.ViewGroup parent,
                                                    int viewType) {
            View view = android.view.LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_attendance_row, parent, false);
            return new HistoryViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull HistoryViewHolder holder,
                                     int position) {
            AttendanceRow row = items.get(position);
            holder.tvDate.setText(row.date);
            holder.tvStatus.setText(row.status);
        }

        @Override
        public int getItemCount() {
            return items.size();
        }

        static class HistoryViewHolder extends RecyclerView.ViewHolder {
            TextView tvDate, tvStatus;

            HistoryViewHolder(@NonNull View itemView) {
                super(itemView);
                tvDate = itemView.findViewById(R.id.tvDate);
                tvStatus = itemView.findViewById(R.id.tvStatus);
            }
        }
    }
}
