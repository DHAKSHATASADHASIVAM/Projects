# Projects

A collection of my Data Science and Software Engineering projects.

---

### [Sustainable Ride Suggestion](Sustainable_ride_Suggestion/)

A predictive recommender that estimates **travel time, cost and lifecycle CO₂** for taxi, bike and e-scooter trips in New York City, then ranks them against a traveller's own trade-off between money, time and carbon.

Trained on real public data — 2.96M NYC TLC yellow-cab records and 1.89M Citi Bike rides. Gradient-boosted regressors reach an MAE of 2.98 min on taxi duration (R² 0.762) and $2.32 on fare (R² 0.917). Ranks modes using a Pareto frontier before applying user weights, so the defensible options are separated from the preference-dependent ones.

Includes a FastAPI service, a Streamlit demo, an executed EDA notebook and 177 tests.

`Python` · `scikit-learn` · `pandas` · `FastAPI` · `Streamlit` · `pytest`

---

### [RAG POML Chatbot](RAG_POML_Chatbot/)

A code-documentation Q&A chatbot built on a retrieval-augmented generation pipeline. Answers questions about **Python, Java and JavaScript** by retrieving from documentation PDFs.

A POML prompt routes each question to the right language, ChromaDB retrieves the relevant chunks, and the answer is generated through LangChain. Response quality is assessed with LLM-as-a-judge evaluation.

`Python` · `LangChain` · `ChromaDB` · `POML` · `Groq` · `sentence-transformers`

---

### [Smart Attendance – Student App](Smart_Attendance_Student/)

An Android application that records class attendance through **on-device face recognition**, using a FaceNet model running under TensorFlow Lite so face embeddings never leave the phone. Location services verify the student is physically present.

Authentication and attendance records are backed by Firebase Auth and Firestore.

`Java` · `Android` · `TensorFlow Lite` · `FaceNet` · `Firebase`
