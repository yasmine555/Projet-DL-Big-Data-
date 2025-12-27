import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getPatientById, getPatientMRIScans, getPatientResults } from "../../../services/api";
import Markdown from 'react-markdown';

function Results() {
    const { patientId } = useParams();
    const navigate = useNavigate();
    const [patient, setPatient] = useState(null);
    const [mriScans, setMriScans] = useState([]);
    const [aiSummary, setAiSummary] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                setError(null);

                // Get patient ID from URL params or sessionStorage
                const id = patientId || sessionStorage.getItem("currentPatientId");

                if (!id) {
                    setError("No patient ID found");
                    setLoading(false);
                    return;
                }

                const userType = localStorage.getItem('userType') || 'patient';
                let summaryData = null;

                // For patients (anonymous users), load from sessionStorage
                // For doctors, fetch from API
                if (userType === 'patient') {
                    // Load patient context from session storage
                    const storedContext = sessionStorage.getItem('patient_context');
                    if (storedContext) {
                        try {
                            setPatient(JSON.parse(storedContext));
                        } catch (e) {
                            console.error("Bad patient context", e);
                        }
                    }
                } else {
                    // Doctor view - fetch from API
                    try {
                        const patientData = await getPatientById(id);
                        setPatient(patientData);

                        // Fetch MRI scans from API
                        try {
                            const mriData = await getPatientMRIScans(id);
                            setMriScans(mriData.scans || []);
                        } catch (err) {
                            console.error("Error fetching MRI scans:", err);
                            setMriScans([]);
                        }

                        // Fetch AI summary/results from API
                        try {
                            const resultsData = await getPatientResults(id);
                            console.log("API Results Data:", resultsData);
                            if (resultsData && resultsData.length > 0) {
                                summaryData = resultsData[0];
                                console.log("Using API summary data:", summaryData);
                            }
                        } catch (err) {
                            console.error("Error fetching AI summary:", err);
                        }
                    } catch (err) {
                        console.error("Error fetching patient data:", err);
                        setError(err.message || "Failed to load patient data");
                        setLoading(false);
                        return;
                    }
                }

                if (!summaryData) {
                    try {
                        // Only try API if not temp (redundant check but safe)
                        if (!isTempPatient) {
                            const resultsData = await getPatientResults(id);
                            console.log("API Results Data:", resultsData);
                            if (resultsData && resultsData.length > 0) {
                                summaryData = resultsData[0];
                                console.log("Using API summary data:", summaryData);
                            }
                        }
                    } catch (err) {
                        console.error("Error fetching AI summary:", err);
                    }
                }

                if (!summaryData) {
                    const cached = sessionStorage.getItem("questionnaire_result");
                    console.log("Session Storage Cached Result:", cached);
                    if (cached) {
                        try {
                            const parsed = JSON.parse(cached);
                            summaryData = parsed;
                            console.log("Using SessionStorage summary data:", summaryData);
                        } catch (e) {
                            console.error("Error parsing cached result", e);
                        }
                    }
                }

                if (summaryData) {
                    setAiSummary(summaryData);
                } else {
                    console.warn("No AI summary data found from API or SessionStorage");
                }

                setLoading(false);
            } catch (err) {
                console.error("Error fetching patient data:", err);
                setError(err.message || "Failed to load patient data");
                setLoading(false);
            }
        };

        fetchData();
    }, [patientId]);

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-xl text-gray-600">Loading patient data...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-xl text-red-600">Error: {error}</div>
            </div>
        );
    }

    if (!patient) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-xl text-gray-600">No patient data found</div>
            </div>
        );
    }

    // Get the latest MRI scan
    const latestMRI = mriScans.length > 0 ? mriScans[0] : null;
    const baseUrl = import.meta.env?.VITE_API_BASE_URL?.replace('/api', '') || 'http://localhost:7000';

    const mriImageUrl = latestMRI?.image_url
        ? `${baseUrl}${latestMRI.image_url}`
        : patient?.mri_image_url
            ? `${baseUrl}${patient.mri_image_url}`
            : null;

    const mriXaiUrl = latestMRI?.xai_url
        ? `${baseUrl}${latestMRI.xai_url}`
        : patient?.mri_xai_url
            ? `${baseUrl}${patient.mri_xai_url}`
            : null;

    // Get user type for view customization
    const userType = localStorage.getItem('userType') || 'patient';

    const handleOpenChatbot = () => {
        const id = patientId || sessionStorage.getItem("currentPatientId");
        if (id) {
            if (userType === 'doctor') {
                navigate(`/doctor/patient/result/chatbot/${id}`);
            } else {
                navigate(`/patient/result/chatbot/${id}`);
            }
        }
    };

    return (
        <div className="min-h-screen relative flex justify-center gap-10">


            {/* Patient Information Panel */}
            <div className={`bg-blue-50/50 overflow-y-auto rounded-lg h-[82vh] my-10 shadow-lg px-4 text-center ${userType === 'patient' ? 'w-1/4' : 'w-1/5 ml-10'}`}>
                <div className="px-4 sm:px-3 py-1">
                    <h3 className="leading-3 font-medium text-blue-500">
                        Patient Information
                    </h3>
                </div>

                {/* Basic Information */}
                <div className="sm:px-3">
                    <h5 className=" leading-6 font-medium text-indigo-600 ">Basic</h5>
                </div>
                <div className="px-2 py-1 sm:p-0">
                    <dl className="sm:divide-gray-200">
                        <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-2">
                            <dt className="text-sm font-medium text-gray-500">Full name</dt>
                            <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">{patient.name || "N/A"}</dd>
                        </div>
                        <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-2">
                            <dt className="text-sm font-medium text-gray-500">Age</dt>
                            <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">{patient.age || "N/A"}</dd>
                        </div>
                        <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-2">
                            <dt className="text-sm font-medium text-gray-500">Gender</dt>
                            <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">{patient.sex || "N/A"}</dd>
                        </div>
                        <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-2">
                            <dt className="text-sm font-medium text-gray-500">Family History</dt>
                            <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2 text-left">
                                {patient.family_history || "N/A"}
                            </dd>
                        </div>

                        {/* Symptoms */}
                        <div className="sm:px-3 mt-4">
                            <h5 className=" leading-6 font-medium text-indigo-600">
                                Symptoms
                            </h5>
                        </div>
                        <div className="sm:px-3 py-1">
                            <div className="text-sm text-gray-700 text-left">
                                {patient.metrics?.motorSymptoms && patient.metrics.motorSymptoms.length > 0
                                    ? patient.metrics.motorSymptoms.join(", ")
                                    : patient.symptoms_list && patient.symptoms_list.length > 0
                                        ? patient.symptoms_list.join(", ")
                                        : "No symptoms recorded"}
                            </div>
                            {patient.metrics?.sleepIssues && (
                                <div className="text-sm text-gray-700 text-left mt-1">
                                    <strong>Sleep Issues:</strong> Yes
                                </div>
                            )}
                        </div>

                        {/* Bio Markers - Only for Doctors */}
                        {userType === 'doctor' && (
                            <>
                                <div className="sm:px-3 mt-4">
                                    <h5 className="leading-6 font-medium text-gray-900">
                                        Bio Markers
                                    </h5>
                                </div>
                                <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-1">
                                    <dt className="text-sm font-medium text-gray-500">MMSE</dt>
                                    <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                                        {patient.mmse_score ?? patient.mse ?? "N/A"}
                                    </dd>
                                </div>
                                <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-1">
                                    <dt className="text-sm font-medium text-gray-500">MoCA</dt>
                                    <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                                        {patient.moca_score ?? patient.moca ?? "N/A"}
                                    </dd>
                                </div>
                                <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-1">
                                    <dt className="text-sm font-medium text-gray-500">Amyloid</dt>
                                    <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                                        {patient.metrics?.biomarkers?.amyloid ?? patient.biomarkers?.amyloid ?? "N/A"}
                                    </dd>
                                </div>
                                <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-1">
                                    <dt className="text-sm font-medium text-gray-500">TAU</dt>
                                    <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                                        {patient.metrics?.biomarkers?.tau ?? patient.biomarkers?.tau ?? "N/A"}
                                    </dd>
                                </div>
                                <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-1">
                                    <dt className="text-sm font-medium text-gray-500">Apoe4</dt>
                                    <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                                        {patient.metrics?.biomarkers?.apoe4 ?? patient.biomarkers?.apoe4 ?? "N/A"}
                                    </dd>
                                </div>
                            </>
                        )}

                        {/* Life Style */}
                        <div className="sm:px-3 mt-4">
                            <h5 className=" leading-6 font-medium text-indigo-600">
                                Life Style
                            </h5>
                        </div>
                        <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-1">
                            <dt className="text-sm font-medium text-gray-500">
                                Physical Activity
                            </dt>
                            <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                                {patient.metrics?.lifestyle?.physicalActivity ?? patient.metrics?.physical_activity ?? "N/A"}
                            </dd>
                        </div>
                        <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-1">
                            <dt className="text-sm font-medium text-gray-500">Smoking</dt>
                            <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                                {patient.metrics?.lifestyle?.smokingStatus ?? patient.metrics?.smoking ?? "N/A"}
                            </dd>
                        </div>
                        <div className="sm:grid sm:grid-cols-3 sm:gap-2 sm:px-3 py-1">
                            <dt className="text-sm font-medium text-gray-500">
                                Alcohol Consumption
                            </dt>
                            <dd className="text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                                {patient.metrics?.lifestyle?.alcoholConsumption ?? patient.metrics?.alcohol_consumption ?? "N/A"}
                            </dd>
                        </div>
                    </dl>
                </div>
            </div>

            {/* MRI and Prediction Section - Only for Doctors */}
            {userType === 'doctor' && (
                <div className="flex flex-col w-[50%] h-[80vh] mt-10 mr-20">
                    {/* MRI Image Card */}
                    <div className="w-full  h-1/2 lg:max-w-full lg:flex shadow-lg rounded-lg overflow-hidden bg-white my-10 ml-10">
                        {mriImageUrl ? (
                            <div
                                className=" m-2 h-full w-full lg:h-auto lg:w-80 flex-none bg-cover bg-center rounded-t lg:rounded-t-none lg:rounded-l"
                                style={{ backgroundImage: `url('${mriImageUrl}')` }}
                                title="MRI Scan"
                            ></div>
                        ) : (
                            <div className="h-full lg:h-auto lg:w-80 flex-none bg-gray-200 rounded-t lg:rounded-t-none lg:rounded-l flex items-center justify-center">
                                <span className="text-gray-500">No MRI Image</span>
                            </div>
                        )}
                        <div className=" bg-white rounded-b lg:rounded-b-none lg:rounded-r  p-4 flex flex-col  justify-between leading-normal">
                            <div className="mb-8">
                                <div className="text-gray-900 font-bold text-xl mb-2">
                                    MRI Diagnosis Prediction
                                </div>
                                {latestMRI ? (
                                    <div className="space-y-4">


                                        {/* All Class Probabilities */}
                                        {latestMRI.probabilities && (
                                            <div>
                                                <div className="mb-3">
                                                    <strong>Confidence:</strong> {(latestMRI.confidence * 100).toFixed(2)}%
                                                </div>
                                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                                                    {Object.entries(latestMRI.probabilities)
                                                        .sort((a, b) => b[1] - a[1]) // Sort by probability descending
                                                        .map(([className, probability]) => {
                                                            const percentage = (probability * 100).toFixed(2);
                                                            const isTopPrediction = className === latestMRI.prediction_class;
                                                            return (
                                                                <div
                                                                    key={className}
                                                                    className={`p-3 rounded-lg border ${isTopPrediction
                                                                        ? 'bg-indigo-100 border-indigo-300'
                                                                        : 'bg-gray-50 border-gray-200'
                                                                        }`}
                                                                >
                                                                    <div className="flex justify-between items-center mb-1">
                                                                        <span className={`font-medium ${isTopPrediction ? 'text-indigo-900' : 'text-gray-700'
                                                                            }`}>
                                                                            {className}
                                                                        </span>
                                                                        <span className={`text-sm font-semibold ${isTopPrediction ? 'text-indigo-700' : 'text-gray-600'
                                                                            }`}>
                                                                            {percentage}%
                                                                        </span>
                                                                    </div>
                                                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                                                        <div
                                                                            className={`h-2 rounded-full ${isTopPrediction ? 'bg-indigo-600' : 'bg-gray-400'
                                                                                }`}
                                                                            style={{ width: `${percentage}%` }}
                                                                        ></div>
                                                                    </div>
                                                                </div>
                                                            );
                                                        })}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <p className="text-gray-700">
                                        Waiting for prediction results...
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* XAI Explanation Card */}
                    <div className="w-2/3 w-full h-1/2 lg:max-w-full lg:flex shadow-lg rounded-lg overflow-hidden bg-white my-10 ml-10">
                        {mriXaiUrl ? (
                            <div
                                className=" m-2 h-full w-full lg:h-auto lg:w-80 flex-none bg-cover bg-center rounded-t lg:rounded-t-none lg:rounded-l"
                                style={{ backgroundImage: `url('${mriXaiUrl}')` }}
                                title="XAI Visualization"
                            ></div>
                        ) : (
                            <div className=" m-2 h-full w-full lg:h-auto lg:w-48 flex-none bg-cover bg-center rounded-t lg:rounded-t-none lg:rounded-l bg-gradient-to-br from-blue-400 to-purple-500" title="XAI Placeholder"></div>
                        )}
                        <div className="bg-white rounded-b lg:rounded-b-none lg:rounded-r p-4 flex flex-col justify-between leading-normal">
                            <div className="mb-8">
                                <div className="text-gray-900 font-bold text-xl mb-2">
                                    XAI (Grad-CAM) Interpretation
                                </div>
                                {mriXaiUrl ? (
                                    <p className="text-gray-700 text-sm">
                                        The heatmap above highlights the specific regions of the brain that most influenced the AI's prediction.
                                        <strong> Warm colors (Red/Yellow)</strong> indicate areas of high diagnostic importance.
                                        This visualization allows clinicians to verify that the model is focusing on relevant pathological markers rather than irrelevant image noise.
                                    </p>
                                ) : (
                                    <p className="text-gray-700 text-base">
                                        Explainable AI visualization will be generated once an MRI scan is uploaded and analyzed by our VGG16 model.
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* AI Summary Panel */}
            <div className={`bg-blue-50/50 overflow-y-auto rounded-lg h-[80vh] my-10 shadow-lg px-4 text-left ${userType === 'patient' ? 'w-1/2' : 'w-1/5 mr-10 end-0'}`}>
                <div className="px-4 sm:px-3 py-4 text-center">
                    <h3 className="text-lg leading-6 font-medium text-blue-500">
                        {userType === 'patient' ? "Health Insights" : "AI Generated Summary and Insights"}
                    </h3>
                </div>
                <div className="px-4 py-2 text-sm text-gray-700 markdown-container">
                    <Markdown>
                        {aiSummary?.summary || aiSummary?.answer || "No AI summary available yet. Complete the questionnaire to generate insights."}
                    </Markdown>
                </div>


            </div>
        </div>
    );
}

export default Results;