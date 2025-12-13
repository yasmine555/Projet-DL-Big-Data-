import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getPatientById, getPatientMRIScans, getPatientResults } from "../../../services/api";

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

                // Fetch patient data
                const patientData = await getPatientById(id);
                setPatient(patientData);

                // Fetch MRI scans
                try {
                    const mriData = await getPatientMRIScans(id);
                    setMriScans(mriData.scans || []);
                } catch (err) {
                    console.error("Error fetching MRI scans:", err);
                    setMriScans([]);
                }

                // Fetch AI summary/results
                try {
                    const resultsData = await getPatientResults(id);
                    if (resultsData && resultsData.length > 0) {
                        setAiSummary(resultsData[0]);
                    }
                } catch (err) {
                    console.error("Error fetching AI summary:", err);
                    setAiSummary(null);
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
    const mriImageUrl = latestMRI?.image_url
        ? `${import.meta.env?.VITE_API_BASE_URL?.replace('/api', '') || 'http://localhost:7000'}${latestMRI.image_url}`
        : patient.mri_image_url
            ? `${import.meta.env?.VITE_API_BASE_URL?.replace('/api', '') || 'http://localhost:7000'}${patient.mri_image_url}`
            : null;

    const handleOpenChatbot = () => {
        const id = patientId || sessionStorage.getItem("currentPatientId");
        if (id) {
            navigate(`/doctor/patient/results/chatbot/${id}`);
        }
    };

    return (
        <div className="min-h-screen relative flex">
        

            {/* Patient Information Panel */}
            <div className="bg-white overflow-y-auto rounded-lg w-1/5 h-[82vh] my-10 ml-10 shadow-lg px-4 text-center">
                <div className="px-4 sm:px-3 py-1">
                    <h3 className="leading-3 font-medium text-gray-900">
                        Patient Information
                    </h3>
                </div>

                {/* Basic Information */}
                <div className="sm:px-3">
                    <h5 className=" leading-6 font-medium text-gray-900">Basic</h5>
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
                            <h5 className=" leading-6 font-medium text-gray-900">
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

                        {/* Bio Markers */}
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

                        {/* Life Style */}
                        <div className="sm:px-3 mt-4">
                            <h5 className=" leading-6 font-medium text-gray-900">
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

            {/* MRI and Prediction Section */}
            <div className="flex flex-col w-[50%] h-[80vh] mt-10 mr-20">
                {/* MRI Image Card */}
                <div className="max-w-md w-full h-1/2 lg:max-w-full lg:flex shadow-lg rounded-lg overflow-hidden bg-white my-10 ml-10">
                    {mriImageUrl ? (
                        <div
                            className="h-full w-[70%] lg:h-auto lg:w-48 flex-none bg-cover bg-center rounded-t lg:rounded-t-none lg:rounded-l"
                            style={{ backgroundImage: `url('${mriImageUrl}')` }}
                            title="MRI Scan"
                        ></div>
                    ) : (
                        <div className="h-full w-[70%] lg:h-auto lg:w-48 flex-none bg-gray-200 rounded-t lg:rounded-t-none lg:rounded-l flex items-center justify-center">
                            <span className="text-gray-500">No MRI Image</span>
                        </div>
                    )}
                    <div className="bg-white rounded-b lg:rounded-b-none lg:rounded-r p-4 flex flex-col justify-between leading-normal">
                        <div className="mb-8">
                            <div className="text-gray-900 font-bold text-xl mb-2">
                                MRI Diagnosis Prediction
                            </div>
                            <p className="text-gray-700 text-base">
                                {latestMRI ? (
                                    <>
                                        <strong>Prediction:</strong> {latestMRI.prediction_class}<br />
                                        <strong>Confidence:</strong> {(latestMRI.confidence * 100).toFixed(2)}%<br />
                                        <strong>Date:</strong> {new Date(latestMRI.created_at).toLocaleDateString()}
                                    </>
                                ) : (
                                    "Waiting for prediction results from your friend's work..."
                                )}
                            </p>
                        </div>
                    </div>
                </div>

                {/* XAI Explanation Card */}
                <div className="max-w-md w-full h-1/2 lg:max-w-full lg:flex shadow-lg rounded-lg overflow-hidden bg-white my-10 ml-10">
                    <div
                        className="h-full w-[70%] lg:h-auto lg:w-48 flex-none bg-cover bg-center rounded-t lg:rounded-t-none lg:rounded-l bg-gradient-to-br from-blue-400 to-purple-500"
                        title="XAI Visualization"
                    ></div>
                    <div className="bg-white rounded-b lg:rounded-b-none lg:rounded-r p-4 flex flex-col justify-between leading-normal">
                        <div className="mb-8">
                            <div className="text-gray-900 font-bold text-xl mb-2">
                                XAI Detailed Explanation
                            </div>
                            <p className="text-gray-700 text-base">
                                Explainable AI visualization and detailed prediction explanation will be available once the prediction model is ready.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* AI Summary Panel */}
            <div className="bg-white overflow-y-auto rounded-lg w-1/5 h-[80vh] my-10 mr-10 end-0 shadow-lg px-4 text-left">
                <div className="px-4 sm:px-3 py-4 text-center">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                        AI Generated Summary and Insights
                    </h3>
                </div>
                <div className="px-4 py-2 text-sm text-gray-700 whitespace-pre-wrap">
                    {aiSummary?.summary || aiSummary?.answer || "No AI summary available yet. Complete the questionnaire to generate insights."}
                </div>
            </div>
        </div>
    );
}

export default Results;