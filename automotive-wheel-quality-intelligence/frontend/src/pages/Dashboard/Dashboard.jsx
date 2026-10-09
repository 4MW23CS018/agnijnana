
import React, { useState } from 'react';
import WheelInspection from '../../components/WheelInspection/WheelInspection';
import SystemStatus from '../../components/SystemStatus/SystemStatus';
import SeverityCard from '../../components/SeverityCard/SeverityCard';
import RootCause from '../../components/RootCause/RootCause';
import RiskPrediction from '../../components/RiskPrediction/RiskPrediction';
import Recommendation from '../../components/Recommendation/Recommendation';
import Alerts from '../../components/Alerts/Alerts';

export default function Dashboard() {
  const [inspectionResult, setInspectionResult] = useState(null);

  return (
    <div className="dashboard-layout">
      <WheelInspection onInspectionComplete={setInspectionResult} />

      <SystemStatus />

      <SeverityCard inspectionResult={inspectionResult} />
      <RootCause />
      <RiskPrediction />
      <Recommendation />
      <Alerts />
    </div>
  );
}
