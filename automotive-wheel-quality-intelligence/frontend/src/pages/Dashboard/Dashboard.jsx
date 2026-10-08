import React from 'react';
import WheelInspection from '../../components/WheelInspection/WheelInspection';
import SystemStatus from '../../components/SystemStatus/SystemStatus';
import SeverityCard from '../../components/SeverityCard/SeverityCard';
import RootCause from '../../components/RootCause/RootCause';
import RiskPrediction from '../../components/RiskPrediction/RiskPrediction';
import Recommendation from '../../components/Recommendation/Recommendation';
import Alerts from '../../components/Alerts/Alerts';

export default function Dashboard() {
  return (
    <div className="dashboard-layout">
      {/* Primary Visual Inspection & Live CNN Classifier */}
      <WheelInspection />

      {/* System Operational Status */}
      <SystemStatus />

      {/* Quality Intelligence Modules (Pending Roadmap) */}
      <SeverityCard />
      <RootCause />
      <RiskPrediction />
      <Recommendation />
      <Alerts />
    </div>
  );
}
