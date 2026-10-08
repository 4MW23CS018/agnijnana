import React from 'react';
import WheelInspection from '../../components/WheelInspection/WheelInspection';
import DefectViewer from '../../components/DefectViewer/DefectViewer';
import SeverityCard from '../../components/SeverityCard/SeverityCard';
import RootCause from '../../components/RootCause/RootCause';
import RiskPrediction from '../../components/RiskPrediction/RiskPrediction';
import Recommendation from '../../components/Recommendation/Recommendation';
import Alerts from '../../components/Alerts/Alerts';
import BatchTable from '../../components/BatchTable/BatchTable';
import MachineStatus from '../../components/MachineStatus/MachineStatus';

export default function Dashboard() {
  return (
    <div className="dashboard-layout">
      {/* Interactive Wheel Inspection Trigger */}
      <WheelInspection />

      {/* Visual Inspection & Severity */}
      <DefectViewer />
      <SeverityCard />

      {/* Root Cause & Future Risk */}
      <RootCause />
      <RiskPrediction />

      {/* Corrective Action & Alerts */}
      <Recommendation />
      <Alerts />

      {/* Batch & Machine Information */}
      <BatchTable />
      <MachineStatus />
    </div>
  );
}
