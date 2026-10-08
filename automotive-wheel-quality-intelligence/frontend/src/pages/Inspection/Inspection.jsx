import React from 'react';
import DefectViewer from '../../components/DefectViewer/DefectViewer';
import SeverityCard from '../../components/SeverityCard/SeverityCard';

export default function Inspection() {
  return (
    <div className="dashboard-layout">
      <DefectViewer />
      <SeverityCard />
    </div>
  );
}
